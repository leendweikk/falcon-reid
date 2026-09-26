"""
Final training recipe (fixed by validation):
  - 30-epoch LR schedule (3 warmup + cosine), stopped at epoch 15
  - EMA (decay 0.998) copy at epoch 15 is saved as the model

  python train_final.py --model base     -> weights/base.pth   (all of train.csv)
  python train_final.py --model small    -> weights/small.pth
  optional: --train-csv <csv> --out <dir> --seed <n> --size <n>
  The saved file records its input size ("input_size"), so inference always uses the right size.
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from reid.data import TrainSet, PKSampler, STRONG_LIGHT_AUG, make_transforms
from reid.losses import RelationalKD, ReIDLoss
from reid.model import ReIDModel
from reid.optim import param_groups

ROOT = Path(__file__).resolve().parent
BACKBONES = {"base": "convnext_base.dinov3_lvd1689m", "small": "convnext_small.dinov3_lvd1689m",
             "vit": "vit_base_patch16_dinov3.lvd1689m"}
SCHEDULE_EPOCHS, STOP_EPOCH, WARMUP_EPOCHS = 30, 15, 3
LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, EMA_DECAY = 1e-4, 1e-3, 1e-4, 0.998

torch.backends.cudnn.benchmark = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=BACKBONES, required=True)
    ap.add_argument("--train-csv", nargs="+", default=[str(ROOT / "data" / "train.csv")],
                    help="one or more CSVs (e.g. our split + data/veri.csv)")
    ap.add_argument("--out", default=str(ROOT / "weights"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-backbone", type=float, default=LR_BACKBONE)
    ap.add_argument("--epochs", type=int, default=SCHEDULE_EPOCHS, help="length of the LR schedule")
    ap.add_argument("--stop-epoch", type=int, default=STOP_EPOCH)
    ap.add_argument("--pool", choices=["avg", "gem"], default="avg", help="gem = generalized mean (ConvNeXt only)")
    ap.add_argument("--P", type=int, default=16, help="cars per batch")
    ap.add_argument("--K", type=int, default=4, help="photos per car")
    ap.add_argument("--camera-aware", action="store_true", help="spread each car's K photos over cameras (answer #2)")
    ap.add_argument("--llrd", type=float, default=1.0, help="layer-wise LR decay for ViT (e.g. 0.75); 1 = off")
    ap.add_argument("--size", type=int, default=256, help="square input size (256 = proven)")
    ap.add_argument("--distill", default=None, help="teacher vectors .npz from tools/teacher_embed.py (A11)")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--distill-tau", type=float, default=0.1)
    args = ap.parse_args()

    assert not STRONG_LIGHT_AUG, "strong light augmentation was rejected: set it to False in data.py"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.model if args.seed == 0 else f"{args.model}_seed{args.seed}"

    train_tf, _ = make_transforms((args.size, args.size))
    ds = TrainSet([Path(p) for p in args.train_csv], transform=train_tf, return_index=bool(args.distill))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, kd_fn = None, None
    if args.distill:                                            # A11: relational distillation from the ensemble
        z = np.load(args.distill)
        row = {iid: j for j, iid in enumerate(z["image_ids"].tolist())}
        missing = [iid for iid, _ in ds.items if iid not in row]
        assert not missing, f"{len(missing)} training photos have no teacher vector, e.g. {missing[:3]}"
        teacher = torch.from_numpy(z["vectors"][[row[iid] for iid, _ in ds.items]]).to(device)
        kd_fn = RelationalKD(tau=args.distill_tau)
        print(f"[{name}] distillation: teacher {tuple(teacher.shape)} from {args.distill} "
              f"(weight {args.distill_weight}, tau {args.distill_tau})")
    print(f"[{name}] training on {ds.num_classes} cars, {len(ds)} photos, input {args.size}x{args.size} -> {out_dir}")
    loader = DataLoader(ds, batch_sampler=PKSampler(ds, P=args.P, K=args.K, camera_aware=args.camera_aware),
                        num_workers=4, pin_memory=True, persistent_workers=True)

    model = ReIDModel(num_classes=ds.num_classes, backbone=BACKBONES[args.model], pool=args.pool).to(device).train()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)
    loss_fn = ReIDLoss(margin=None, smoothing=0.1)                 # soft-margin triplet

    optimizer = torch.optim.AdamW(param_groups(model, args.lr_backbone, LR_HEAD, args.llrd),
                                  weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(loader)
    total_steps, warmup_steps = args.epochs * steps_per_epoch, WARMUP_EPOCHS * steps_per_epoch

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    for epoch in range(1, args.stop_epoch + 1):
        start, sums = time.time(), np.zeros(4)
        for batch in loader:
            imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                feat, logits = model(imgs, labels)
            loss, id_l, tri_l = loss_fn(feat, logits, labels)
            kd_l = 0.0
            if teacher is not None:
                kd = kd_fn(model.last_feat_bn, teacher[batch[2].to(device)])
                loss = loss + args.distill_weight * kd
                kd_l = kd.item()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update_parameters(model)
            sums += [loss.item(), id_l, tri_l, kd_l]
        avg = sums / steps_per_epoch
        torch.save(ema.module.state_dict(), out_dir / f"{name}_ema_last.pth")   # crash insurance
        print(f"[{name}] epoch {epoch:2d}/{args.stop_epoch} | loss {avg[0]:.3f} "
              f"(id {avg[1]:.3f}, tri {avg[2]:.3f}" + (f", kd {avg[3]:.3f}" if teacher is not None else "")
              + f") | {time.time() - start:.0f}s")

    state = ema.module.state_dict()
    state["input_size"] = torch.tensor(args.size)              # inference reads this
    torch.save(state, out_dir / f"{name}.pth")
    print(f"[{name}] saved final EMA model -> {out_dir / (name + '.pth')}")


if __name__ == "__main__":
    main()