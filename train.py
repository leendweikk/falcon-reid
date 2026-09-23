import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from reid import evaluate as official                 # the organizers' evaluate.py
from reid.data import TrainSet, PKSampler, CROPS, make_transforms
from reid.losses import ReIDLoss
from reid.model import ReIDModel

ROOT = Path(__file__).resolve().parent

# ----------------------------- settings -----------------------------
# Validation run of the final recipe on the seed-42 split (the split all our comparisons use).
#   python train.py --model base  --run-name val_base_ema    (paper trail: runs/convnext_b_v6_ema, EMA ep15 = 0.7554)
#   python train.py --model small --run-name val_small_ema   (paper trail: runs/convnext_s_v2_ema, EMA ep15 = 0.7306)
#   python train.py --model base --size 320 --run-name val_base_320   (Experiment: bigger input)
BACKBONES = {"base": "convnext_base.dinov3_lvd1689m", "small": "convnext_small.dinov3_lvd1689m"}
HEAD = "linear"                             # "cosface" was tested and rejected (overfits)
TRIPLET_MARGIN = None                       # None = soft-margin triplet (kept); 0.3 = classic
SPLITS = ROOT / "data" / "splits"
TRAIN_CSVS = [SPLITS / "train_split.csv"]   # ONLY the organizers' data (VeRi rejected: visible plates)
EMA_DECAY = 0.998                           # smoothed copy ~ average of the last ~500 steps
EPOCHS = 30                                 # length of the LR schedule
STOP_EPOCH = 20                             # EMA peaks ~15 (plateau 10-20); final models use the EMA at 15
WARMUP_EPOCHS = 3
LR_BACKBONE, LR_HEAD = 1e-4, 1e-3
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 5
# --------------------------------------------------------------------

torch.backends.cudnn.benchmark = True       # free speedup for fixed-size images


@torch.no_grad()
def embed(model, image_ids, test_transform, batch_size=64):
    model.eval()
    feats = []
    for i in range(0, len(image_ids), batch_size):
        batch = torch.stack([test_transform(Image.open(CROPS / f"{iid}.jpg").convert("RGB"))
                             for iid in image_ids[i:i + batch_size]]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(batch) + model(torch.flip(batch, dims=[3]))     # flip averaging
        feats.append(f.float().cpu())
    model.train()
    return torch.nn.functional.normalize(torch.cat(feats), dim=1).numpy()


def validate(model, q_ids, g_ids, gt_query, gt_gallery, test_transform):
    q_emb, g_emb = embed(model, q_ids, test_transform), embed(model, g_ids, test_transform)
    order = np.argsort(-(q_emb @ g_emb.T), axis=1)[:, :10]
    ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
    return official.ranking_metrics(gt_query, gt_gallery, ranked)   # official scoring code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=BACKBONES, default="base")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--size", type=int, default=256, help="square input size (256 = proven)")
    args = ap.parse_args()
    BACKBONE = BACKBONES[args.model]
    OUT = ROOT / "runs" / (args.run_name or f"val_{args.model}_ema")

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    input_hw = (args.size, args.size)
    train_tf, test_tf = make_transforms(input_hw)
    ds = TrainSet(TRAIN_CSVS, transform=train_tf)
    print(f"training on {ds.num_classes} cars, {len(ds)} photos | input (h, w) = {input_hw}")
    loader = DataLoader(ds, batch_sampler=PKSampler(ds, P=16, K=4),
                        num_workers=4, pin_memory=True, persistent_workers=True)

    model = ReIDModel(num_classes=ds.num_classes, backbone=BACKBONE, head=HEAD).cuda().train()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)
    loss_fn = ReIDLoss(margin=TRIPLET_MARGIN, smoothing=0.1)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": list(model.bnneck.parameters()) + list(model.classifier.parameters()), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(loader)
    total_steps, warmup_steps = EPOCHS * steps_per_epoch, WARMUP_EPOCHS * steps_per_epoch

    def lr_factor(step):                    # warmup, then cosine decay
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")

    q_ids = pd.read_csv(SPLITS / "val_query.csv", dtype={"image_id": str}).image_id.tolist()
    g_ids = pd.read_csv(SPLITS / "val_gallery.csv", dtype={"image_id": str}).image_id.tolist()
    gt_query, gt_gallery = official.load_gt(SPLITS / "val_gt.csv")

    best = {"normal": 0.0, "ema": 0.0}
    log = open(OUT / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "loss", "id_loss", "tri_loss", "mAP@10", "Rank-1", "Rank-5",
                     "EMA_mAP@10", "EMA_Rank-1", "EMA_Rank-5", "seconds"])

    for epoch in range(1, STOP_EPOCH + 1):
        start, sums = time.time(), np.zeros(3)
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                feat, logits = model(imgs, labels)
            loss, id_l, tri_l = loss_fn(feat, logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update_parameters(model)                    # update the smoothed copy
            sums += [loss.item(), id_l, tri_l]

        avg = sums / steps_per_epoch
        secs = time.time() - start
        row = [epoch, *np.round(avg, 4), "", "", "", "", "", "", round(secs)]
        msg = f"epoch {epoch:2d} | loss {avg[0]:.3f} (id {avg[1]:.3f}, tri {avg[2]:.3f}) | {secs:.0f}s"

        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == STOP_EPOCH:
            for kind, net, col in [("normal", model, 4), ("ema", ema.module, 7)]:
                m = validate(net, q_ids, g_ids, gt_query, gt_gallery, test_tf)
                row[col:col + 3] = [round(m["mAP@10"], 4), round(m["Rank-1"], 4), round(m["Rank-5"], 4)]
                msg += f" | {kind} mAP@10 {m['mAP@10']:.4f}"
                if m["mAP@10"] > best[kind]:
                    best[kind] = m["mAP@10"]
                    torch.save(net.state_dict(), OUT / f"best_{kind}.pth")
                    msg += " *"

        torch.save(model.state_dict(), OUT / "last.pth")          # crash insurance, every epoch
        torch.save(ema.module.state_dict(), OUT / "last_ema.pth")
        writer.writerow(row); log.flush()
        print(msg)

    print(f"done. best normal = {best['normal']:.4f} | best EMA = {best['ema']:.4f} "
          f"(reference: Base EMA ep15 0.7554, Small EMA ep15 0.7306; seed-to-seed noise ~0.7)")


if __name__ == "__main__":
    main()