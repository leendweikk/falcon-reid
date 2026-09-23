import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

import evaluate as official                 # the organizers' evaluate.py
from data import TrainSet, PKSampler, test_transform, CROPS
from losses import ReIDLoss
from model import ReIDModel

# ----------------------------- settings -----------------------------
RUN_NAME = "convnext_b_v5_veri"
BACKBONE = "convnext_base.dinov3_lvd1689m"
HEAD = "linear"                             # "linear" or "cosface"
TRIPLET_MARGIN = None                       # None = soft-margin triplet, 0.3 = classic
SPLITS = Path("C:/falcon/data/splits")
TRAIN_CSVS = [SPLITS / "train_split.csv", SPLITS / "veri.csv"]   # our cars + VeRi
EPOCHS = 30
WARMUP_EPOCHS = 3
LR_BACKBONE, LR_HEAD = 1e-4, 1e-3
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 5
OUT = Path("C:/falcon/runs") / RUN_NAME
# --------------------------------------------------------------------


@torch.no_grad()
def embed(model, image_ids, batch_size=64):
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


def validate(model, q_ids, g_ids, gt_query, gt_gallery):
    q_emb, g_emb = embed(model, q_ids), embed(model, g_ids)
    order = np.argsort(-(q_emb @ g_emb.T), axis=1)[:, :10]
    ranked = {qid: [g_ids[j] for j in order[i]] for i, qid in enumerate(q_ids)}
    return official.ranking_metrics(gt_query, gt_gallery, ranked)   # official scoring code


def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    ds = TrainSet(TRAIN_CSVS)
    print(f"training on {ds.num_classes} cars, {len(ds)} photos")
    loader = DataLoader(ds, batch_sampler=PKSampler(ds, P=16, K=4),
                        num_workers=4, pin_memory=True, persistent_workers=True)

    model = ReIDModel(num_classes=ds.num_classes, backbone=BACKBONE, head=HEAD).cuda().train()
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

    best_map = 0.0
    log = open(OUT / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "loss", "id_loss", "tri_loss", "mAP@10", "Rank-1", "Rank-5", "seconds"])

    for epoch in range(1, EPOCHS + 1):
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
            sums += [loss.item(), id_l, tri_l]

        avg = sums / steps_per_epoch
        secs = time.time() - start
        row = [epoch, *np.round(avg, 4), "", "", "", round(secs)]
        msg = f"epoch {epoch:2d} | loss {avg[0]:.3f} (id {avg[1]:.3f}, tri {avg[2]:.3f}) | {secs:.0f}s"

        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            m = validate(model, q_ids, g_ids, gt_query, gt_gallery)
            row[4:7] = [round(m["mAP@10"], 4), round(m["Rank-1"], 4), round(m["Rank-5"], 4)]
            msg += f" | mAP@10 {m['mAP@10']:.4f}  R1 {m['Rank-1']:.4f}  R5 {m['Rank-5']:.4f}"
            if m["mAP@10"] > best_map:
                best_map = m["mAP@10"]
                torch.save(model.state_dict(), OUT / "best.pth")
                msg += "  <- best, saved"

        torch.save(model.state_dict(), OUT / "last.pth")    # crash insurance, every epoch
        writer.writerow(row); log.flush()
        print(msg)

    print(f"done. best mAP@10 = {best_map:.4f}  (without VeRi: 0.7019)")


if __name__ == "__main__":
    main()