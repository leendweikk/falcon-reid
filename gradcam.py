"""
Grad-CAM for re-identification: WHICH parts of the car made the model say "same vehicle"?
For a query and its top-1 match, we ask: which regions of the query push its fingerprint
towards the match's fingerprint? (gradient of the cosine similarity w.r.t. the last ConvNeXt stage)

  python gradcam.py runs/speed_base_noflip --weights runs/convnext_b_v6_ema/best_ema.pth
Writes docs/gradcam/*.jpg : [query + heatmap | match + heatmap], green = correct match, red = wrong
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from reid.data import CROPS, INPUT_HW, test_transform
from reid.model import ReIDModel

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"


def load(path, device):
    state = torch.load(path, map_location="cpu")
    m = ReIDModel(num_classes=state["classifier.weight"].shape[0], backbone="convnext_base.dinov3_lvd1689m",
                  pretrained=False)
    m.load_state_dict(state)
    m.backbone.set_grad_checkpointing(False)
    return m.to(device).eval()


def embed(model, image_id, device):
    x = test_transform(Image.open(CROPS / f"{image_id}.jpg").convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return F.normalize(model(x).float(), dim=1)


def grad_cam(model, image_id, target_emb, device):
    """Heatmap (0..1) of the regions that increase cosine(query, target)."""
    store = {}
    layer = model.backbone.stages[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    x = test_transform(Image.open(CROPS / f"{image_id}.jpg").convert("RGB")).unsqueeze(0).to(device)
    model.zero_grad()
    score = (F.normalize(model(x).float(), dim=1) * target_emb).sum()
    score.backward()
    h1.remove(); h2.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)                     # importance of each channel
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=INPUT_HW, mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), float(score.detach())


def overlay(image_id, cam, label, color):
    img = Image.open(CROPS / f"{image_id}.jpg").convert("RGB").resize(INPUT_HW[::-1])
    heat = np.stack([np.clip(2 * cam, 0, 1), np.clip(2 * cam - 1, 0, 1), np.zeros_like(cam)], axis=2)  # black-red-yellow
    mixed = (0.55 * np.asarray(img) / 255 + 0.45 * heat) * 255
    tile = Image.new("RGB", (INPUT_HW[1] + 8, INPUT_HW[0] + 30), color)
    tile.paste(Image.fromarray(mixed.astype(np.uint8)), (4, 4))
    ImageDraw.Draw(tile).text((8, INPUT_HW[0] + 10), label, fill="white")
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="pipeline output folder on the validation split (with submission.csv)")
    ap.add_argument("--weights", required=True, help="Base model trained WITHOUT the validation cars")
    ap.add_argument("--n-correct", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load(args.weights, device)
    gt = pd.read_csv(SPLITS / "val_gt.csv", dtype={"image_id": str}).set_index("image_id")
    ranked = {r[0]: r[1:] for r in (l.strip().split(",") for l in open(Path(args.pred) / "submission.csv"))}

    correct, wrong = [], []
    for q, cands in ranked.items():
        v, c = gt.loc[q, "vehicle_id"], gt.loc[q, "camera_id"]
        cross = [g for g in cands if not (gt.loc[g, "vehicle_id"] == v and gt.loc[g, "camera_id"] == c)]
        if not cross or not ((gt.vehicle_id == v) & (gt.camera_id != c) & (gt.split == "gallery")).any():
            continue                                             # open-set query or nothing cross-camera
        (correct if gt.loc[cross[0], "vehicle_id"] == v else wrong).append((q, cross[0]))
    rng = np.random.default_rng(0)
    picks = [(p, True) for p in rng.permutation(len(correct))[:args.n_correct].tolist()] + \
            [(p, False) for p in rng.permutation(len(wrong))[:args.n_wrong].tolist()]

    out = ROOT / "docs" / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    for n, (i, ok) in enumerate(picks, 1):
        q, g = (correct if ok else wrong)[i]
        cam_q, s = grad_cam(model, q, embed(model, g, device), device)
        cam_g, _ = grad_cam(model, g, embed(model, q, device), device)
        color = (0, 150, 0) if ok else (190, 0, 0)
        tiles = [overlay(q, cam_q, f"QUERY  (cross-camera match, sim {s:.2f})", color),
                 overlay(g, cam_g, "MATCH  " + ("correct" if ok else "WRONG car"), color)]
        grid = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
        grid.paste(tiles[0], (0, 0)); grid.paste(tiles[1], (tiles[0].width, 0))
        grid.save(out / f"gradcam_{n:02d}_{'correct' if ok else 'wrong'}.jpg", quality=92)
    print(f"saved {len(picks)} Grad-CAM pairs -> {out}")


if __name__ == "__main__":
    main()
