"""
A9: VeRi-776 as extra training data, with licence plates BLURRED first.

Why blur although it is allowed without (answers #46/#47)? The organizers re-test finalists with the plate
area painted over (#48). If VeRi taught the model to read plates, the score would drop on that test.
Blurring VeRi the same way the organizers blurred their images removes that risk.

Steps (for every VeRi photo, image_train + image_test):
  1. a public plate detector finds plates (YOLO11 fine-tuned on plates, Hugging Face
     morsetechlab/yolov11-license-plate-detection, file license-plate-finetune-v1s.pt, pinned revision below)
  2. each plate box (+25% margin) gets a strong Gaussian blur
  3. the photo is resized like our crops (longer side 384) and saved as data/crops/veri_<name>.jpg
  4. data/veri.csv lists image_id, vehicle_id (+100000, never clashes with ours), camera_id (+100000)
It also saves runs/veri_blur_check.jpg (before/after pairs) to CHECK BY EYE before training.
The detector is used only here, for training data; it is not part of the submitted pipeline.

  pip install ultralytics==8.4.163
  python experiments/prepare_veri.py --veri external/veri/VeRi
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

CROPS = ROOT / "data" / "crops"
OUT_CSV = ROOT / "data" / "veri.csv"
LONG_SIDE = 384
ID_OFFSET = 100000
DETECTOR_REPO = "morsetechlab/yolov11-license-plate-detection"
DETECTOR_FILE = "license-plate-finetune-v1s.pt"
DETECTOR_REV = "251a30d7daedca065f56e04b0af04052c907c68f"     # pinned (reproducibility, answer #46)


def load_detector(path):
    from ultralytics import YOLO
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DETECTOR_REPO, DETECTOR_FILE, revision=DETECTOR_REV)
    return YOLO(path), str(path)


def blur_plates(img, boxes, margin=0.25):
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        box = (max(0, int(x0 - margin * w)), max(0, int(y0 - margin * h)),
               min(img.width, int(x1 + margin * w) + 1), min(img.height, int(y1 + margin * h) + 1))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        region = out.crop(box)
        radius = max(4, min(region.size) // 2)
        out.paste(region.filter(ImageFilter.GaussianBlur(radius)), box)
    return out


def contact_sheet(pairs, path, cell=160):
    sheet = Image.new("RGB", (cell * 2 * 4, cell * ((len(pairs) + 3) // 4)), "white")
    for k, (before, after, n) in enumerate(pairs):
        r, c = divmod(k, 4)
        for j, im in enumerate((before, after)):
            t = im.copy()
            t.thumbnail((cell, cell))
            sheet.paste(t, (c * 2 * cell + j * cell, r * cell))
        ImageDraw.Draw(sheet).text((c * 2 * cell + 3, r * cell + 3), f"{n} plate(s)", fill="red")
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", default=str(ROOT / "external" / "veri" / "VeRi"))
    ap.add_argument("--detector", default=None, help="local .pt (default: download the pinned HF file)")
    ap.add_argument("--conf", type=float, default=0.15, help="low on purpose: better blur too much than too little")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--check", type=int, default=32, help="before/after pairs in runs/veri_blur_check.jpg")
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos (quick test)")
    args = ap.parse_args()

    veri = Path(args.veri)
    paths = sorted(p for folder in ("image_train", "image_test") for p in (veri / folder).glob("*.jpg"))
    if args.limit:
        paths = paths[:args.limit]
    assert paths, f"no VeRi photos under {veri}"
    CROPS.mkdir(parents=True, exist_ok=True)
    detector, det_path = load_detector(args.detector)
    rng = np.random.default_rng(0)
    check_idx = set(rng.choice(len(paths), size=min(args.check, len(paths)), replace=False).tolist())

    rows, pairs, n_with = [], [], 0
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        results = detector.predict([np.asarray(im)[:, :, ::-1] for im in imgs], conf=args.conf, verbose=False)
        for k, (p, im, res) in enumerate(zip(chunk, imgs, results)):
            boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
            n_with += bool(boxes)
            out = blur_plates(im, boxes)
            if i + k in check_idx:
                pairs.append((im, out, len(boxes)))
            s = LONG_SIDE / max(out.size)
            if s < 1:
                out = out.resize((round(out.width * s), round(out.height * s)), Image.BICUBIC)
            image_id = f"veri_{p.stem}"
            out.save(CROPS / f"{image_id}.jpg", quality=95)
            car, cam = p.stem.split("_")[:2]              # "0001_c001_00016450_0" -> "0001", "c001"
            rows.append((image_id, ID_OFFSET + int(car), ID_OFFSET + int(cam[1:])))
        print(f"\r  {min(i + args.batch, len(paths))}/{len(paths)}  photos with a plate found: {n_with}",
              end="", flush=True)
    print()

    df = pd.DataFrame(rows, columns=["image_id", "vehicle_id", "camera_id"])
    df.to_csv(OUT_CSV, index=False)
    (ROOT / "runs").mkdir(exist_ok=True)
    contact_sheet(pairs, ROOT / "runs" / "veri_blur_check.jpg")
    print(f"detector: {det_path}")
    print(f"VeRi photos: {len(df)} | cars: {df.vehicle_id.nunique()} | cameras: {df.camera_id.nunique()}")
    print(f"a plate was found and blurred in {n_with}/{len(df)} photos ({n_with / len(df):.1%})")
    print(f"saved -> {OUT_CSV}  and  runs/veri_blur_check.jpg (CHECK IT BY EYE)")


if __name__ == "__main__":
    main()
