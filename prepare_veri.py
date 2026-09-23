from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

VERI = Path("C:/falcon/external/veri/VeRi")
CROPS = Path("C:/falcon/data/crops")
OUT_CSV = Path("C:/falcon/data/splits/veri.csv")
LONG_SIDE = 384
ID_OFFSET = 100000          # VeRi car ids become 100001, 100002, ... (never clash with ours)

rows = []
for folder in ["image_train", "image_test"]:
    for path in tqdm(sorted((VERI / folder).glob("*.jpg")), desc=folder):
        car, cam = path.stem.split("_")[:2]            # "0001_c001_00016450_0" -> "0001", "c001"
        image_id = f"veri_{path.stem}"

        img = Image.open(path).convert("RGB")
        scale = LONG_SIDE / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
        img.save(CROPS / f"{image_id}.jpg", quality=95)

        rows.append((image_id, ID_OFFSET + int(car), ID_OFFSET + int(cam[1:])))

df = pd.DataFrame(rows, columns=["image_id", "vehicle_id", "camera_id"])
df.to_csv(OUT_CSV, index=False)

print("VeRi photos:", len(df))
print("VeRi cars:", df.vehicle_id.nunique(), " cameras:", df.camera_id.nunique())
print("photos per car  min/median/max:", *df.groupby("vehicle_id").size().agg(["min", "median", "max"]))