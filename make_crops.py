from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

DATA = (ROOT / "data")
OUT = DATA / "crops"
OUT.mkdir(exist_ok=True)
LONG_SIDE = 384   # longer side of each saved crop, in pixels

files = {p.stem: p for p in (DATA / "images").iterdir()}

for csv_name in ["train.csv", "test_query.csv", "test_gallery.csv"]:
    df = pd.read_csv(DATA / csv_name, dtype={"image_id": str})
    for row in tqdm(df.itertuples(), total=len(df), desc=csv_name):
        img = Image.open(files[row.image_id]).convert("RGB")
        car = img.crop((row.x, row.y, row.x + row.w, row.y + row.h))

        # shrink so the longer side = LONG_SIDE, keeping the car's proportions
        scale = LONG_SIDE / max(car.size)
        if scale < 1:
            new_size = (round(car.width * scale), round(car.height * scale))
            car = car.resize(new_size, Image.BICUBIC)

        car.save(OUT / f"{row.image_id}.jpg", quality=95)

print("crops saved:", len(list(OUT.iterdir())))