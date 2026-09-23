from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA = Path("C:/falcon/data")

# 1. Build a lookup: image_id (file name without extension) -> real file path
files = {p.stem: p for p in (DATA / "images").iterdir()}
print("image files on disk:", len(files))

# 2. Load the three tables
train = pd.read_csv(DATA / "train.csv", dtype={"image_id": str})
query = pd.read_csv(DATA / "test_query.csv", dtype={"image_id": str})
gallery = pd.read_csv(DATA / "test_gallery.csv", dtype={"image_id": str})

# 3. Check each table: missing files, and boxes that fall outside the photo
for name, df in [("train", train), ("query", query), ("gallery", gallery)]:
    missing, bad_box = 0, 0
    for row in tqdm(df.itertuples(), total=len(df), desc=name):
        path = files.get(row.image_id)
        if path is None:
            missing += 1
            continue
        W, H = Image.open(path).size          # reads only the header, fast
        if row.x < 0 or row.y < 0 or row.x + row.w > W or row.y + row.h > H or row.w < 1 or row.h < 1:
            bad_box += 1
    print(f"{name}: rows={len(df)} missing_files={missing} boxes_outside_image={bad_box}")

# 4. Leak check: the same image_id in two different sets
t, q, g = set(train.image_id), set(query.image_id), set(gallery.image_id)
print("overlap train-query:", len(t & q), " train-gallery:", len(t & g), " query-gallery:", len(q & g))

# 5. Basic statistics for the training set
per_car = train.groupby("vehicle_id").size()
cams_per_car = train.groupby("vehicle_id").camera_id.nunique()
print("cars:", train.vehicle_id.nunique(), " cameras:", train.camera_id.nunique())
print("photos per car  min/median/max:", per_car.min(), per_car.median(), per_car.max())
print("cameras per car min/median/max:", cams_per_car.min(), cams_per_car.median(), cams_per_car.max())
print("box width median:", int(train.w.median()), " box height median:", int(train.h.median()))