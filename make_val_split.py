import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

DATA = (ROOT / "data")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42          # 42 = our original split
OUT = DATA / ("splits" if SEED == 42 else f"splits_seed{SEED}")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

N_VAL_CARS = 300       # same size as the real test
N_OPENSET_CARS = 50    # "stranger" cars: queries only, nothing in gallery

train = pd.read_csv(DATA / "train.csv", dtype={"image_id": str})

cars = train.vehicle_id.unique()
val_cars = rng.choice(cars, size=N_VAL_CARS, replace=False)
openset_cars = set(val_cars[:N_OPENSET_CARS])

train_split = train[~train.vehicle_id.isin(val_cars)]
val = train[train.vehicle_id.isin(val_cars)].copy()

val["split"] = "query"
for (car, cam), idx in val.groupby(["vehicle_id", "camera_id"]).groups.items():
    if car in openset_cars:
        continue
    val.loc[rng.choice(list(idx)), "split"] = "gallery"

train_split.to_csv(OUT / "train_split.csv", index=False)
val[val.split == "query"][["image_id", "x", "y", "w", "h"]].to_csv(OUT / "val_query.csv", index=False)
val[val.split == "gallery"][["image_id", "x", "y", "w", "h"]].to_csv(OUT / "val_gallery.csv", index=False)
val[["image_id", "vehicle_id", "camera_id", "split"]].to_csv(OUT / "val_gt.csv", index=False)

q, g = (val.split == "query").sum(), (val.split == "gallery").sum()
overlap = len(set(val_cars) & set(pd.read_csv(DATA / "splits" / "val_gt.csv").vehicle_id)) if SEED != 42 else 300
print(f"seed {SEED} -> {OUT}")
print("train cars:", train_split.vehicle_id.nunique(), " photos:", len(train_split))
print(f"val queries: {q}  val gallery: {g}  ratio: {q / g:.2f}")
print(f"val cars shared with the original split: {overlap}/300")