from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("C:/falcon/data")
OUT = DATA / "splits"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)          # fixed seed = same split every run

N_VAL_CARS = 300       # same size as the real test
N_OPENSET_CARS = 50    # "stranger" cars: queries only, nothing in gallery

train = pd.read_csv(DATA / "train.csv", dtype={"image_id": str})

# 1. Choose the held-out cars (split by CAR, never by photo)
cars = train.vehicle_id.unique()
val_cars = rng.choice(cars, size=N_VAL_CARS, replace=False)
openset_cars = set(val_cars[:N_OPENSET_CARS])

train_split = train[~train.vehicle_id.isin(val_cars)]
val = train[train.vehicle_id.isin(val_cars)].copy()

# 2. Organizers' rule: for each (car, camera), one random photo -> gallery
val["split"] = "query"
for (car, cam), idx in val.groupby(["vehicle_id", "camera_id"]).groups.items():
    if car in openset_cars:
        continue                          # strangers: all photos stay as queries
    val.loc[rng.choice(list(idx)), "split"] = "gallery"

# 3. Save files
train_split.to_csv(OUT / "train_split.csv", index=False)
val[val.split == "query"][["image_id", "x", "y", "w", "h"]].to_csv(OUT / "val_query.csv", index=False)
val[val.split == "gallery"][["image_id", "x", "y", "w", "h"]].to_csv(OUT / "val_gallery.csv", index=False)
val[["image_id", "vehicle_id", "camera_id", "split"]].to_csv(OUT / "val_gt.csv", index=False)

# 4. Report
q, g = (val.split == "query").sum(), (val.split == "gallery").sum()
print("train cars:", train_split.vehicle_id.nunique(), " photos:", len(train_split))
print("val cars:", N_VAL_CARS, " (open-set:", N_OPENSET_CARS, ")")
print(f"val queries: {q}  val gallery: {g}  ratio: {q/g:.2f}")
print("open-set queries:", val[val.vehicle_id.isin(openset_cars)].shape[0])