import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

CROPS = Path(__file__).resolve().parent / "data" / "crops"
INPUT_HW = (256, 256)          # proven setting; (224, 288) was tested in Experiment 3 and REJECTED
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
STRONG_LIGHT_AUG = False       # Experiment 1 (night/glare augmentation) was tested and REJECTED

if STRONG_LIGHT_AUG:
    light = [
        T.ColorJitter(brightness=(0.35, 1.4), contrast=(0.6, 1.4), saturation=(0.7, 1.3)),  # no hue
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),             # night blur
    ]
else:
    light = [T.ColorJitter(brightness=0.2, contrast=0.15)]                                   # proven setting

train_transform = T.Compose([
    T.Resize(INPUT_HW),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop(INPUT_HW),
    *light,
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.5, value="random"),
])

test_transform = T.Compose([
    T.Resize(INPUT_HW),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class TrainSet(Dataset):
    """Returns (image, car_label). Accepts one CSV path or a list of CSV paths (merged)."""

    def __init__(self, csv_paths):
        if not isinstance(csv_paths, (list, tuple)):
            csv_paths = [csv_paths]
        df = pd.concat([pd.read_csv(p, dtype={"image_id": str}) for p in csv_paths], ignore_index=True)
        self.car_to_label = {car: i for i, car in enumerate(sorted(df.vehicle_id.unique()))}
        self.items = [(r.image_id, self.car_to_label[r.vehicle_id]) for r in df.itertuples()]
        self.num_classes = len(self.car_to_label)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image_id, label = self.items[i]
        img = Image.open(CROPS / f"{image_id}.jpg").convert("RGB")
        return train_transform(img), label


class PKSampler(Sampler):
    """Builds batches of P cars x K photos each."""

    def __init__(self, dataset, P=16, K=4):
        self.P, self.K = P, K
        self.by_label = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.items):
            self.by_label[label].append(idx)

    def __iter__(self):
        labels = list(self.by_label)
        random.shuffle(labels)
        for start in range(0, len(labels) - self.P + 1, self.P):
            batch = []
            for label in labels[start:start + self.P]:
                idxs = self.by_label[label]
                batch += random.sample(idxs, self.K) if len(idxs) >= self.K else random.choices(idxs, k=self.K)
            yield batch

    def __len__(self):
        return len(self.by_label) // self.P