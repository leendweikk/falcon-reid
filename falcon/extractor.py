"""
Feature extraction: one vehicle (full frame + bbox) -> one L2-normalized float32 vector.

extract() is the part the organizers time (answers #31/#32): read file, decode, crop by bbox,
preprocess, forward pass, L2-normalize. Search and re-ranking are NOT inside it.

Preprocessing is identical to what the models were trained and validated on
(make_crops.py + reid/data.py test transform):
    crop by bbox -> if the longer side > 384, shrink it to 384 (bicubic, keep proportions)
    -> resize to input_size x input_size (bilinear) -> ImageNet mean/std normalization.

Each image is processed on its own: no information from other images is used (answer #38).
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reid.data import MEAN, STD          # noqa: E402  (same constants as training)
from reid.model import ReIDModel          # noqa: E402  (same architecture code as training)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def load_infer_weights(path, device):
    """Load a slim inference file written by tools/export_weights.py (no classifier head).
    Builds the backbone with pretrained=False, so no internet is needed."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt.get("format") == "falcon-infer-v1", f"{path}: not an inference weights file"
    model = ReIDModel(num_classes=1, backbone=ckpt["backbone"], pretrained=False, pool=ckpt["pool"])
    state = {k: v.float() for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ["classifier.weight"] and not unexpected, (missing, unexpected)
    del model.classifier                                    # the ID head is only needed for training
    model.backbone.set_grad_checkpointing(False)            # checkpointing is a training-only memory trick
    return model.to(device).eval(), int(ckpt["input_size"])


def open_image(image):
    """File path, raw bytes or a PIL image -> PIL image (not decoded yet, so JPEG draft mode still works)."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(image))
    return Image.open(image)


def crop_vehicle(img, bbox, long_side=384):
    """Crop by bbox (x, y, w, h); if the longer side is > long_side, shrink it (bicubic, keep proportions).
    Identical to make_crops.py, i.e. to what the models were trained on."""
    x, y, w, h = bbox
    car = img.convert("RGB").crop((round(x), round(y), round(x + w), round(y + h)))
    scale = long_side / max(car.size)
    if scale < 1:
        car = car.resize((round(car.width * scale), round(car.height * scale)), Image.BICUBIC)
    return car


class Extractor:
    """Loads the models of one mode once, then extracts vectors.

    extract(image, bbox)          -> (D,) float32, L2-normalized
    extract_batch([(image, bbox)]) -> (N, D) float32, L2-normalized (same result per image, just faster)
    `image` can be a file path, raw bytes, or a PIL image; bbox = (x, y, w, h) in pixels.
    """

    def __init__(self, config=None, mode=None, device=None, decode=None):
        self.cfg = config if isinstance(config, dict) else load_config(config)
        self.mode = mode or self.cfg["mode"]
        spec = self.cfg["modes"][self.mode]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.decode = decode or self.cfg.get("decode", "pil")
        self.long_side = int(self.cfg.get("crop_long_side", 384))
        self.flip = bool(spec["flip"])
        self.models = []                                          # (model, input_size, weight)
        for m in spec["models"]:
            model, size = load_infer_weights(ROOT / m["file"], self.device)
            self.models.append((model, size, float(m["weight"])))
        self.sizes = sorted({s for _, s, _ in self.models})
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        # CPU decoding (read + JPEG decode + crop + resize) is the throughput bottleneck, not the GPU
        # (Colab T4 with 2 CPU threads: 47 FPS whatever the GPU). "auto" = one worker per CPU thread,
        # capped at 16, so a many-core test machine (the judges': 128 threads) decodes in parallel.
        w = self.cfg.get("decode_workers", 8)
        workers = min(16, os.cpu_count() or 1) if w == "auto" else int(w)
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.decode_workers = workers
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # ---------------- CPU part: read + decode + crop + resize ----------------
    def _prepare(self, image, bbox):
        """-> {input_size: uint8 array (size, size, 3)} for every input size the models need."""
        img = open_image(image)
        x, y, w, h = (float(v) for v in bbox)
        if self.decode == "pil-draft" and getattr(img, "format", None) == "JPEG":
            # JPEG can be decoded directly at 1/2, 1/4, 1/8 size; pick the biggest reduction that
            # still keeps the car's longer side >= crop_long_side. EXPERIMENTAL: check on validation first.
            for s in (8, 4, 2):
                if max(w, h) / s >= self.long_side:
                    full_w = img.width
                    img.draft("RGB", (img.width // s, img.height // s))
                    f = full_w / img.width
                    x, y, w, h = x / f, y / f, w / f, h / f
                    break
        car = crop_vehicle(img, (x, y, w, h), self.long_side)
        return {s: np.asarray(car.resize((s, s), Image.BILINEAR)) for s in self.sizes}

    # ---------------- GPU part: normalize + forward + L2 ----------------
    @torch.no_grad()
    def _forward(self, prepared):
        parts = []
        use_fp16 = self.device.type == "cuda"
        for model, size, weight in self.models:
            batch = torch.from_numpy(np.stack([p[size] for p in prepared])).to(self.device, non_blocking=True)
            batch = (batch.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_fp16):
                f = model(batch)
                if self.flip:                                      # single-image TTA (answer #38)
                    f = f + model(torch.flip(batch, dims=[3]))
            f = F.normalize(f.float(), dim=1)
            parts.append(np.sqrt(weight) * f)                      # "glued" ensemble, same as ensemble_test.py
        out = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return F.normalize(out, dim=1).cpu().numpy().astype(np.float32)

    def extract(self, image, bbox):
        """One vehicle -> (D,) float32 L2-normalized. This is the timed function."""
        return self._forward([self._prepare(image, bbox)])[0]

    def extract_batch(self, items):
        prepared = list(self.pool.map(lambda it: self._prepare(*it), items)) if self.pool \
            else [self._prepare(*it) for it in items]
        return self._forward(prepared)

    @property
    def dim(self):
        return sum(m.bnneck.num_features for m, _, _ in self.models)


_default = None


def extract(image, bbox):
    """Module-level shortcut: extract(path, (x, y, w, h)) with the default config (loaded once)."""
    global _default
    if _default is None:
        _default = Extractor()
    return _default.extract(image, bbox)
