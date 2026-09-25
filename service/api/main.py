"""
Falcon ReID — inference backend (FastAPI). Interactive OpenAPI docs: /api/docs (Swagger UI).

Microservices (ТЗ §6): this API (inference) | PostgreSQL + pgvector (gallery vectors + metadata) |
nginx web client (static page in the browser). The search logic is the same code as the batch run
(falcon.extractor + falcon.search.rank_candidates), so the demo gives the same answers as submission.csv.

Input is always an image + the vehicle bbox (answer #45: no detection in this task).
Each request is one query processed on its own (stream protocol, answer #38).
"""
import csv
import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from falcon.extractor import Extractor, crop_vehicle, load_config, open_image
from falcon.search import rank_candidates

from .db import GalleryDB

MODE = os.environ.get("FALCON_MODE")                        # default: falcon/config.json "mode"
DEVICE = os.environ.get("FALCON_DEVICE")                    # default: cuda if available
EXACT_BELOW = int(os.environ.get("EXACT_SEARCH_BELOW", "50000"))   # exact scan for small galleries
IMPORT_CSV = os.environ.get("GALLERY_IMPORT_CSV")           # optional: fill an empty gallery at start
IMPORT_IMAGES = os.environ.get("GALLERY_IMPORT_IMAGES")

state = {}


def _jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def import_gallery(csv_path, images_dir, batch=32):
    """Bulk-load a gallery from a CSV (image_id,x,y,w,h[,camera]) + folder of full frames."""
    ex, db = state["extractor"], state["db"]
    df = pd.read_csv(csv_path, dtype={"image_id": str})
    files = {p.stem: p for p in Path(images_dir).iterdir()}
    for i in range(0, len(df), batch):
        rows = list(df.iloc[i:i + batch].itertuples())
        items = [(files[r.image_id], (r.x, r.y, r.w, r.h)) for r in rows]
        with state["lock"]:
            vecs = ex.extract_batch(items)
        for r, (path, box), v in zip(rows, items, vecs):
            crop = crop_vehicle(open_image(path), box, ex.long_side)
            db.add(r.image_id, v, _jpeg(crop), camera=str(getattr(r, "camera", "") or "") or None,
                   source=Path(csv_path).name)
    return len(df)


@asynccontextmanager
async def lifespan(app):
    cfg = load_config()
    mode = MODE or cfg["mode"]
    threshold = cfg["modes"][mode]["refusal_threshold"]
    if threshold is None:
        raise RuntimeError(f"mode '{mode}' has no tuned refusal threshold in falcon/config.json")
    ex = Extractor(cfg, mode=mode, device=DEVICE)
    state.update(cfg=cfg, mode=mode, threshold=float(threshold), extractor=ex, lock=threading.Lock(),
                 db=GalleryDB(ex.dim), rerank=cfg["rerank"])
    if IMPORT_CSV and IMPORT_IMAGES and state["db"].count() == 0:
        n = import_gallery(IMPORT_CSV, IMPORT_IMAGES)
        print(f"imported {n} gallery vehicles from {IMPORT_CSV}")
    yield


app = FastAPI(
    title="Falcon ReID API",
    version="1.0.0",
    description="Vehicle re-identification without licence plates: image + bbox in, ranked gallery matches "
                "with a confidence score out, or a refusal when no confident match exists.",
    docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ models (documented in Swagger)
class Match(BaseModel):
    rank: int
    gallery_id: str
    similarity: float = Field(description="cosine similarity of the vectors (1 = identical)")
    camera: str | None = None
    label: str | None = None
    image_url: str


class SearchResult(BaseModel):
    search_id: str
    refused: bool = Field(description="true = no confident match in the gallery (refusal mode)")
    confidence: float = Field(description="cos+gap score of the top-1 match; answered when >= threshold")
    threshold: float
    top_match: str | None = Field(description="gallery_id of the accepted match, null when refused")
    results: list[Match] = Field(description="ranked candidates (shown even when refused, for the operator)")
    extract_ms: float
    search_ms: float
    gallery_size: int


class Health(BaseModel):
    status: str
    mode: str
    models: list[str]
    embedding_dim: int
    device: str
    threshold: float
    gallery_size: int


class Added(BaseModel):
    gallery_id: str
    added: bool


class GalleryItem(BaseModel):
    gallery_id: str
    camera: str | None
    label: str | None
    source: str | None
    created_at: str
    image_url: str


class GalleryPage(BaseModel):
    total: int
    items: list[GalleryItem]


# ------------------------------------------------------------------ helpers
async def _read_vehicle(file: UploadFile, x: int, y: int, w: int, h: int):
    data = await file.read()
    try:
        img = open_image(data)
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "the uploaded file is not a readable JPEG/PNG image")
    if w <= 0 or h <= 0:
        raise HTTPException(422, "bbox width and height must be positive")
    if x < 0 or y < 0 or x + w > img.width or y + h > img.height:
        raise HTTPException(422, f"bbox ({x}, {y}, {w}, {h}) is outside the image ({img.width}x{img.height})")
    return data, img


# ------------------------------------------------------------------ endpoints
@app.get("/api/health", response_model=Health, tags=["service"])
def health():
    ex = state["extractor"]
    return Health(status="ok", mode=state["mode"], models=[m["name"] for m in state["cfg"]["modes"][state["mode"]]["models"]],
                  embedding_dim=ex.dim, device=str(ex.device), threshold=state["threshold"],
                  gallery_size=state["db"].count())


@app.post("/api/search", response_model=SearchResult, tags=["search"],
          summary="Find the same vehicle in the gallery (or refuse)")
async def search(file: UploadFile = File(..., description="full camera frame (JPEG/PNG)"),
                 x: int = Form(..., description="bbox left, pixels"), y: int = Form(..., description="bbox top"),
                 w: int = Form(..., description="bbox width"), h: int = Form(..., description="bbox height"),
                 top_k: int = Form(10, ge=1, le=100, description="how many candidates to return")):
    data, _ = await _read_vehicle(file, x, y, w, h)
    ex, db, rr = state["extractor"], state["db"], state["rerank"]
    t0 = time.perf_counter()
    with state["lock"]:
        q = ex.extract(data, (x, y, w, h))
    t1 = time.perf_counter()
    size = db.count()
    if size == 0:
        raise HTTPException(409, "the gallery is empty: add vehicles first (POST /api/gallery)")
    k = max(min(rr["topk"], size), top_k)
    ids, vecs, cos, meta = db.nearest(q, k, exact=size <= EXACT_BELOW)
    order, conf = rank_candidates(q, vecs, cos, rr["k1"], rr["k2"], rr["lambda"], rr["enabled"])
    t2 = time.perf_counter()
    refused = conf < state["threshold"]
    results = [Match(rank=i + 1, gallery_id=ids[j], similarity=round(float(cos[j]), 4),
                     camera=meta[ids[j]]["camera"], label=meta[ids[j]]["label"],
                     image_url=f"/api/gallery/{ids[j]}/image") for i, j in enumerate(order[:top_k])]
    return SearchResult(search_id=uuid.uuid4().hex, refused=refused, confidence=round(conf, 4),
                        threshold=state["threshold"], top_match=None if refused else results[0].gallery_id,
                        results=results, extract_ms=round(1000 * (t1 - t0), 1),
                        search_ms=round(1000 * (t2 - t1), 1), gallery_size=size)


@app.post("/api/gallery", response_model=Added, tags=["gallery"], summary="Add a known vehicle to the gallery")
async def add_to_gallery(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...),
                         w: int = Form(...), h: int = Form(...),
                         gallery_id: str | None = Form(None, description="default: a new random id"),
                         camera: str | None = Form(None), label: str | None = Form(None)):
    data, img = await _read_vehicle(file, x, y, w, h)
    ex = state["extractor"]
    with state["lock"]:
        v = ex.extract(data, (x, y, w, h))
    gid = gallery_id or uuid.uuid4().hex
    state["db"].add(gid, v, _jpeg(crop_vehicle(img, (x, y, w, h), ex.long_side)), camera, label)
    return Added(gallery_id=gid, added=True)


@app.get("/api/gallery", response_model=GalleryPage, tags=["gallery"])
def list_gallery(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    items = [GalleryItem(**r, image_url=f"/api/gallery/{r['gallery_id']}/image")
             for r in state["db"].list(limit, offset)]
    return GalleryPage(total=state["db"].count(), items=items)


@app.get("/api/gallery/export.csv", tags=["gallery"], summary="Download the gallery list as CSV")
def export_gallery():
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["gallery_id", "camera", "label", "source", "created_at"])
    for r in state["db"].all_rows():
        wr.writerow([r[0], r[1] or "", r[2] or "", r[3] or "", r[4].isoformat()])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=gallery.csv"})


@app.get("/api/gallery/{gallery_id}/image", tags=["gallery"], summary="Cropped vehicle image (JPEG)",
         responses={200: {"content": {"image/jpeg": {}}}})
def gallery_image(gallery_id: str):
    data = state["db"].crop(gallery_id)
    if data is None:
        raise HTTPException(404, f"no gallery item '{gallery_id}'")
    return Response(data, media_type="image/jpeg")


@app.delete("/api/gallery/{gallery_id}", tags=["gallery"])
def delete_gallery_item(gallery_id: str):
    if not state["db"].delete(gallery_id):
        raise HTTPException(404, f"no gallery item '{gallery_id}'")
    return {"deleted": True}
