# Falcon ReID — API contract (backend ↔ web client)

The live, always-correct version is the OpenAPI spec served by the backend:
**Swagger UI `/api/docs`**, ReDoc `/api/redoc`, JSON `/api/openapi.json`.
This file is the human summary. All paths start with `/api`. Errors return `{"detail": "..."}`.

The service never uses licence plates, and there is no plate field anywhere (task rule, answer #48).
Input is always a full frame **plus the vehicle bbox**, and the service does no detection (answer #45).

## GET /api/health
```json
{ "status": "ok", "mode": "fast", "models": ["vit"], "embedding_dim": 768,
  "device": "cuda", "threshold": 0.86, "gallery_size": 750 }
```

## POST /api/search — find the same vehicle (or refuse)
`multipart/form-data`

| field | type | required | meaning |
|---|---|---|---|
| file | JPEG/PNG | yes | full camera frame |
| x, y, w, h | int | **yes** | vehicle bbox in pixels (must lie inside the image) |
| top_k | int 1–100 | no (10) | number of candidates returned |

```json
{
  "search_id": "3f1c…",
  "refused": false,
  "confidence": 0.9907,
  "threshold": 0.86,
  "top_match": "df7ea501…",
  "results": [
    { "rank": 1, "gallery_id": "df7ea501…", "similarity": 0.977,
      "camera": null, "label": null, "image_url": "/api/gallery/df7ea501…/image" }
  ],
  "extract_ms": 33.8, "search_ms": 8.7, "gallery_size": 750
}
```
- `refused = true` when `confidence < threshold`. Then `top_match` is `null`, and the UI must show
  "no confident match in the gallery". The candidates are still returned so the operator can look at
  them, but they must be shown as NOT accepted.
- `confidence` is the cos+gap score (same rule as `candidates.csv`); `similarity` is the plain cosine
  of each candidate.
- The ranking is the same as the batch run: cosine top-100, then k-reciprocal re-ranking of that query
  alone, never using other queries (answer #38).

## POST /api/gallery — add a known vehicle
`multipart/form-data`: `file`, `x`, `y`, `w`, `h` (required), plus `gallery_id` (optional, default random),
`camera` (optional) and `label` (optional).
Response: `{ "gallery_id": "…", "added": true }`. The same `gallery_id` again replaces the entry.

## GET /api/gallery?limit=50&offset=0
`{ "total": 750, "items": [ { "gallery_id", "camera", "label", "source", "created_at", "image_url" } ] }`

## GET /api/gallery/{gallery_id}/image
The stored vehicle crop (`image/jpeg`), or 404.

## GET /api/gallery/export.csv
The whole gallery list as CSV (`gallery_id,camera,label,source,created_at`).

## DELETE /api/gallery/{gallery_id}
`{ "deleted": true }`, or 404.

## Error codes
400 unreadable image · 404 unknown gallery id · 409 empty gallery · 422 bad bbox / missing field
