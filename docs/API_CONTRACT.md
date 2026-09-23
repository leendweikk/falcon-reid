# Falcon ReID — API contract (backend ↔ frontend)

Backend: FastAPI, `http://localhost:8000`. Interactive docs (Swagger): `http://localhost:8000/docs`.
All responses are JSON unless stated otherwise. CORS is enabled for the frontend.

## GET /health
Response:
```json
{ "status": "ok", "model": "convnext_base+small ensemble", "gallery_size": 750 }
```

## POST /search
Find the most similar gallery vehicles for a query photo.
Request: `multipart/form-data`
| field  | type  | required | meaning |
|--------|-------|----------|---------|
| file   | image (jpg/png) | yes | full camera frame |
| x, y, w, h | int | no | vehicle bounding box in pixels; if omitted, the whole image is used |
| top_k  | int   | no (default 10) | how many candidates to return |

Response:
```json
{
  "refused": false,
  "threshold": 0.60,
  "top1_confidence": 0.83,
  "latency_ms": 42.5,
  "results": [
    { "rank": 1, "gallery_id": "df7ea501...", "confidence": 0.83,
      "plate_number": "A123BC77", "camera": "cam_12",
      "image_url": "/gallery/df7ea501.../image" }
  ]
}
```
- `refused = true` when `top1_confidence < threshold`: the UI must show a clear
  "No confident match in the database" message, but may still show `results` greyed out.
- `plate_number` and `camera` may be `null`.

## POST /gallery/add
Register a known vehicle photo in the database.
Request: `multipart/form-data` — `file`, optional `x, y, w, h`, optional `gallery_id`,
optional `plate_number`, optional `camera`.
Response:
```json
{ "gallery_id": "df7ea501...", "added": true }
```

## GET /gallery?limit=50&offset=0
Response:
```json
{ "total": 750, "items": [ { "gallery_id": "...", "plate_number": null, "camera": null,
                              "image_url": "/gallery/.../image" } ] }
```

## GET /gallery/{gallery_id}/image
Returns the cropped vehicle image (`image/jpeg`).

## DELETE /gallery/{gallery_id}
Response: `{ "deleted": true }`

## Errors
Any error returns HTTP 4xx/5xx with `{ "detail": "human-readable message" }`
(e.g. box outside the image, unreadable file).