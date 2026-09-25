"""
Run the web service on a laptop WITHOUT Docker (development / demo):
  - starts an embedded PostgreSQL + pgvector (pgserver) with its data in runs/pgdev
  - serves the API on http://localhost:8000/api/docs and the web page on http://localhost:8000/

  pip install -r requirements-service.txt -r requirements-dev.txt
  python service/dev_server.py                                  # empty gallery
  python service/dev_server.py --import-csv data/test_gallery.csv --images data/images
  python service/dev_server.py --device cpu                      # keep the GPU free (e.g. while training)

In Docker the same API runs behind nginx with a real PostgreSQL container (docker-compose.yml).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    ap.add_argument("--mode", default=None, help="fast | accurate ... (default: falcon/config.json)")
    ap.add_argument("--import-csv", default=None, help="fill an EMPTY gallery from this CSV at start")
    ap.add_argument("--images", default=None, help="folder with the full frames for --import-csv")
    ap.add_argument("--pgdata", default=str(ROOT / "runs" / "pgdev"))
    args = ap.parse_args()

    import pgserver
    srv = pgserver.get_server(args.pgdata, cleanup_mode="stop")
    os.environ["DATABASE_URL"] = srv.get_uri()
    if args.device:
        os.environ["FALCON_DEVICE"] = args.device
    if args.mode:
        os.environ["FALCON_MODE"] = args.mode
    if args.import_csv:
        os.environ["GALLERY_IMPORT_CSV"] = args.import_csv
        os.environ["GALLERY_IMPORT_IMAGES"] = args.images or str(ROOT / "data" / "images")

    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from service.api.main import app
    app.mount("/", StaticFiles(directory=ROOT / "service" / "web", html=True), name="web")   # after the /api routes
    print(f"web page: http://localhost:{args.port}/   API docs: http://localhost:{args.port}/api/docs")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
