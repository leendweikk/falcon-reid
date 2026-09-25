"""
Make sure every weights file listed in weights/manifest.json is present AND has the right sha256
(answer #39: weights downloaded during docker build must be checked by checksum).

  python tools/fetch_weights.py            # download missing files, verify all
  python tools/fetch_weights.py --update   # (maintainers) write the sha256 of the local files into the manifest

Used in the Dockerfile at build time; the container itself never goes online.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights" / "manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    ok = True
    for entry in manifest["files"]:
        path = ROOT / "weights" / entry["name"]
        if args.update:
            entry["sha256"] = sha256(path)
            entry["bytes"] = path.stat().st_size
            print(f"{entry['name']}: {entry['sha256']}")
            continue
        if not path.exists():
            print(f"downloading {entry['name']} <- {entry['url']}")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(entry["url"], tmp)
            tmp.rename(path)
        digest = sha256(path)
        good = digest == entry["sha256"]
        ok &= good
        print(f"{'OK  ' if good else 'BAD '} {entry['name']} sha256 {digest}")

    if args.update:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"updated {MANIFEST}")
    elif not ok:
        sys.exit("checksum mismatch: refusing to continue")


if __name__ == "__main__":
    main()
