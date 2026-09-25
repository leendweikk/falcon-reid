"""Falcon ReID — inference package (what runs inside Docker).

    from falcon import Extractor
    ex = Extractor()                              # reads falcon/config.json
    vec = ex.extract("frame.jpg", (x, y, w, h))   # one vehicle -> L2-normalized float32 vector

Batch run that writes the three official files:
    python -m falcon.predict --images <dir> --query test_query.csv --gallery test_gallery.csv --out <dir>
"""
from falcon.extractor import Extractor, extract

__all__ = ["Extractor", "extract"]
