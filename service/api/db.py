"""
Gallery storage in PostgreSQL + pgvector (ТЗ §6: a vector DBMS for embeddings and gallery metadata).

One table: every gallery vehicle has its embedding, optional metadata and its cropped image (so the
API itself stays stateless). An HNSW index answers "nearest vectors by cosine distance" in
sub-linear time, which is what lets the search scale to large galleries.
In two-stage mode each row also stores the heavier "rescore" vector (Base+ViT ensemble). It needs no
index: it is only read for the fast top-K candidates of one query, exactly like the batch run.
"""
import os

import numpy as np
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://falcon:falcon@db:5432/falcon")



def _np(v):
    """pgvector returns a Vector object (newer versions) or an array: -> float32 numpy."""
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32)


class GalleryDB:
    def __init__(self, dim, url=DATABASE_URL, rescore_dim=None):
        self.dim, self.rescore_dim = dim, rescore_dim
        with ConnectionPool(url, min_size=1, max_size=1, open=True) as bootstrap:   # create the extension first
            with bootstrap.connection() as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.pool = ConnectionPool(url, min_size=1, max_size=8, open=True, configure=register_vector)
        with self.pool.connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS gallery (
                    gallery_id  TEXT PRIMARY KEY,
                    embedding   vector({dim}) NOT NULL,
                    camera      TEXT,
                    label       TEXT,
                    source      TEXT,
                    crop_jpeg   BYTEA NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS gallery_hnsw ON gallery "
                         "USING hnsw (embedding vector_cosine_ops)")
            stored = conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'gallery'::regclass "
                                  "AND attname = 'embedding'").fetchone()[0]
            stored2, missing2 = None, 0
            if rescore_dim:
                conn.execute(f"ALTER TABLE gallery ADD COLUMN IF NOT EXISTS rescore vector({rescore_dim})")
                stored2 = conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = 'gallery'::regclass "
                                       "AND attname = 'rescore'").fetchone()[0]
                missing2 = conn.execute("SELECT count(*) FROM gallery WHERE rescore IS NULL").fetchone()[0]
        reset = "the model mode changed, so reset the database volume and re-import the gallery"
        if stored != dim:
            raise RuntimeError(f"the gallery table stores {stored}-d vectors but the model makes {dim}-d vectors: {reset}")
        if rescore_dim and stored2 != rescore_dim:
            raise RuntimeError(f"the gallery stores {stored2}-d rescore vectors, stage 2 makes {rescore_dim}-d: {reset}")
        if missing2:
            raise RuntimeError(f"{missing2} gallery rows have no stage-2 (rescore) vector: {reset}")

    def add(self, gallery_id, vector, crop_jpeg, camera=None, label=None, source="api", rescore=None):
        if self.rescore_dim and rescore is None:
            raise ValueError("two-stage mode: every gallery row needs its rescore vector")
        cols = ["gallery_id", "embedding", "camera", "label", "source", "crop_jpeg"]
        vals = [gallery_id, np.asarray(vector, np.float32), camera, label, source, crop_jpeg]
        if self.rescore_dim:
            cols.append("rescore")
            vals.append(np.asarray(rescore, np.float32))
        update = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[1:])
        with self.pool.connection() as conn:
            conn.execute(f"INSERT INTO gallery ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
                         f"ON CONFLICT (gallery_id) DO UPDATE SET {update}", vals)

    def nearest(self, q, k, exact=False):
        """Top-k gallery items by cosine similarity. Returns ids, vectors (k, D), cosines (k,), metadata,
        rescore vectors (k, D2) or None when not in two-stage mode.
        exact=True scans every row (same answer as the batch run); otherwise the HNSW index is used
        (approximate, sub-linear: for big galleries)."""
        extra = ", rescore" if self.rescore_dim else ""
        with self.pool.connection() as conn, conn.transaction():
            if exact:
                conn.execute("SET LOCAL enable_indexscan = off")
            else:
                conn.execute("SET LOCAL hnsw.ef_search = %d" % max(40, 2 * k))
            rows = conn.execute("SELECT gallery_id, embedding, camera, label, 1 - (embedding <=> %s) AS cos"
                                f"{extra} FROM gallery ORDER BY embedding <=> %s LIMIT %s",
                                (np.asarray(q, np.float32), np.asarray(q, np.float32), k)).fetchall()
        ids = [r[0] for r in rows]
        vecs = np.stack([_np(r[1]) for r in rows]) if rows else np.zeros((0, self.dim), np.float32)
        meta = {r[0]: {"camera": r[2], "label": r[3]} for r in rows}
        resc = None
        if self.rescore_dim:
            resc = np.stack([_np(r[5]) for r in rows]) if rows else np.zeros((0, self.rescore_dim), np.float32)
        return ids, vecs, np.array([r[4] for r in rows], np.float32), meta, resc

    def count(self):
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM gallery").fetchone()[0]

    def list(self, limit, offset):
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT gallery_id, camera, label, source, created_at FROM gallery "
                                "ORDER BY created_at, gallery_id LIMIT %s OFFSET %s", (limit, offset)).fetchall()
        return [{"gallery_id": r[0], "camera": r[1], "label": r[2], "source": r[3],
                 "created_at": r[4].isoformat()} for r in rows]

    def crop(self, gallery_id):
        with self.pool.connection() as conn:
            row = conn.execute("SELECT crop_jpeg FROM gallery WHERE gallery_id = %s", (gallery_id,)).fetchone()
        return bytes(row[0]) if row else None

    def delete(self, gallery_id):
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM gallery WHERE gallery_id = %s", (gallery_id,)).rowcount > 0

    def all_rows(self):
        with self.pool.connection() as conn:
            return conn.execute("SELECT gallery_id, camera, label, source, created_at FROM gallery "
                                "ORDER BY created_at, gallery_id").fetchall()
