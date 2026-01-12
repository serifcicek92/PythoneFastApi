#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_long_mpnet.py
paraphrase-multilingual-mpnet-base-v2  (768-D, 512 token, Apache-2.0)
Sliding-window (overlap) ile uzun metni kesmeden embed eder.
pgvector tablosuna tek satır yazar.
"""
import argparse
import os
import tiktoken
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
WIN_SIZE   = 512          # token
STRIDE     = 256          # overlap
ENC        = tiktoken.get_encoding("cl100k_base")
DB_DSN     = os.getenv("DB_DSN", "host=localhost dbname=postgres user=postgres password=sql123 port=5432")

# ---------- MODEL ----------
model = SentenceTransformer(MODEL_NAME)   # 768 boyut

# ---------- DB ----------
def get_conn():
    return psycopg2.connect(DB_DSN)

def init_table():
    sql = """
    CREATE TABLE IF NOT EXISTS doc_chunks_long (
        id          SERIAL PRIMARY KEY,
        parent_id   INT,
        chunk_no    INT DEFAULT 0,
        title       TEXT,
        text        TEXT,
        embedding   VECTOR(768)
    );
    CREATE INDEX IF NOT EXISTS idx_long_embedding ON doc_chunks_long USING ivfflat (embedding);
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()

# ---------- CORE ----------
def token_count(text: str) -> int:
    return len(ENC.encode(text))

def sliding_windows(text: str, win_size: int = WIN_SIZE, stride: int = STRIDE):
    tokens = ENC.encode(text)
    start = 0
    while start < len(tokens):
        yield tokens[start:start + win_size]
        start += stride

def embed_long(text: str) -> np.ndarray:
    vecs = []
    for window_tokens in sliding_windows(text):
        window_text = ENC.decode(window_tokens)
        vecs.append(model.encode(window_text, normalize_embeddings=True))
    return np.mean(vecs, axis=0)   # 768-D

def add_long_doc(title: str, long_text: str, parent_id: int | None = None) -> int:
    emb = embed_long(long_text)
    emb_str = "[" + ",".join(map(str, emb)) + "]"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO doc_chunks_long (parent_id, chunk_no, title, text, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            RETURNING id
        """, (parent_id, 0, title, long_text, emb_str))
        new_id = cur.fetchone()[0]
        conn.commit()
    return new_id

def search_long(query: str, top_k: int = 3):
    emb = embed_long(query)
    emb_str = "[" + ",".join(map(str, emb)) + "]"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, text,
                   embedding <-> %s::vector AS dist
            FROM   doc_chunks_long
            ORDER  BY dist
            LIMIT  %s
        """, (emb_str, top_k))
        return cur.fetchall()

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPNet-TR uzun metin embed + pgvector")
    parser.add_argument("--init", action="store_true", help="Tablo oluştur")
    parser.add_argument("--add", metavar="TITLE", help="Metin ekle (stdin'den oku)")
    parser.add_argument("--search", metavar="SORU", help="Arama yap")
    parser.add_argument("--test", action="store_true", help="Hızlı test")
    args = parser.parse_args()

    if args.init:
        init_table()
        print("✅ Tablo oluşturuldu / hazır.")
    elif args.add:
        text = sys.stdin.read().strip()
        pid = add_long_doc(title=args.add, long_text=text)
        print(f"✅ Eklendi, id={pid}")
    elif args.search:
        for row in search_long(args.search):
            print(f"dist={row[3]:.4f}  id={row[0]}  title={row[1]}")
    elif args.test:
        init_table()
        sample = """
        İhale fatura eşleştirme ekranında kapalı ihaleler listelenmez.
        Kapalı ihaleye fatura bağlamak için önce ihaleyi aktif yapmalısınız.
        Aksi hâlde 286086 no’lu fiş 10107 no’lu ihaleye bağlanamaz çünkü
        liste boş görünür. Bunun sebebi IH_DURUM = 'K' filtresidir.
        """
        pid = add_long_doc(title="Test - Kapalı ihale", long_text=sample)
        print("🔍 Aranıyor: '286086 fiş bağlanmıyor'")
        for r in search_long("286086 fiş bağlanmıyor"):
            print(r[2][:100] + "…")
    else:
        parser.print_help()