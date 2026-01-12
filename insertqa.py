# bulk_insert_qa.py
import json, re, psycopg2, os
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DB = dict(host="localhost", dbname="postgres",
          user="postgres", password="sql123", port=5432)
JL_FILE = "kural.jsonl"          # <- json-lines dosyan
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# ------------------------------

CLEAN = lambda t: re.sub(r"\s+", " ", t.strip())

conn = psycopg2.connect(**DB)
cur = conn.cursor()

insert_sql = """
INSERT INTO qa_pairs (question, answer, embedding)
VALUES (%s, %s, %s::vector)
"""

with open(JL_FILE, encoding="utf-8") as f:
    for line in f:
        if not line.strip():          # boş satır
            continue
        try:
            rec = json.loads(line)
            q = CLEAN(rec["prompt"])
            a = CLEAN(rec["completion"])
            emb = "[" + ",".join(map(str, MODEL.encode(q))) + "]"
            cur.execute(insert_sql, (q, a, emb))
        except Exception as e:
            print("HATA:", e, line[:80])

conn.commit()
print("Tüm QA'lar qa_pairs tablosuna eklendi.")
cur.close(); conn.close()