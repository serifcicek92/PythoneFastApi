from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os
import psycopg2
import json
import re
import ollama
import requests  # Swagger API çağrısı için
import time
from llama_cpp import Llama
from openai import OpenAI


try:
    from keys import Keys
except ImportError:
    Keys = None

OPENAI_KEY = getattr(Keys, "OPENAI_API_KEY", None)

#openai.api_key = OPENAI_KEY

app = FastAPI(title="Boyut Bilgisayar Kurumsal Asistan")

# ------------------------------
# PostgreSQL bağlantısı
# ------------------------------
DB_HOST = "localhost"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "sql123"
DB_PORT = 5432

conn = psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT
)
cur = conn.cursor()

# ------------------------------
# Embedding modeli
# ------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# Veri modelleri
# ------------------------------
class Endpoint(BaseModel):
    module: str
    service: str
    method: str
    endpoint: str
    description: str
    parameters: list
    elementtypeid: int
    menuid: int
    explain: str

class Question(BaseModel):
    user_id: str
    query: str
    user_company: str
    user_name: str

# ------------------------------
# Conversation history fonksiyonları
# ------------------------------
def save_conversation(user_id, role, message_text,user_name,user_company):
    try:
        cur.execute("""
            INSERT INTO conversation_history (user_id, message_role, message_text,user_name,user_company)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, role, message_text, user_name, user_company))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def get_conversation_history(user_id, user_name, limit=3):
    try:
        cur.execute("""
            SELECT message_role, message_text
            FROM conversation_history
            WHERE user_id=%s AND
                user_name=%s AND 
                message_text IS NOT NULL AND 
                message_text <> '' AND
                created_at >= NOW() - interval '30 seconds' 
            ORDER BY created_at desc
            LIMIT %s
        """, (user_id, user_name, limit))
        rows = cur.fetchall()
        #return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
# ------------------------------
# Endpoint ekleme
# ------------------------------
@app.post("/add_endpoint")
def add_endpoint(data: Endpoint):
    try:
        embedding = embedding_model.encode(clean_text(data.description))
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        cur.execute("""
            SELECT id, description, embedding <-> %s::vector AS dist
            FROM api_endpoints
            ORDER BY dist ASC
            LIMIT 1
        """, (embedding_str,))
        best = cur.fetchone()

        if best and best[2] < 0.55:   # 0.45 cosine-distance eşiği
            return {
                "status": "exists",
                "message": f"Benzer açıklama zaten mevcut (benzerlik: {best[2]:.2f}).",
                "existing_id": best[0],
                "existing_description": best[1]
            }

        cur.execute("""
            INSERT INTO api_endpoints (module, service, method, endpoint, description, parameters, embedding,elementtypeid,menuid,explain)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector,%s,%s,%s)
        """, (
            data.module,
            data.service,
            data.method,
            data.endpoint,
            data.description,
            data.parameters,
            embedding_str,
            data.elementtypeid,
            data.menuid,
            data.explain
        ))
        conn.commit()
        return {"status": "success", "endpoint": data.endpoint}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
# Varsayımsal Python Kodu (FastAPI/Psycopg2 kullanılarak)
def add_business_rule(rule_text: str):
    try:
        # 1. Metni Vektörleştirme
        embedding = embedding_model.encode(rule_text)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        
        # 2. Veritabanına Kaydetme
        cur.execute("""
            INSERT INTO business_rules (rule_text, embedding)
            VALUES (%s, %s::vector)
        """, (rule_text, embedding_str))
        conn.commit()
        
        return {"status": "success", "eklenenbilgi": rule_text}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)          # fazla boşluk
    text = re.sub(r"[^\w\sçğıöşü]", " ", text)  # noktalama
    return text.strip()


def add_qa(q: str, a: str):
    emb = "[" + ",".join(map(str, embedding_model.encode(clean_text(q)))) + "]"
    cur.execute("""INSERT INTO qa_pairs (question, answer, embedding)
                   VALUES (%s, %s, %s::vector)""", (q, a, emb))
    conn.commit()
    return {"status": "ok"}



# ------------------------------
# Kullanıcı sorusu
# ------------------------------
@app.post("/ask")
def ask_question(data: Question):
    baslangic = time.time()
    print(f"ask-a girdi:{baslangic}")
    try:
        # 0️⃣ Yönetim Komutu Kontrolü
        if data.query.lower().startswith("bilgiekle:"):
            payload  = data.query[len("bilgiekle:"):].strip()
            if not payload:
                return {"status": "error", "response": "Hata: 'bilgiekle:' komutundan sonra eklenecek bir metin girilmelidir."}
            try:
                parts = re.split(r"\s+answer:\s*", payload, flags=re.I)
                prompt = parts[0][len("prompt:"):].strip()
                answer = parts[1].strip()
                if not prompt:
                    return {"status": "error", "response": "Hata: 'prompt:' kısmı boş bırakılamaz."}
                if not answer:
                    return {"status": "error", "response": "Hata: 'answer:' kısmı boş bırakılamaz."}

                emb_str = "[" + ",".join(map(str, embedding_model.encode(clean_text(prompt)))) + "]"
                #q_emb = "[" + ",".join(map(str, embedding_model.encode(clean_text(data.query)))) + "]"
                cur.execute("""SELECT question, embedding <-> %s::vector AS dist FROM qa_pairs ORDER BY dist ASC LIMIT 1""", (emb_str,))
                best = cur.fetchone()
                if best and best[1] < 0.60:   
                    return {"answer": f"Bu soru zaten mevcut (benzerlik: {best[1]:.2f}). Soru: {prompt}"}
            except Exception:
                return {"status": "error", "response": "Format hatası: prompt:... answer:... şeklinde girin."}
            return add_qa(prompt, answer)

        # 1️⃣ Kullanıcı mesajını DB'ye kaydet
        save_conversation(data.user_id, "user", data.query,data.user_name,data.user_company)

        # 2️⃣ Geçmiş mesajları al
        history = get_conversation_history(data.user_id, data.user_name, limit=1)

        print(f"====================================================================================")
        print(f"history:{history}")
        print(f"====================================================================================")

        # 3️⃣ Vektör Hazırlığı
        query_embedding = embedding_model.encode(clean_text(data.query))
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # 4️⃣ RAG: Endpoint ve İş Kuralı Araması
        cur.execute("""
                SELECT 
                    ae.module, ae.service, ae.method, ae.endpoint,
                    ae.description, ae.parameters,
                    ae.embedding <#> %s::vector AS dist,
                    ae.elementtypeid AS "elementtype",
                    ae.explain as "explain"
                FROM api_endpoints ae
                WHERE ae.elementtypeid IN (
                    SELECT id FROM elementtype WHERE aktif=1
                )
                ORDER BY dist ASC
                LIMIT 4
            """, (embedding_str,))
        endpoint_matches = cur.fetchall()

        # b) İş Kuralı Araması (Sadece kural metnini al)
        cur.execute("""SELECT question, answer, embedding <-> %s::vector AS dist
               FROM qa_pairs
               ORDER BY dist ASC LIMIT 3""", (embedding_str,))

        rule_matches = cur.fetchall()
        
        # 5️⃣ Konteksleri Formatlama ve Birleştirme (Tek ve Güçlü SYSTEM Prompt'u İçin)

        # Endpoint Konteksi (Zengin Bilgi)
        endpoint_filtered = [m for m in endpoint_matches if m[6] ];#< 0.6
        endpoint_context = ""
        if not endpoint_filtered:
            endpoint_context = "Kullanıcı sorusuyla alakalı bir API uç noktası bulunamadı."
        else:
            endpoint_context = "\n---\n".join([
                f"Endpoint: {m[3]} | Metod: {m[2]} | Modül: {m[0]} | Elementtype: {m[7]}\nAçıklama: {m[4]}\n{m[8]}\nParametreler: {m[5]}"
                for m in endpoint_matches
            ]) or "Kullanıcı sorusuyla alakalı bir API uç noktası (endpoint) bulunamadı."
        #print(f"endpointcontex : {endpoint_context}")
        
        # İş Kuralı Konteksi (Doğrudan Kural Metinleri)
        #rule_context = "\n- ".join([m[0] for m in rule_matches])
        print("====================================================")
        print("raw rule_matches:", rule_matches)
        print("====================================================")
        #rule__filtered = [m for m in rule_matches if m[1] < 0.65];
        rule__filtered = rule_matches;
        if not rule__filtered and rule_matches:
            rule__filtered = [rule_matches[0]] 
        rule_context = ""
        if rule__filtered:
            rule_context = "\n".join(
                    f"Soru {i+1}: {q.strip()}\nCevap {i+1}: {a.strip()}\n"
                    for i, (q, a, _) in enumerate(rule__filtered)
                )
        else:
            rule_context = "Kullanıcı sorusuyla alakalı bir iş kuralı bulunamadı."


        # 6️⃣ Tek, Güçlü ve Düzgün JSON Formatı Zorlayan System Prompt'u Oluşturma
        system_content = f"""
Sen Boyut Bilgisayar adına nervus programı için görev yapan bir yapay zeka asistanısın.

Vereceğin cevaplar her zaman {{"response":"Yanıtın mesaj kısmı.","system_debug":"sisteme vermek istediğin mesaj varsa","bilgiiste":**true/false**,"endpoint":"...", "metod":"...","modul":"...","elementtype":"..."}} şeklinde yanlızca json formatında olmalı. Chat için bir mesajın varsa **response** keyinin value kısmına yaz.

Eğer kullanıcının sorduğu soru aşağıda vereceğim [REHBER] olarak belirttiğim listedeki bir kayıt ile eşleşirse onu Json içinde dön bu kayıtlardan referans al.

Eğer kullanıcı bir liste istiyorsa:
-[ENDPOINTLER] olarak belirttiğim listede bir endpoint varsa endpointi hiçbirşekilde değiştirmeden jsondaki endpoint keyine value olarak yaz.Kesinlike endpointi Büyük/küçük harf veya kelime değiştirme olduğu gibi kalsın. Mesajın varsa response keyine olduğu gibi yaz. Elementtype kısmınıda elementtype keyine olduğu gibi yaz değiştirme
-Ben senin bu gönderdiğin endpoint ile data çekip göstereceğim.
-Eğer endpoint var ise bilgiiste keyi true olacak.
-Gündelik sohbet konularında cevap ver ama rehberde yoksa ve gündelik sohbet dışında ise Üzgünüm bu konuda cevap veremiyorum. Başka nasıl yardımcı olabilirim. şeklinde cevap ver

--- [ENDPOINTLER]  ---
{endpoint_context}

--- [REHBER] ---
{rule_context}
"""
        system_message = {"role": "system", "content": system_content}
        print("=======================================")
        print(f"rule_context:{rule_context}")
        print("=======================================")
        print(f"history:{history}")
        print("=============================================================================")
        print(f"history:{system_message}")
        print("=============================================================================")
        # 7️⃣ Tüm mesajları birleştir (SADECE TEK SYSTEM MESAJI + GEÇMİŞ)
        messages = [system_message] + history 
        #return endpoint_context;
        bitis = time.time()
        gecen_sure = bitis - baslangic
        print(f"Geçen süre: {gecen_sure:.3f} saniye ({gecen_sure*1000:.0f} milisaniye)")
        #print("-----------------------------------")
        #print(f"Endpointler:{endpoint_context}")
        #print(f"Rehber:{rule_context}")
        #print("---------------------------------------------------------")
        #print(f"HİSTORY:{history}")
        ("---------------------------------------------------------")
        #print(f"System mesajı:{messages}")
        baslangic = time.time()
        client = OpenAI(api_key=OPENAI_KEY)
        # 8️⃣ chatgpt çağrısı
        response = client.responses.create(
            model="gpt-4.1-mini",
            input = messages
        )
        
        bitis = time.time()
        gecen_sure = bitis - baslangic
        print(f"Model Geçen süre: {gecen_sure:.3f} saniye ({gecen_sure*1000:.0f} milisaniye)")
        #print("--------------------------------")
        #print(f"dönen:{response}")
        #print("---------------------------------------------------------")
        
        assistant_text = response.output_text.strip()
        print(f"assistant_text={assistant_text}")
        # 9️⃣ Yanıtı DB'ye kaydet (API çağrısı ve JSON işlemeyi kaldırdım, sadece yanıtı döndürdüm)
        # Bu kısım daha sonra Ajan mantığı olarak ayrı bir fonksiyonda ele alınmalıdır.
        save_conversation(data.user_id, "assistant", assistant_text,data.user_name, data.user_company)

        return {"answer": assistant_text}
    
        ##buradan aşağısı şuan işlemiyor ama endpoint varsa bilgi çek gibi düşündüm
        api_data = None
        try:
            parsed = json.loads(assistant_text)
            if parsed.get("endpointlist"):
                # API URL, senin verdiğin base URL + endpoint
                base_url = "http://192.168.3.101:6969"
                endpoint_list = parsed.get("endpointlist", [])
                if isinstance(endpoint_list, list):
                    # sadece ilk endpointi al
                    endpoint_path = endpoint_list[0]
                else:
                    endpoint_path = endpoint_list
                print("--------------------------------------")
                print(f"endpoint:{endpoint_path}")
                print("--------------------------------------")
                endpoint_url = f"{base_url}{endpoint_path}"
                print(f"endpoint_url:{endpoint_url}")
                print("--------------------------------------")
                # DB’den endpoint parametrelerini al
                cur.execute("""
                    SELECT parameters
                    FROM api_endpoints
                    WHERE endpoint = %s
                """, (endpoint_url,))
                
                

                params = {
                            "FaturaTarihiBas": "2025-09-01",
                            "FaturaTarihiBit": "2025-10-28"
                        }
                # API key header
                API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1laWQiOiIxIiwidW5pcXVlX25hbWUiOiJhZG1pbiIsInJvbGUiOiJbXSIsIlN1YmVJZCI6IjEiLCJDZXBTdWJlSWQiOiIyODEiLCJuYmYiOjE3NjA0NDk0MzYsImV4cCI6MTc2MzEyNzgzNiwiaWF0IjoxNzYwNDQ5NDM2LCJpc3MiOiJCb3l1dCIsImF1ZCI6IkJveXV0In0.IpS9v-soj6ea62tXGL40NvwbV9j3Suh1KCYHBvbiey0"
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                # API çağrısı
                try:
                    api_response = requests.get(endpoint_url, params=params, headers=headers)
                    try:
                        api_data = api_response.json()
                    except ValueError:
                        api_data = {"raw_response": api_response.text}
                except Exception as api_e:
                    print("API çağrısı sırasında hata:", api_e)
                    api_data = {"error": str(api_e)}
                # Dönen veriyi assistant_text içine ekle
                api_data = api_response.json()
                return api_data
                assistant_text = json.dumps(parsed, ensure_ascii=False)
                return {"answer": assistant_text}
        except Exception as e:
            # Hata olsa bile assistant_text’i döndür
            print("API çağrısı sırasında hata:", e)
            return e



        # 8️⃣ Endpoint çağrısı gerekiyorsa
        try:
            parsed = json.loads(assistant_text)
            if parsed.get("endpointvarsa"):
                endpoint_url = parsed["endpointvarsa"]
                params = parsed.get("parameters", {})
                api_response = requests.get(endpoint_url, params=params)
                parsed["resonse"] = api_response.json()
                assistant_text = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        # 9️⃣ Yanıtı DB'ye kaydet
        save_conversation(data.user_id, "assistant", assistant_text,data.user_name,data.user_company)

        return {"answer": assistant_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pairs")
def get_qa_pairs():
    cur = conn.cursor()
    cur.execute("""
        SELECT id, question, answer, created_at
        FROM qa_pairs
        ORDER BY created_at DESC
    """)
    
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]

    data = [dict(zip(cols, row)) for row in rows]

    return data