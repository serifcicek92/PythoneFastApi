from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
import json
import ollama
import requests  # Swagger API çağrısı için

app = FastAPI(title="Boyut Bilgisayar Kurumsal Asistan Demo")

# ------------------------------
# PostgreSQL bağlantısı
# ------------------------------
DB_HOST = "localhost"
DB_NAME = "ollamadb"
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

class Question(BaseModel):
    user_id: str
    query: str

# ------------------------------
# Conversation history fonksiyonları
# ------------------------------
def save_conversation(user_id, role, message_text):
    cur.execute("""
        INSERT INTO conversation_history (user_id, message_role, message_text)
        VALUES (%s, %s, %s)
    """, (user_id, role, message_text))
    conn.commit()

def get_conversation_history(user_id, limit=3):
    cur.execute("""
        SELECT message_role, message_text
        FROM conversation_history
        WHERE user_id=%s AND 
            message_text IS NOT NULL AND 
            message_text <> '' AND
            created_at >= NOW() - interval '5 minutes'
        ORDER BY id DESC
        LIMIT %s
    """, (user_id, limit))
    rows = cur.fetchall()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

# ------------------------------
# Endpoint ekleme
# ------------------------------
@app.post("/add_endpoint")
def add_endpoint(data: Endpoint):
    try:
        embedding = embedding_model.encode(data.description)
        embedding_100 = embedding[:100]  # Cube max 100 boyut
        vec_str = "(" + ",".join(map(str, embedding_100)) + ")"

        cur.execute("""
            INSERT INTO api_endpoints (module, service, method, endpoint, description, parameters, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, CUBE(%s))
        """, (
            data.module,
            data.service,
            data.method,
            data.endpoint,
            data.description,
            data.parameters,
            vec_str
        ))
        conn.commit()
        return {"status": "success", "endpoint": data.endpoint}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Varsayımsal Python Kodu (FastAPI/Psycopg2 kullanılarak)
def add_business_rule(rule_text: str):
    try:
        # 1. Metni Vektörleştirme
        embedding = embedding_model.encode(rule_text)
        embedding_100 = embedding[:100]
        vec_str = "(" + ",".join(map(str, embedding_100)) + ")"
        
        # 2. Veritabanına Kaydetme
        cur.execute("""
            INSERT INTO business_rules (rule_text, embedding)
            VALUES (%s, CUBE(%s))
        """, (rule_text, vec_str))
        conn.commit()
        
        return {"status": "success", "eklenenbilgi": rule_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ------------------------------
# Kullanıcı sorusu
# ------------------------------
@app.post("/ask")
def ask_question(data: Question):
    try:
        # 1️⃣ Kullanıcı mesajını DB'ye kaydet
        save_conversation(data.user_id, "user", data.query)

        # 2️⃣ Geçmiş mesajları al
        history = get_conversation_history(data.user_id, limit=10)

        # 3️⃣ Kullanıcı sorgusunu embedding'e çevir ve ilk 100 boyutu al
        query_embedding = embedding_model.encode(data.query)
        query_embedding_100 = query_embedding[:100]
        vec_str = "(" + ",".join(map(str, query_embedding_100)) + ")"

        # 4️⃣ En yakın endpointleri Cube ile ara
        cur.execute("""
            SELECT module, service, method, endpoint, description, parameters
            FROM api_endpoints
            ORDER BY cube_distance(embedding, CUBE(%s)) ASC
            LIMIT 2
        """, (vec_str,))
        matches = cur.fetchall()
        endpoint_message = {
            "role": "system",
            "content": f"En uygun endpointler: {[m[3] for m in matches]}"
        }


        # 5️⃣ System rolü mesajı
        system_message = {
            "role": "system",
            "content": """
            Sen Boyut Bilgisayar şirketinin Nervus programı için görev yapan bir kurumsal asistansın. Kurallar dışında bir mesaj gelirse yetkin olmadığını belirt.

            Kurallar:
            1. Gizli veya kişisel verileri paylaşma.
            2. Her zaman güvenli ve şirket politikalarına uygun yanıt ver.
            3. Çıktı daima json formatında olmalı
            4. Cevabını JSON formatında döndür:
            {"resonse":"mesaj","system":"systemebilgi","bilgiiste":"endpoint varsa true","endpointlist":"" #En uygun endpointler
            }
            """
        }

        

        #dbdeki oluşturduğum bilgilerden bilgi
        cur.execute("""
            SELECT rule_text
            FROM business_rules
            ORDER BY cube_distance(embedding, CUBE(%s)) ASC
            LIMIT 2
        """, (vec_str,))
        mylearnmatches = cur.fetchall()
        systemhelper_message = {
            "Kendi oluşturduğum bilgilerden bulduğum yanıta göre doğru yanıtı üret. "
            "Aşağıdaki kuralları kullan:\n\n"
            f"BULUNAN KURALLAR: {[m[0] for m in mylearnmatches]}"
        }

        # 6️⃣ Tüm mesajları birleştir
        messages = [system_message] + [endpoint_message] + [systemhelper_message] + history 

        print(f"ollamaya gönderecek mesaj:{messages}")
        print("---------------------------------------------------------")
        # 7️⃣ Ollama GPT-OSS çağrısı
        response = ollama.chat(
            #model='gpt-oss:latest',
            model='llama3.1:latest',
            messages=messages
        )
        print(f"ollamadan dönen:{response}")
        print("---------------------------------------------------------")
        
        assistant_text = response["message"]["content"]
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
        save_conversation(data.user_id, "assistant", assistant_text)

        return {"answer": assistant_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
