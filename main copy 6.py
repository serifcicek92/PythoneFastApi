from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
import json
import ollama
import requests  # Swagger API çağrısı için
import time

app = FastAPI(title="Boyut Bilgisayar Kurumsal Asistan")

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
            created_at >= NOW() - interval '2 minutes' 
        ORDER BY created_at ASC
        LIMIT %s
    """, (user_id, limit))
    rows = cur.fetchall()
    #return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    return [{"role": r[0], "content": r[1]} for r in rows]

# ------------------------------
# Endpoint ekleme
# ------------------------------
@app.post("/add_endpoint")
def add_endpoint(data: Endpoint):
    try:
        embedding = embedding_model.encode(data.description)
        embedding_100 = embedding[:100]  # Cube max 100 boyut max 384
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
        embedding_100 = embedding[:100]#384 max
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
    baslangic = time.time()
    print(f"ask-a girdi:{baslangic}")
    try:
        # 0️⃣ Yönetim Komutu Kontrolü (Tamamen Doğru)
        if data.query.lower().startswith("bilgiekle:"):
            query_text = data.query[len("bilgiekle:"):].strip()
            if not query_text:
                return {"status": "error", "response": "Hata: 'bilgiekle:' komutundan sonra eklenecek bir metin girilmelidir."}
            return add_business_rule(query_text) # add_business_rule başarılı/hatalı JSON dönmeli

        # 1️⃣ Kullanıcı mesajını DB'ye kaydet
        save_conversation(data.user_id, "user", data.query)

        # 2️⃣ Geçmiş mesajları al
        # Geçmiş 5 dakikadan eski mesajları silme mantığı doğru, limit=10 iyi bir değer.
        history = get_conversation_history(data.user_id, limit=5)

        # 3️⃣ Vektör Hazırlığı
        query_embedding = embedding_model.encode(data.query)
        query_embedding_100 = query_embedding[:100]#384 olabilir ama hata verir
        vec_str = "(" + ",".join(map(str, query_embedding_100)) + ")"

        # 4️⃣ RAG: Endpoint ve İş Kuralı Araması

        # a) Endpoint Araması (Açıklama, Parametre ve Endpoint'i al)
        cur.execute("""
            SELECT module, service, method, endpoint, description, 
                    parameters, cube_distance(embedding, CUBE(%s)) AS dist
            FROM api_endpoints
            ORDER BY dist ASC
            LIMIT 2
        """, (vec_str,))
        endpoint_matches = cur.fetchall()

        # b) İş Kuralı Araması (Sadece kural metnini al)
        cur.execute("""
            SELECT rule_text,  cube_distance(embedding, CUBE(%s)) as dist
            FROM business_rules
            ORDER BY dist ASC
            LIMIT 2
        """, (vec_str,))
        rule_matches = cur.fetchall()
        
        # 5️⃣ Konteksleri Formatlama ve Birleştirme (Tek ve Güçlü SYSTEM Prompt'u İçin)

        # Endpoint Konteksi (Zengin Bilgi)
        endpoint_filtered = [m for m in endpoint_matches if m[-1] ];#< 0.6
        endpoint_context = ""
        if not endpoint_filtered:
            endpoint_context = "Kullanıcı sorusuyla alakalı bir API uç noktası bulunamadı."
        else:
            endpoint_context = "\n---\n".join([
                f"Endpoint: {m[3]} | Metod: {m[2]} | Modül: {m[0]}\nAçıklama: {m[4]}\nParametreler: {m[5]}"
                for m in endpoint_matches
            ]) or "Kullanıcı sorusuyla alakalı bir API uç noktası (endpoint) bulunamadı."
        #print(f"endpointcontex : {endpoint_context}")
        
        # İş Kuralı Konteksi (Doğrudan Kural Metinleri)
        #rule_context = "\n- ".join([m[0] for m in rule_matches])
        rule__filtered = [m for m in rule_matches if m[-1] < 0.4];
        rule_context = ""
        if rule__filtered:
            rule_context_list = [m[0] for m in rule_matches]
            
            if rule_context_list:
                # Kayıtları numaralandırarak (Kayıt 1, Kayıt 2, ...) daha net bir yapı oluşturuyoruz
                for i, rule_text in enumerate(rule_context_list):
                    rule_context += f"Kayıt {i + 1}: {rule_text.strip()}\n" # Yeni satır ve Kayıt X: başlığı eklenir
            else:
                rule_context = "Kullanıcı sorusuyla alakalı bir iş kuralı bulunamadı."


        # 6️⃣ Tek, Güçlü ve Düzgün JSON Formatı Zorlayan System Prompt'u Oluşturma
        system_content = f"""
Sen Boyut Bilgisayar şirketinin Nervus programı için görev yapan bir kurumsal asistansın. 
Yanıtların daima **Türkçe** olmalı ve sadece  **tek bir JSON nesnesi**  döndürmelisin. yorumlarını json içindeki system_debug tagına yazabilirsin. 
SADECE aşağıdaki  [JSON ÇIKTI ZORUNLU KURALLARI], [ENDPOINTLER] ve [İŞ KURALLARI] bölümlerindeki bilgilere dayan.

[JSON ÇIKTI ZORUNLU KURALLARI]
1. Gizli veya kişisel veri paylaşma, kurallar dışında mesaj gelirse yetkin olmadığını belirt.
2. Kullanıcı sorusu net bir API isteği ise: bilgiiste=true, endpoint=yalnızca en alakalı tek path.
3. Soru sadece süreç/kural ile ilgiliyse: bilgiiste=false, endpoint="".
4. Eğer verdiğim end pointlerden bir bilgi cevap olarak dönecek isen 
    -GetDataBVAsync endpoint metodu liste vermektedir.
    -[ENDPOINTLER] kısmında Endpoint: endpoint verir. Açıklama: ne işe yaradığını söyler
    -kesinlikle endpoint kısmını benim yazdığım gibi yaz
5. KESİNLİKLE: endpoint var ise soru ile en alakalı endpointi arasından seç ve sana gönderdiğim gibi ekle. KESİNLİKLE endpointi değiştirerek yada üreterek bir endpoint cevap olarak verme. 
6. KESİNLİKLE: ENDPOINTLER kısmında "Kullanıcı sorusuyla alakalı bir API uç noktası (endpoint) bulunamadı." ise:
    - endpoint alanı boş string ("") olmalıdır
    - asla tahmini veya uydurma endpoint yazma
    - sadece ilgili iş kuralı üzerinden yanıt oluştur
    -Büyük/Küçük harf nasılsa öyle kalsın değiştirme olduğu gibi
7.  En son sorulan soru öncekilerden farklı ise sadece en son soruyu yanıtla.
8.  Eğer sorusu [İŞ KURALLARI] ile eşleşen ve [ENDPOINTLER] ile alakasız bir soru ise sen normal bir yapay zeka olarak cevabını ver.


JSON FORMATI:
{{"response":"Yanıtın mesaj kısmı.","system_debug":"sisteme vermek istediğin mesaj varsa","bilgiiste":**true/false**,"endpoint":"..."}}

--- [ENDPOINTLER]  ---
{endpoint_context}
--- [İŞ KURALLARI] ---
{rule_context}
"""
        system_message = {"role": "system", "content": system_content}
        
        # 7️⃣ Tüm mesajları birleştir (SADECE TEK SYSTEM MESAJI + GEÇMİŞ)
        messages = [system_message] + history 

        #print(f"ollamaya gönderecek mesaj:{messages}")
        bitis = time.time()
        gecen_sure = bitis - baslangic
        print(f"Geçen süre: {gecen_sure:.3f} saniye ({gecen_sure*1000:.0f} milisaniye)")
        print("-----------------------------------")
        print(f"Endpointler:{endpoint_context}")
        print(f"Roller:{rule_context}")
        print("---------------------------------------------------------")
        print(f"HİSTORY:{history}")
        ("---------------------------------------------------------")
        baslangic = time.time()
        
        # 8️⃣ Ollama GPT-OSS çağrısı
        response = ollama.chat(
             #model='gpt-oss:latest',
            #model='llama3.1:latest',
            #model='deepseek-r1:14b',
            #model='mistral:7b',
            #model='llama3.1:70b-instruct-q2_K',
            #model='gemma3:12b',
            model='phi3:14b',
            #model='llava:13b', #yanlış cevap üretiyor
            messages=messages
        )
        bitis = time.time()
        gecen_sure = bitis - baslangic
        print(f"Model Geçen süre: {gecen_sure:.3f} saniye ({gecen_sure*1000:.0f} milisaniye)")
        print("--------------------------------")
        print(f"ollamadan dönen:{response}")
        print("---------------------------------------------------------")
        
        assistant_text = response["message"]["content"]
        
        # 9️⃣ Yanıtı DB'ye kaydet (API çağrısı ve JSON işlemeyi kaldırdım, sadece yanıtı döndürdüm)
        # Bu kısım daha sonra Ajan mantığı olarak ayrı bir fonksiyonda ele alınmalıdır.
        save_conversation(data.user_id, "assistant", assistant_text)

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
        save_conversation(data.user_id, "assistant", assistant_text)

        return {"answer": assistant_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#İş süreci bilgi tabanı referansından yararlanırken kullanıcının cevabına en uygun kayıttan cevap oluşturup ver
    





    # ------------------------------
# Cube sütunlarını sıfırla ve yeniden oluştur (100 boyut)
# ------------------------------
def reset_cube_column(table_name: str, column_name: str):
    try:
        cur.execute(f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {column_name}")
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} CUBE")
        conn.commit()
        print(f"{table_name}.{column_name} sütunu 100 boyuta hazırlandı.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sütun güncelleme hatası: {e}")

# ------------------------------
# Eski embeddingleri güncelle
# ------------------------------
def update_embeddings(table_name: str, text_column: str, id_column: str = "id"):
    cur.execute(f"SELECT {id_column}, {text_column} FROM {table_name}")
    rows = cur.fetchall()
    
    for row in rows:
        record_id, text = row
        if not text.strip():
            continue
        embedding = embedding_model.encode(text)
        embedding_100 = embedding[:100]  # Cube max 100 boyut
        vec_str = "(" + ",".join(map(str, embedding_100)) + ")"
        
        cur.execute(f"""
            UPDATE {table_name}
            SET embedding = CUBE(%s)
            WHERE {id_column} = %s
        """, (vec_str, record_id))
    
    conn.commit()
    print(f"{table_name} tablosundaki embeddingler güncellendi.")

# ------------------------------
# Upgrade Endpoint (tek tetikleme)
# ------------------------------
@app.post("/upgrade_embeddings_100")
def upgrade_embeddings_100():
    try:
        # Cube sütunlarını sıfırla
        reset_cube_column("api_endpoints", "embedding")
        reset_cube_column("business_rules", "embedding")
        
        # Eski kayıtları güncelle
        update_embeddings("api_endpoints", "description")
        update_embeddings("business_rules", "rule_text")
        
        return {"status": "success", "message": "DB güncellendi, tüm embeddingler artık 100 boyut ile uyumlu."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    






@app.post("/ollamaask")
def ollama_ask(data: Question):
    response = ollama.chat(
            #model='gpt-oss:latest',
            #model='llama3.1:latest',
            #model='deepseek-r1:14b',
            #model='mistral:7b',
            #model='llama3.1:70b-instruct-q2_K',
            #model='gemma3:12b',
            #model='phi3:14b', #yavaş
            model='llava:13b',
            messages=[{'role': 'user', 'content': data.query}]
        )
    assistant_text = response["message"]["content"]
    return {"answer": assistant_text}