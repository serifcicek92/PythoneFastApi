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

app = FastAPI(title="Boyut Bilgisayar Kurumsal Asistan")

#burası llama kkullanılacaksa açılacak ramim yetmedi cpu çalışıyor gpu sorunlu
# print("🧠 Model yükleniyor... (birkaç saniye sürebilir)")
# llm = Llama(
#     model_path="C:\\llama_cpp\\models\\Qwen3-14B-Q4_K_M.gguf",
#     n_gpu_layers=20,   # GPU’ya kaç katman yüklenecek
#     n_ctx=4096,        # maksimum context (token penceresi)
#     n_threads=6,       # işlemci çekirdek sayısı
#     verbose=True       # log detaylarını göster
# )
# print("✅ Model yüklendi!")

print("🧠 Model yükleniyor... (birkaç saniye sürebilir)")
llm = Llama(
    model_path="C:\\huggingface\\generated\\gpt2-turkish\\gpt2-turkish.gguf",  # kendi modelin
    n_ctx=2048,        # GPT-2 için genelde yeterli
    n_threads=6,        # CPU çekirdek sayın kadar
    n_gpu_layers=0,     # GPU devre dışı, CPU kullanılıyor
    verbose=False
)
print("✅ GPT-2 Türkçe model yüklendi!")


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

class Question(BaseModel):
    user_id: str
    query: str

# ------------------------------
# Conversation history fonksiyonları
# ------------------------------
def save_conversation(user_id, role, message_text):
    try:
        cur.execute("""
            INSERT INTO conversation_history (user_id, message_role, message_text)
            VALUES (%s, %s, %s)
        """, (user_id, role, message_text))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def get_conversation_history(user_id, limit=3):
    try:
        cur.execute("""
            SELECT message_role, message_text
            FROM conversation_history
            WHERE user_id=%s AND 
                message_text IS NOT NULL AND 
                message_text <> '' AND
                created_at >= NOW() - interval '30 seconds' 
            ORDER BY created_at ASC
            LIMIT %s
        """, (user_id, limit))
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
        embedding = embedding_model.encode(data.description)
        
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        cur.execute("""
            INSERT INTO api_endpoints (module, service, method, endpoint, description, parameters, embedding,elementtypeid,menuid)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector,%s,%s)
        """, (
            data.module,
            data.service,
            data.method,
            data.endpoint,
            data.description,
            data.parameters,
            embedding_str,
            data.elementtypeid,
            data.menuid
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
        history = get_conversation_history(data.user_id, limit=1)

        # 3️⃣ Vektör Hazırlığı
        query_embedding = embedding_model.encode(data.query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # 4️⃣ RAG: Endpoint ve İş Kuralı Araması
        cur.execute("""
                SELECT 
                    ae.module, ae.service, ae.method, ae.endpoint,
                    ae.description, ae.parameters,
                    ae.embedding <#> %s::vector AS dist,
                    ae.elementtypeid AS "elementtype"
                FROM api_endpoints ae
                WHERE ae.elementtypeid IN (
                    SELECT id FROM elementtype WHERE aktif=1
                )
                ORDER BY dist ASC
                LIMIT 4
            """, (embedding_str,))
        endpoint_matches = cur.fetchall()

        # b) İş Kuralı Araması (Sadece kural metnini al)
        cur.execute("""
            SELECT rule_text,  
                   embedding <-> %s::vector AS dist
            FROM business_rules
            ORDER BY dist ASC
            LIMIT 2
        """, (embedding_str,))
        rule_matches = cur.fetchall()
        
        # 5️⃣ Konteksleri Formatlama ve Birleştirme (Tek ve Güçlü SYSTEM Prompt'u İçin)

        # Endpoint Konteksi (Zengin Bilgi)
        endpoint_filtered = [m for m in endpoint_matches if m[6] ];#< 0.6
        endpoint_context = ""
        if not endpoint_filtered:
            endpoint_context = "Kullanıcı sorusuyla alakalı bir API uç noktası bulunamadı."
        else:
            endpoint_context = "\n---\n".join([
                f"Endpoint: {m[3]} | Metod: {m[2]} | Modül: {m[0]} | Elementtype: {m[7]}\nAçıklama: {m[4]}\nParametreler: {m[5]}"
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
Sen Boyut Bilgisayar adına nervus programı için görev yapan bir yapay zeka asistanısın.

Vereceğin cevaplar her zaman {{"response":"Yanıtın mesaj kısmı.","system_debug":"sisteme vermek istediğin mesaj varsa","bilgiiste":**true/false**,"endpoint":"...", "metod":"...","modul":"...","elementtype":"..."}} şeklinde yanlızca json formatında olmalı. Chat için bir mesajın varsa **response** keyinin value kısmına yaz.

Eğer kullanıcının sorduğu soru aşağıda vereceğim [REHBER] olarak belirttiğim listedeki bir kayıt ile eşleşirse onu Json içinde dön bu kayıtlardan referans al.

Eğer kullanıcı bir liste istiyorsa:
-[ENDPOINTLER] olarak belirttiğim listede bir endpoint varsa endpointi hiçbirşekilde değiştirmeden jsondaki endpoint keyine value olarak yaz.Kesinlike endpointi Büyük/küçük harf veya kelime değiştirme olduğu gibi kalsın. Mesajın varsa response keyine olduğu gibi yaz. Elementtype kısmınıda elementtype keyine olduğu gibi yaz değiştirme
-Ben senin bu gönderdiğin endpoint ile data çekip göstereceğim.
-Eğer endpoint var ise bilgiiste keyi true olacak.

--- [ENDPOINTLER]  ---
{endpoint_context}

--- [REHBER] ---
{rule_context}
"""
        system_message = {"role": "system", "content": system_content}
        
        # 7️⃣ Tüm mesajları birleştir (SADECE TEK SYSTEM MESAJI + GEÇMİŞ)
        messages = [system_message] + history 
        #return endpoint_context;
        #print(f"ollamaya gönderecek mesaj:{messages}")
        bitis = time.time()
        gecen_sure = bitis - baslangic
        print(f"Geçen süre: {gecen_sure:.3f} saniye ({gecen_sure*1000:.0f} milisaniye)")
        print("-----------------------------------")
        print(f"Endpointler:{endpoint_context}")
        print(f"Rehber:{rule_context}")
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
            #model='phi3:14b',
            #model='llava:13b', #yanlış cevap üretiyor
            model = 'qwen2.5:1.5b',
            #model = 'phi3:3.8b',
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








# 8️⃣ KoboldCPP GPT çağrısı
def query_koboldcpp(messages):
    url = "http://127.0.0.1:5001/api/v1/generate"  # KoboldCPP varsayılan API
    prompt = ""
    for msg in messages:
        prompt += f"{msg['role'].upper()}: {msg['content']}\n"
    prompt += "ASSISTANT:"

    payload = {
        "prompt": prompt,
        "max_context_length": 2048,
        "max_length": 512,
        "temperature": 0.7,
        "stop_sequence": ["USER:", "SYSTEM:"],
        "stream": False
    }

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="KoboldCPP API hatası")
    data = response.json()
    return data.get("results", [{}])[0].get("text", "")







# ------------------------------
# Kullanıcı sorusu llama 
# ------------------------------
@app.post("/askllama_cpp_local")
def ask_question_local(data: Question):
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
        history = get_conversation_history(data.user_id, limit=1)

        # 3️⃣ Vektör Hazırlığı
        query_embedding = embedding_model.encode(data.query)
        query_embedding_100 = query_embedding[:100]#384 olabilir ama hata verir
        vec_str = "(" + ",".join(map(str, query_embedding_100)) + ")"

        # 4️⃣ RAG: Endpoint ve İş Kuralı Araması

        # a) Endpoint Araması (Açıklama, Parametre ve Endpoint'i al)
        cur.execute("""
            SELECT ae.module, ae.service, ae.method, ae.endpoint, ae.description, 
                   ae.parameters, cube_distance(ae.embedding, CUBE(%s)) AS dist,
                   ae.elementtypeid as "elementtype"
            FROM api_endpoints ae
            left join elementtype e on e.id = ae.elementtypeid  and e.aktif=1
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
                f"Endpoint: {m[3]} | Metod: {m[2]} | Modül: {m[0]} | Elementtype: {m[7]}\nAçıklama: {m[4]}\nParametreler: {m[5]}"
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
Sen Boyut Bilgisayar adına nervus programı için görev yapan bir yapay zeka asistanısın.

Vereceğin cevaplar her zaman {{"response":"Yanıtın mesaj kısmı.","system_debug":"sisteme vermek istediğin mesaj varsa","bilgiiste":**true/false**,"endpoint":"...", "metod":"...","modul":"...","elementtype":"..."}} şeklinde yanlızca json formatında olmalı. Chat için bir mesajın varsa **response** keyinin value kısmına yaz.

Eğer kullanıcının sorduğu soru aşağıda vereceğim [REHBER] olarak belirttiğim listedeki bir kayıt ile eşleşirse onu Json içinde dön bu kayıtlardan referans al.

Eğer kullanıcı bir liste istiyorsa:
-[ENDPOINTLER] olarak belirttiğim listede bir endpoint varsa endpointi hiçbirşekilde değiştirmeden jsondaki endpoint keyine value olarak yaz.Kesinlike endpointi Büyük/küçük harf veya kelime değiştirme olduğu gibi kalsın. Mesajın varsa response keyine olduğu gibi yaz. Elementtype kısmınıda elementtype keyine olduğu gibi yaz değiştirme
-Ben senin bu gönderdiğin endpoint ile data çekip göstereceğim.
-Eğer endpoint var ise bilgiiste keyi true olacak.

--- [ENDPOINTLER]  ---
{endpoint_context}

--- [REHBER] ---
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
        response = llm.create_chat_completion(
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
        bitis = time.time()
        gecen_sure = bitis - baslangic
        print(f"Model Geçen süre: {gecen_sure:.3f} saniye ({gecen_sure*1000:.0f} milisaniye)")
        print("--------------------------------")
        print(f"ollamadan dönen:{response}")
        print("---------------------------------------------------------")
        assistant_raw = response["choices"][0]["message"]["content"]

        cleaned_text = re.sub(r"<think>.*?</think>", "", assistant_raw, flags=re.DOTALL).strip()
        
        try:
            assistant_text = json.loads(cleaned_text)
        except json.JSONDecodeError:
            assistant_text = {"response": cleaned_text}

        #assistant_text = response#response["choices"][0]["message"]["content"]
        
        # 9️⃣ Yanıtı DB'ye kaydet (API çağrısı ve JSON işlemeyi kaldırdım, sadece yanıtı döndürdüm)
        # Bu kısım daha sonra Ajan mantığı olarak ayrı bir fonksiyonda ele alınmalıdır.
        #save_conversation(data.user_id, "assistant", assistant_text)

        return {"answer": assistant_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Kullanıcı sorusu llama 
# ------------------------------
@app.post("/askllama_cpp")
def ask_question_llama_cpp(data: Question):
    import requests, json, time

    
    baslangic = time.time()
    print(f"askllama_cpp çağrıldı: {baslangic}")

    try:
        # 1️⃣ Yönetim komutu kontrolü
        if data.query.lower().startswith("bilgiekle:"):
            query_text = data.query[len("bilgiekle:"):].strip()
            if not query_text:
                return {"status": "error", "response": "Hata: 'bilgiekle:' komutundan sonra eklenecek bir metin girilmelidir."}
            return add_business_rule(query_text)

        # 2️⃣ Mesaj kaydet
        save_conversation(data.user_id, "user", data.query)

        # 3️⃣ Geçmiş konuşmayı al
        history = get_conversation_history(data.user_id, limit=1)

        # 4️⃣ Embedding hesapla
        query_embedding = embedding_model.encode(data.query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # 5️⃣ API endpoint arama
        cur.execute("""
                SELECT 
                    ae.module, ae.service, ae.method, ae.endpoint,
                    ae.description, ae.parameters,
                    ae.embedding <#> %s::vector AS dist,
                    ae.elementtypeid AS "elementtype"
                FROM api_endpoints ae
                WHERE ae.elementtypeid IN (
                    SELECT id FROM elementtype WHERE aktif=1
                )
                ORDER BY dist ASC
                LIMIT 4
            """, (embedding_str,))
        endpoint_matches = cur.fetchall()

        # 6️⃣ İş kuralı arama
        cur.execute("""
            SELECT rule_text,  
                   embedding <-> %s::vector AS dist
            FROM business_rules
            ORDER BY dist ASC
            LIMIT 2
        """, (embedding_str,))
        rule_matches = cur.fetchall()

        # Endpoint context
        endpoint_filtered = [m for m in endpoint_matches if m[6]]
        endpoint_context = "\n---\n".join([
            f"Endpoint: {m[3]} | Metod: {m[2]} | Modül: {m[0]} | Elementtype: {m[7]}\nAçıklama: {m[4]}\nParametreler: {m[5]}"
            for m in endpoint_filtered
        ]) or "Kullanıcı sorusuyla alakalı bir API uç noktası bulunamadı."

        # Rule context
        rule_filtered = [m for m in rule_matches if m[-1] < 0.4]
        rule_context = ""
        if rule_filtered:
            for i, r in enumerate(rule_filtered):
                rule_context += f"Kayıt {i + 1}: {r[0].strip()}\n"
        else:
            rule_context = "Kullanıcı sorusuyla alakalı bir iş kuralı bulunamadı."

        # System mesajı
        system_content = f"""
Sen Boyut Bilgisayar adına nervus programı için görev yapan bir yapay zeka asistanısın.

Vereceğin cevaplar her zaman {{"response":"Yanıtın mesaj kısmı.","system_debug":"sisteme vermek istediğin mesaj varsa","bilgiiste":true/false,"endpoint":"...","metod":"...","modul":"...","elementtype":"..."}} formatında olmalı.

--- [ENDPOINTLER]  ---
{endpoint_context}

--- [REHBER] ---
{rule_context}
"""
        system_message = {"role": "system", "content": system_content}

        # Geçmiş mesajları birleştir
        # messages = [system_message] + history + [{"role": "user", "content": data.query}]
        messages = [system_message] + [{"role": "user", "content": data.query}]


        # 7️⃣ Prompt hazırlığı
        prompt = ""
        # for msg in messages:
        #     prompt += f"{msg['role'].upper()}: {msg['content']}\n"
        # prompt += "ASSISTANT:"

        prompt = f"{data.query}"

        
        # 8️⃣ llama.cpp REST API çağrısı
        url = "http://192.168.3.32:8081/completion"
        payload = {
            "prompt": prompt,
            "n_predict": 128,  # Küçük model için yeterli
            "temperature": 0.7,
            "stop": ["\n"],  # Tek satır output
            "stream": False
        }

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Llama.cpp HTTP Hatası: {response.status_code} - {response.text}")

        result = response.json()
        return result
        assistant_text = result.get("content", "").strip()

        # 9️⃣ Yanıtı DB'ye kaydet
        save_conversation(data.user_id, "assistant", assistant_text)

        return {"answer": assistant_text}

    except Exception as e:
        print("askllama_cpp hata:", e)
        raise HTTPException(status_code=500, detail=str(e))


def query_llamacpp(messages):
    url = "http://127.0.0.1:8081/completion"  # llama.cpp server endpoint
    
    # Mesajları Llama'ya uygun formatta birleştiriyoruz
    prompt = ""
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"].strip()
        prompt += f"{role}: {content}\n"
    prompt += "ASSISTANT:"

    payload = {
        "prompt": prompt,
        "n_predict": 256,           # 512 büyükse bazen timeout olabilir, 256 daha güvenli
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["USER:", "SYSTEM:", "ASSISTANT:"],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"llama.cpp API hatası: {e}")

    # Bazı sürümler 'content' değil 'content'/'response'/'text' anahtarı döndürebiliyor:
    return data.get("content") or data.get("response") or data.get("text") or ""

## servis olarak başlatma    llama-cli.exe --model gpt2-turkish.gguf --server --port 8080
##llama-cli.exe --model C:\huggingface\generated\gpt2-turkish\gpt2-turkish.gguf --server 0.0.0.0 --port 8080
## yada llama-cli.exe --model gpt2-turkish.gguf --server --port 8080 --n-predict 200 --temperature 0.7 --top-p 0.9
##Doğrusu
##C:\llama_cpp\llama-server.exe --model "C:\huggingface\generated\gpt2-turkish\gpt2-turkish.gguf" --port 8081 --host 0.0.0.0
#Chat Şeklinde
#C:\llama_cpp>C:\llama_cpp\llama-server.exe --model "C:\huggingface\generated\gpt2-turkish\gpt2-turkish.gguf" --port 8081 --host 0.0.0.0 --chat


## tetiklemesi response = query_llamacpp(messages)
