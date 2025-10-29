from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import psycopg2
import json
import re
import logging

# --------------------------
# Logging ayarları
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------
# FastAPI & Model
# --------------------------
app = FastAPI()

class Query(BaseModel):
    question: str
    context: dict = None

# --------------------------
# PostgreSQL bağlantı bilgileri
# --------------------------
DB_HOST = "localhost"
DB_NAME = "ollamadb"
DB_USER = "postgres"
DB_PASSWORD = "sql123"
DB_PORT = 5432

# --------------------------
# Direktif kaydetme fonksiyonu
# --------------------------
def save_direktif(komut, aciklama, parameters, olusturan="yapayzeka"):
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO direktifler (komut, aciklama, filtre_parametreleri, olusturan) VALUES (%s, %s, %s, %s)",
            (komut, aciklama, json.dumps(parameters), olusturan)
        )
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f"Direktif kaydedildi: {komut}")
    except Exception as e:
        logging.error(f"DB Hatası: {e}")

# --------------------------
# Direktifleri DB'den çekme
# --------------------------
def get_direktifler():
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT komut, aciklama, filtre_parametreleri FROM direktifler")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        direktif_list = []
        for r in rows:
            direktif_list.append({
                "komut": r[0],
                "aciklama": r[1],
                "parameters": json.loads(r[2]) if r[2] else {}
            })
        return direktif_list
    except Exception as e:
        logging.error(f"DB Hatası get drektifler: {e}")
        return []
# --------------------------
#direktif varmı
# --------------------------
def direktif_var_mi(komut_name):
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM direktifler WHERE komut = %s LIMIT 1",
            (komut_name,)
        )
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result is not None  # True ise direktif var, False ise yok
    except Exception as e:
        logging.error(f"DB kontrol hatası direktif_var_mi: {e}")
        return False
# --------------------------
# direktif getir
# --------------------------
def direktif_getir(komut_name):
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT id, komut, aciklama, filtre_parametreleri FROM direktifler WHERE lower(komut) = lower(%s) LIMIT 1",
            (komut_name.strip(),)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            # JSON parse yapmadan direkt string olarak döndür
            return {
                "komut": row[1],
                "aciklama": row[2],
                "parameters": row[3]  # Burada artık string döner
            }
        return None
    except Exception as e:
        logging.error(f"DB kontrol hatası direktif_getir: {e}")
        return None
# --------------------------
# /ask endpoint
# --------------------------
@app.post("/ask")
def ask(q: Query):
    #logging.info(f"Kullanıcı sorusu geldi: {q.question}")
    try:
        # Mevcut direktifleri al
        direktifler = get_direktifler()
        direktif_prompt = "\n".join(
            [f"{d['komut']}: {d['aciklama']}, filtre: {json.dumps(d['parameters'])}" for d in direktifler]
        )

        # Yapay zekaya gönderilecek prompt
        prompt_text = f"""
Sen bir sistem yönetimi asistanısın. Kullanıcı sana bazen direktif (komut tanımı) oluşturmanı isteyebilir, bazen sadece soru sorabilir.

Kullanıcının girişi:
{q.question}

Context verileri: {json.dumps(q.context, ensure_ascii=False)}

Kurallar:
1. Eğer kullanıcı sadece bilgi almak, sohbet etmek veya soru sormak istiyorsa — yani direktif oluşturman gerektiğini açıkça söylemiyorsa — JSON veya #### içeren hiçbir şey döndürme. Normal Türkçe yanıt ver. ve sadece kullanıcının girişine yanıt ver. Ayrıca kullanıcı id 1 yada admin gelmiyors direktif ekleme işlemi olmayacak ve sen direktifle ilgili fikirlerini söylemeyeceksin. neden bu cevabı verdiğinide yazma yani sisteme yönelik değil kullanıcıya yönelik cevap ver
2. Eğer kullanıcı "direktif oluştur", "komut tanımla", "yeni direktif", "şu parametrelerle bir komut oluştur" gibi ifadeler kullanıyorsa — o zaman yeni bir direktif oluştur. Sadece aşağıdaki formatta dön:
#### 
{{ 
    "directive": "<direktifadı>",
    "aciklama": "<açıklama>",
    "parameters": {{ "parametre1": null, "parametre2": null }},
    "olusturan": "yapayzeka"
}}
####
3. Yukarıdaki kurala uymadan JSON veya #### içeren cevap üretme.
4. JSON dışında başka metin yazma, yorum ekleme.
5. Lütfen tüm cevapların Türkçe olsun.

Önceden tanımlı direktifler:
{direktif_prompt}
"""

        #print(f"Soru:{prompt_text}")
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt_text}]
        )

        answer_text = response["message"]["content"]
        logging.info(f"Yapay zekadan cevap: {answer_text}")
        print("-------------------------------")
        #print(f"cevap: {answer_text}")
        
        #print(f"Burada Bakacak cevap :{answer_text}")
        print("-------------------------------")
        
        #komut_match = re.search(r"####\s*(\{.*?\})", answer_text, re.DOTALL)    
        komut_match = re.search(r"####\s*(\{.*\})", answer_text, re.DOTALL)
        if komut_match:  # <-- önce kontrol:
            cmd_json_str = komut_match.group(1)
            cmd_json_str = cmd_json_str.replace("####", "")
            print(f"{cmd_json_str}")
            print("-------------------------------")

        
            if komut_match:
            
                try:
                    cmd_data = json.loads(cmd_json_str)
                    komut_name = cmd_data["directive"]
                    direktif = direktif_getir(komut_name.strip())
                    print(f"ife girdi.{komut_name.strip()}")
                    print("--------------------------------")
                    if direktif:
                        return {"answer":direktif}
                    else:
                        aciklama = cmd_data["aciklama"]
                        parameters = cmd_data.get("parameters", [])
                        print(f"komut_name:{komut_name},aciklama:{aciklama},parameters{parameters}")

                        save_direktif(
                            komut=komut_name,
                            aciklama=aciklama,
                            parameters=parameters,
                            olusturan="yapayzeka"
                        )
                        return {"answer":"DIRECTIVEADD"}     
                    
                except Exception as e:
                    logging.error(f"Komut parse veya kaydetme hatası: {e}")
            else:
                print("else girdi malesef")
            
        return {"answer": answer_text}

    except Exception as e:
        logging.error(f"Hata oluştu: {e}")
        return {"answer": f"Hata: {str(e)}"}

# --------------------------
# /dbtest endpoint
# --------------------------
@app.get("/dbtest")
def db_test():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return {"status": "ok", "db_response": result[0]}
    except Exception as e:
        logging.error(f"DB test hatası: {e}")
        return {"status": "error", "message": str(e)}




'''
try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": q.question}]
        )
        print(response)  # Bu satır response'u PowerShell'de göreceğiz
        return {"answer": response}  # Önce tüm cevabı JSON olarak dön
    except Exception as e:
        return {"answer": f"Hata: {str(e)}"}
'''



'''
http://127.0.0.1:8000/ask
{   
  "user_id": "3",
  "question": "nekadar iyi bir yapay zekasın sen öyle seni şirin şey seni",
  "context": {}
}
'''