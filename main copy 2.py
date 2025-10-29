from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import logging
import ollama  # veya openai, kullandığın modele göre değiştir



# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------
# FastAPI
# --------------------------
app = FastAPI()

# --------------------------
# Modeller
# --------------------------
class Query(BaseModel):
    user_id: str
    question: str
    context: dict = None

# --------------------------
# Kullanıcı geçmişi
# --------------------------
user_history = {}  # basit örnek, prod için DB önerilir

# --------------------------
# Swagger API Bilgisi
# --------------------------
SWAGGER_BASE_URL = "https://your-api.com"
API_KEY = "YOUR_API_KEY"

# Swagger endpointlerini örnek JSON olarak
SWAGGER_ENDPOINTS = [
    {
        "name": "get_invoice",
        "path": "/invoice",
        "method": "GET",
        "params": ["ID", "MUSCARIHESAPID", "FATURATARIHI"]
    },
    {
        "name": "get_customer",
        "path": "/customer",
        "method": "GET",
        "params": ["CUSTOMER_ID"]
    }
]

# --------------------------
# Swagger veri çekme fonksiyonu
# --------------------------
def call_swagger(endpoint_name, param_values):
    endpoint = next((e for e in SWAGGER_ENDPOINTS if e["name"] == endpoint_name), None)
    if not endpoint:
        return {"error": "Endpoint bulunamadı"}
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {k: param_values.get(k) for k in endpoint["params"] if k in param_values}
    
    if endpoint["method"].upper() == "GET":
        resp = requests.get(SWAGGER_BASE_URL + endpoint["path"], headers=headers, params=params)
    else:
        resp = requests.post(SWAGGER_BASE_URL + endpoint["path"], headers=headers, json=params)
    
    return resp.json()

# --------------------------
# /ask endpoint
# --------------------------
@app.post("/ask")
def ask(q: Query):
    SWAGGER_URL = "http://192.168.3.101:6969/swagger/v1/swagger.json"
    response = requests.get(SWAGGER_URL, verify=False)  # Lokal sertifika için verify=False
    openapi_spec = response.json()
    try:
        # Kullanıcı geçmişini al
        history = user_history.get(q.user_id, [])
        history_text = "\n".join(history)

        # AI prompt
        prompt_text = f"""
Kullanıcı girişi: {q.question}
Kullanıcı geçmişi: {history_text}

Sen Boyut Bilgisayar şirketinin nervus programı için görev yapan bir kurumsal asistansın

Kurallar:
1. Gizli veya kişisel verileri paylaşma.
2. Her zaman güvenli ve şirket politikalarına uygun yanıt ver.
3. Çıktı daima json formatında olmalı
4. Herzaman türkçe cevap ver
5. Eğer kullanıcı sistemden bir veri istemiyorsa sen cevabını kullanıcının isteğine uygun  JSON formatında döndür:
{{
"resonse":"mesaj",
"system":"systemebilgi"
}}
6. Eğer kullanıcı bir sistemden veri istiyorsa, kullanıcının isteğine uygun endpoint'i OpenAPI tanımından seç.
JSON formatında döndür:
{{
  "endpoint": "<url>",
  "method": "GET",
  "params": {{ "key": "value" }}
}}
7. Swagger endpointleri:
{json.dumps(openapi_spec["paths"])}
"""
        

        # AI cevabı
        response = ollama.chat(
            #model="gpt-oss",
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": prompt_text}]
        )

        return response;
        answer_text = response["message"]["content"]
        logging.info(f"AI cevabı: {answer_text}")

        # JSON direktif algılama
        komut_match = re.search(r"####\s*(\{.*\})\s*####", answer_text, re.DOTALL)
        if komut_match:
            cmd_json_str = komut_match.group(1)
            cmd_data = json.loads(cmd_json_str)
            endpoint_name = cmd_data["directive"]
            params = cmd_data.get("parameters", {})

            # Swagger API çağır
            api_result = call_swagger(endpoint_name, params)
            user_history.setdefault(q.user_id, []).append(f"Direktif: {q.question}")
            return {"answer": api_result}

        # Normal cevap
        user_history.setdefault(q.user_id, []).append(q.question)
        return {"answer": answer_text}

    except Exception as e:
        logging.error(f"Hata: {e}")
        return {"answer": f"Hata: {str(e)}"}

#uvicorn main:app --reload
#ollama run gpt-oss
#ollama pull llama3.1:latest