from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging
import json

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
# Kullanıcı mesaj modeli
# --------------------------
class Query(BaseModel):
    user_id: str
    question: str

# --------------------------
# Claude API bilgileri
# --------------------------
CLAUDE_API_URL = "https://api.anthropic.com/v1/complete"
CLAUDE_API_KEY = "AAAAC3NzaC1lZDI1NTE5AAAAILg7ji6zfz2MwJRNqC0Hd050kBB5xamgr8RwzSu2N/Sx"

# --------------------------
# /ask endpoint
# --------------------------
@app.post("/ask")
def ask(q: Query):
    try:
        # Claude prompt
        prompt_text = f"""
You are a helpful assistant. Answer the user's question in JSON format.
User: {q.question}
JSON output format:
{{
  "response": "<your answer here>"
}}
"""

        headers = {
            "Authorization": f"Bearer {CLAUDE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-oss:120b-cloud",   # veya hesabına uygun Claude modeli
            "prompt": prompt_text,
            "max_tokens_to_sample": 500
        }

        response = requests.post(CLAUDE_API_URL, headers=headers, 
        json=payload)
        response.raise_for_status()
        data = response.json()

        # Claude yanıtı
        claude_text = data.get("completion", "")

        logging.info(f"Claude yanıtı: {claude_text}")

        # JSON parse et
        try:
            answer_json = json.loads(claude_text)
        except ValueError:
            answer_json = {"response": claude_text}

        return {"answer": answer_json}

    except Exception as e:
        logging.error(f"Hata: {e}")
        return {"answer": f"Hata: {str(e)}"}
#uvicorn main:app --reload
#ollama run gpt-oss
#ollama pull llama3.1:latest