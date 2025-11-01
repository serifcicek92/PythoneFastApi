"""
#uvicorn main:app --reload
#ollama run gpt-oss:latest
#ollama pull llama3.1:latest

#python -m uvicorn main:app --reload
#python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

#netsh advfirewall firewall add rule name="Uvicorn 8000" dir=in action=allow protocol=TCP localport=8000


#pip install fastapi[all]
#pip install uvicorn
#pip install sentence-transformers
#pip install psycopg2
#pip install ollama

#venv\Scripts\activate

#wmic cpu get NumberOfLogicalProcessors
#set OLLAMA_NUM_THREADS=all
#ollama run llama3.1 --num-thread 12

{
  "module": "Ilac",
  "service": "IlcFaturalarServis",
  "method": "GetDataBVAsync",
  "endpoint": "/Ilac/IlcFaturalarServis/GetDataBVAsync",
  "description": "Faturaların listesini getirir. Parametreler: Id, FaturaTarihiBas (zorunlu), FaturaTarihiBit, FaturaNo, CepSubeStr.",
  "parameters": [
    "Id",
    "FaturaTarihiBas (zorunlu)",
    "FaturaTarihiBit",
    "FaturaNo",
    "CepSubeStr"
  ]
}

CREATE EXTENSION IF NOT EXISTS cube;

CREATE TABLE api_endpoints (
    id SERIAL PRIMARY KEY,
    module TEXT,
    service TEXT,
    method TEXT,
    endpoint TEXT,
    description TEXT,
    parameters TEXT[],
    embedding cube
);

CREATE TABLE business_rules (
    id SERIAL PRIMARY KEY,
    rule_text TEXT NOT NULL,         -- Kullanıcının "bilgi ekle" ile girdiği metin
    created_at TIMESTAMP DEFAULT NOW(), -- Kayıt tarihi (opsiyonel)
    embedding cube                  -- 100 boyutlu vektör
);

CREATE TABLE conversation_history
(
    id SERIAL PRIMARY KEY,
    user_id character varying(64) COLLATE pg_catalog."default",
    message_role character varying(10) COLLATE pg_catalog."default",
    message_text text COLLATE pg_catalog."default",
    created_at timestamp without time zone DEFAULT now()
);



🔹 Rol Tanımı
Rol	Kullanım
system	Modelin davranışını belirler: kurallar, gizlilik, JSON çıktısı zorunluluğu, Türkçe yanıt vb.
user	Kullanıcının gönderdiği mesajlar (soru, sohbet, merhaba vs.)
assistant	Modelin verdiği yanıtlar. Bu rol, geçmişi DB’de saklayacak ve gelecekteki cevaplar için referans olacak


{
  "module": "Ilac",
  "service": "IlcFaturalarServis",
  "method": "GetDataBVAsync",
  "endpoint": "/Ilac/IlcFaturalarServis/GetDataBVAsync",
  "description": "Faturaların listesini getirir",
  "parameters": [
    "Id",
    "FaturaTarihiBasZorunlu",
    "FaturaTarihiBit",
    "FaturaNo",
    "CepSubeStr"
  ]
}












[
  {
    "role": "system",
    "content": "Sen sadece mizah yapan ve esprili bir asistansın."
  },
  {
    "role": "user",
    "content": "Bugün hava durumu nasıl?"
  },
  {
    "role": "assistant",
    "content": "Dışarısı o kadar bulutlu ki, gökyüzü ciddiyete boğulmuş. Güneş bile gülümsemekten yorulmuş!"
  },
  {
    "role": "user",
    "content": "Peki yarın ne beklemeliyim?"
  }
]






"""