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

#netsh advfirewall firewall add rule name="Uvicorn 8000" dir=in action=allow protocol=TCP localport=8000


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




system_content = f"""
Sen Boyut Bilgisayar şirketinin Nervus programı için görev yapan bir kurumsal asistanssın. 
Çıktı daima **Türkçe** olmalı ve sadece tek bir **JSON** nesnesi döndürmelisin. 
Cevap üretirken SADECE aşağıdaki [KONTEKS BİLGİLERİ] bölümlerindeki en uygun bilgiye dayan. Başka bir bilgi kullanma.

[JSON ÇIKTI ZORUNLU KURALLARI]
1.  Gizli veya kişisel verileri paylaşma.
2.  Kurallar dışında bir mesaj gelirse yetkin olmadığını belirt.
3.  Eğer kullanıcının sorusu bir API ucuyla (endpoint) cevaplanabilecek **net bir veri sorgusu** ise:
    a. **bilgiiste** değerini **true** yap.
    b. **endpoint** alanına, [API ENDPOINT KONTEKS] kısmından **sadece en alakalı ve tek bir endpoint'in path'ini** (Örn: /api/siparis/giris) string olarak yaz. 
    c. Endpointleri benim yazdığım gibi kabul et değiştirme.
4.  Eğer soru sadece bir süreç veya kural bilgisi gerektiriyorsa yani endpoint gerektirmiyorsa:
    a. **bilgiiste** değerini **false** yap.
    b. **endpoint** alanını **boş string ("")** veya **null** yap.
    c. İş süreci bilgi tabanı referansından yararlanırken, **verilen cevabın yalnızca kullanıcının SADECE son sorusuna en uygun kayıttan oluşmasını sağla**. Eğer çekilen bir kayıt (Kayıt X) birden fazla farklı konuyu içeriyorsa, **sadece alakalı olan cümleyi seç ve diğer cümleleri yoksay.**
JSON FORMATI:
{{"response":"Yanıtın mesaj kısmı.","system_debug":"Bulunan konteks bilgiyi özetle (İngilizce).","bilgiiste":{bool(endpoint_matches)},"endpoint":"{[m[3] for m in endpoint_matches]}"}}

[KONTEKS BİLGİLERİ - YALNIZCA BURAYI KULLAN]

--- API ENDPOINT KONTEKS REFERANSI ---
{endpoint_context}
--- İŞ SÜRECİ/BİLGİ TABANI REFERANSI ---
{rule_context}
"""
        system_message = {"role": "system", "content": system_content}

gemma3:27b	17GB
qwen3:14b	19GB
qwen3:8b	5.2gb
llama3.1:8b-instruct-q5_1	6.1gb
llama3.1:8b-instruct-q6_K	6.6gb
mistral:text	4.1GB
mistral:7b	4.4GB
phi3:14b	7.9GB  Microsoftun++++
llava:13b	8GB
dolphin3:8b	4.9GB
olmo2:13b	8.4GB
codellama:13b	7.4 gb tartışmak için ideal
mistral-nemo:12b	7.1GB
mistral-small:latest	14GB
falcon3:latest	4.6gb 10milyon parametre
falcon3:10b	6.3GB
aya:8b		4.8GB
aya:35b-23-q2_K	14GB

