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

==================================================================================
ollama alternatif
KoboldCPP ve llama.cpp 

https://huggingface.co/ modeller buradan indirilecek


git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release
cd ..

modeli çalıştırmak için
./main -m ./gpt-oss-20b.Q3_K_M.gguf -p "Merhaba, nasılsın?"

⚙️ 2️⃣ Kurulum
pip install llama-cpp-python --upgrade


CUDA (GPU hızlandırma) kullanmak istiyorsan:

pip install llama-cpp-python[cuda]


Eğer Windows’taysan, Visual Studio C++ Redistributable yüklü olmalı.

🧠 3️⃣ Basit örnek (tek prompt)
from llama_cpp import Llama

llm = Llama(
    model_path="gpt-oss-20b.Q3_K_M.gguf",
    n_gpu_layers=50,       # GPU'da kaç katman çalışacak
    n_ctx=2048,            # context uzunluğu
    n_threads=6,           # CPU thread sayısı
    temperature=0.7,       # rastgelelik
    top_p=0.9              # nucleus sampling
)

response = llm("Merhaba, nasılsın?", max_tokens=200)
print(response["choices"][0]["text"])

💬 4️⃣ Chat tarzı (system / user / assistant) mesajları
from llama_cpp import Llama

llm = Llama(model_path="gpt-oss-20b.Q3_K_M.gguf", n_gpu_layers=50)

messages = [
    {"role": "system", "content": "Sen nazik ve kısa yanıtlar veren bir asistansın."},
    {"role": "user", "content": "Bana uzay hakkında kısa bir bilgi ver."}
]

response = llm.create_chat_completion(
    messages=messages,
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop=["</s>"]  # İsteğe bağlı durdurucu
)

print(response["choices"][0]["message"]["content"])


💡 Bu format, OpenAI Chat API ile birebir aynıdır.
Yani eğer daha önce openai.ChatCompletion.create() kullandıysan, aynı şekilde çalışır.

⚙️ 5️⃣ Desteklenen parametreler
Parametre	Açıklama
model_path	GGUF model dosya yolu
n_ctx	Context uzunluğu (örnek: 2048, 4096)
n_threads	CPU thread sayısı
n_gpu_layers	GPU'da kaç katman yüklenecek (örnek: 50–70 RTX2060 için)
temperature	Rastgelelik (0.0 → deterministik, 1.0 → yaratıcı)
top_p	Nucleus sampling
top_k	K olasılık filtresi
repeat_penalty	Tekrar eden kelimeleri cezalandırma katsayısı
max_tokens	Üretilen maksimum token sayısı
stop	Durdurma dizileri listesi
stream	True ise çıktıyı token token döner (canlı akış)
🔄 6️⃣ Stream (canlı çıktı) örneği
for token in llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Kısa ve samimi cevaplar ver."},
        {"role": "user", "content": "Python neden popüler?"}
    ],
    temperature=0.7,
    stream=True
):
    print(token["choices"][0]["delta"].get("content", ""), end="", flush=True)


Canlı akışlı çıktı verir (terminalde kelime kelime yazar).

💡 7️⃣ Kullanışlı ayarlar (RTX 2060 için önerilen)
llm = Llama(
    model_path="gpt-oss-20b.Q3_K_M.gguf",
    n_gpu_layers=50,    # RTX 2060: genelde 50-60 arası optimum
    n_threads=6,
    n_ctx=2048,
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)

🧩 8️⃣ JSON biçiminde çıktı almak istersen
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Her zaman JSON formatında yanıt ver."},
        {"role": "user", "content": "Kedi hakkında kısa bilgi ver."}
    ],
    temperature=0.2
)

print(response["choices"][0]["message"]["content"])

🔚 Özet

✅ llama.cpp Python API → OpenAI Chat API’ye çok benzer
✅ System / User / Assistant destekli
✅ GPU hızlandırmalı (CUDA, Metal, Vulkan)
✅ Stream, JSON, sıcaklık, top_p, top_k, stop gibi tüm parametreler kullanılabilir


=========================================================
=========================================================



git büyük dosyalar indirirken hata verirse
cd C:\llama_cpp\models\gpt-oss-20b\gpt-oss-20b-GGUF
git lfs pull
git lfs fetch --all
git lfs checkout


bu mesaj sorar
llama-cli.exe -m models\Qwen3-14B-Q4_K_M.gguf -p "Merhaba nasılsın?"    

buda server olarak başlatır
llama-server.exe -m models\Qwen3-14B-Q4_K_M.gguf --host 127.0.0.1 --port 8080


pip install llama-cpp-python --upgrade
pip uninstall llama-cpp-python -y
pip install llama-cpp-python[cuda] --upgrade   --cuda destekli
pip install llama-cpp-python --force-reinstall --upgrade --extra-index-url https://jllllll.github.io/llama-cpp-python-cu124

-------------------
pip uninstall llama-cpp-python -y   

python -m pip install llama-cpp-python==0.1.62 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX/cu121

pip show llama-cpp-python   
----------------------


from llama_cpp import Llama

llm = Llama(
      model_path="C:\\llama_cpp\\models\\Qwen3-14B.Q3_K_M.gguf",
      n_gpu_layers=20,   # GPU’ya kaç katman yüklenecek
      n_ctx=4096,        # maksimum context (token penceresi)
      n_threads=6,       # işlemci çekirdek sayısı
      verbose=True       # log detaylarını göster
  )
# response = llm(
#           prompt="Merhaba! Nasılsın?",
#           max_tokens=200,
#           temperature=0.7,
#           top_p=0.9,
#           stop=["User:", "Assistant:"]
#       )


response = llm.create_chat_completion(
    messages=messages,
    max_tokens=200,
    temperature=0.7,
    top_p=0.9
)
assistant_text = response["choices"][0]["message"]["content"]
⚙️ 3️⃣ Önemli parametreler
Parametre	Açıklama
model_path	GGUF dosyasının yolu
n_ctx	Maksimum token sayısı (örneğin 4096)
n_gpu_layers	GPU'ya taşınacak katman sayısı
temperature	Rastgelelik derecesi (0.7 önerilir)
top_p	Nucleus sampling oranı
max_tokens	Yanıtın maksimum uzunluğu
repeat_penalty	Tekrar eden kelimeleri bastırır


=========================================================
=========================================================
ALTER USER postgres WITH PASSWORD 'yeni_parola';


POSTGRE SQL YEDEK ALMA VE VERSİYOM DÜŞÜRME
backup alma
pg_dumpall.exe -U postgres -f "d:\PostgreSQL\full_backup02112025.sql"


=========================================================
=========================================================
PGVECTOR Kurulum

Windows
Ensure C++ support in Visual Studio is installed and run x64 Native Tools Command Prompt for VS [version] as administrator. Then use nmake to build:

set "PGROOT=C:\Program Files\PostgreSQL\18"
cd %TEMP%
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install

CREATE EXTENSION vector;

Create a vector column with 3 dimensions

CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
Insert vectors

INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
Get the nearest neighbors by L2 distance

SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;



CREATE TABLE api_endpoints (
    id SERIAL PRIMARY KEY,
    module TEXT,
    service TEXT,
    method TEXT,
    endpoint TEXT,
    description TEXT,
    parameters TEXT[],
    embedding VECTOR(384),
    elementtypeid BIGINT,
    menuid BIGINT
);

CREATE TABLE business_rules (
    id SERIAL PRIMARY KEY,
    rule_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    embedding VECTOR(384)
);


ALTER TABLE api_endpoints
ADD CONSTRAINT unique_endpoint UNIQUE (endpoint);


2️⃣ Distance operatörü doğru mu?

pgvector’da <-> → Euclidean distance

<#> → Cosine distance

<=> → Inner product

Eğer semantic search yapıyorsan genelde cosine distance (<#>) daha tutarlı sonuç verir.
=========================================================
=========================================================