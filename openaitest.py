from openai import OpenAI
OPENAI_KEY = KEYGELECEK

client = OpenAI(api_key=OPENAI_KEY)

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{"role":"user","content":"Merhaba nasılsın"}]
)

assistant_text = response.output_text

print(f"cevap:{assistant_text}")