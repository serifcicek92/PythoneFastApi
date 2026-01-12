import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================
# 1️⃣ Model ve tokenizer tanımı
# ================================
base_model = "vngrs-ai/Kumru-2B"

model_cache_dir = os.path.join(os.getcwd(), "model_cache")

print("⏳ Tokenizer yükleniyor / indiriliyor...")
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=model_cache_dir)

print("⏳ Model yükleniyor / indiriliyor...")
model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=model_cache_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ {base_model} modeli yüklendi ({device})")

# ================================
# 2️⃣ Basit prompt testi
# ================================
def generate_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # token_type_ids varsa kaldır
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_prompt = "Merhaba! Kendini tanıtır mısın?"
answer = generate_text(test_prompt)
print(f"\n💬 Test prompt: {test_prompt}")
print(f"🤖 Model cevabı: {answer}")

# ================================
# 3️⃣ Sohbet döngüsü
# ================================
print("\n💬 Kumru-2B ile sohbet! (çıkmak için 'q' yaz)")
while True:
    user_input = input("\n🧠 Sen: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 Görüşürüz!")
        break
    response = generate_text(user_input, max_new_tokens=256)
    print(f"🤖 Model: {response}")
