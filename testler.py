##############################################################################
#train_mistral7b
#eğitmek için gpu ve ram yetmedi 
##############################################################################
##############################################################################
##############################################################################
#C:\huggingface\generated\mistral7b>python d:\Projeler\PythoneFastApi\train_mistral7b.py
#pip install torch accelerate transformers peft datasets
#ollama create mistral-7b-finetuned --model-path ./mistral-7b-finetuned

#pip install torch accelerate transformers peft datasets
#ollama create mistral-7b-finetuned --model-path ./mistral-7b-finetuned
############################################
# -*- coding: utf-8 -*-
"""
VRAM dostu Mistral 7B + LoRA fine-tune
RTX 2060 (6GB) için optimize edildi
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# -------------------
# 1️⃣ Ayarlar
# -------------------
base_model = "mistralai/Mistral-7B-v0.1"
dataset_path = r"D:\Projeler\PythoneFastApi\kural.jsonl"
output_dir = r"C:\huggingface\mistral-7b-finetuned"
epochs = 3
per_device_batch = 1
grad_accum = 8       # efektif batch 8
lr = 3e-4
max_length = 512

gpu_available = torch.cuda.is_available()
print("GPU mevcut mu?", gpu_available)
if gpu_available:
    print("GPU adı:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")

# -------------------
# 2️⃣ Tokenizer ve Model
# -------------------
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# CPU offload ile yükleme
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",        # HuggingFace Accelerate ile CPU+GPU dengeli
    torch_dtype=torch.float32,  # FP16 kapalı (RTX 2060 için)
    low_cpu_mem_usage=True
)

# -------------------
# 2️⃣a LoRA Config
# -------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # sadece dikkat katmanları
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# -------------------
# 3️⃣ Dataset ve Tokenize
# -------------------
dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize_fn(example):
    text = f"{example['prompt']}\n{example['completion']}"
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

# -------------------
# 4️⃣ Data Collator
# -------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# -------------------
# 5️⃣ TrainingArguments
# -------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_batch,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=epochs,
    learning_rate=lr,
    fp16=False,                     # FP16 kapalı
    logging_strategy="steps",
    logging_steps=5,                # anlık ilerlemeyi görmek için
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",           # 8-bit optimizer kaldırıldı
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -------------------
# 6️⃣ Eğitim
# -------------------
print("🚀 Eğitim başlıyor...")
trainer.train()

# -------------------
# 7️⃣ Kaydet
# -------------------
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✅ Fine-tune tamamlandı! Model '{output_dir}' klasörüne kaydedildi.")





##############################################################################
##############################################################################
##############################################################################
#chat test 

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "C:/huggingface/flan-t5-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

while True:
    prompt = input("\n🧠 Soru: ")
    if prompt.lower() in ["q", "quit", "exit"]:
        break
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"💬 Cevap: {response}")

##############################################################################
##############################################################################
##############################################################################
#flan-t5-base

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

# ===== GPU Bilgisi =====
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    print(f"✅ GPU aktif: {gpu_name}\nVRAM (GB): {vram_gb}")
else:
    print("❌ GPU bulunamadı, CPU kullanılacak.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Model ve Dataset Ayarları =====
base_model = "google/flan-t5-base"
dataset_path = r"D:\Projeler\PythoneFastApi\kural.jsonl"
output_dir = r"C:\huggingface\flan-t5-finetuned"

print("\n📥 Model yükleniyor...")
model = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("📂 Dataset yükleniyor...")
dataset = load_dataset("json", data_files={"train": dataset_path})

# ===== Tokenization Fonksiyonu =====
def tokenize(examples):
    inputs = examples["prompt"]
    outputs = examples["completion"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(outputs, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("✏️ Dataset tokenize ediliyor...")
tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# ===== Eğitim Ayarları =====
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=False,#torch.cuda.is_available(),
    bf16=True,
    logging_steps=1,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# ===== Eğitim =====
print("🚀 Eğitim başlıyor...")
trainer.train()

# ===== Kaydet =====
print("\n✅ Eğitim tamamlandı, model kaydediliyor...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"🎉 Model kaydedildi: {output_dir}")

##############################################################################
##############################################################################
##############################################################################
#google/flan-t5-base tetikleme kısmı promptest
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================================
# 1️⃣ Orijinal model ve tokenizer
# ================================
base_model = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ {base_model} modeli yüklendi ({device})")

# ================================
# 2️⃣ Sohbet fonksiyonu
# ================================
def chat(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ================================
# 3️⃣ Sohbet döngüsü
# ================================
print("\n💬 FLAN-T5 Base ile sohbet! (çıkmak için 'q' yaz)")
while True:
    user_input = input("\n🧠 Sen: ")
    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 Görüşürüz!")
        break
    response = chat(user_input)
    print(f"🤖 Model: {response}")

##############################################################################
##############################################################################
##############################################################################
#flan-t5-base part 2 DOĞRU ÇALIŞTI AMA İNGİLİZCEDEN DOLAYI SIKINTI
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# Ayarlar
base_model = "google/flan-t5-base"  # Küçük model, 6GB GPU için uygun
dataset_path = "D:/Projeler/PythoneFastApi/kural.jsonl"
output_dir = "C:/huggingface/flan-t5-finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ GPU aktif: {device}" if device=="cuda" else "✅ CPU kullanılacak")

# Model ve tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)

# Pad token ayarı
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Dataset yükleme
dataset = load_dataset("json", data_files=dataset_path)

# Tokenizasyon fonksiyonu
def tokenize(examples):
    inputs = examples["prompt"]
    targets = examples["completion"]
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=256, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=10,
    fp16=torch.cuda.is_available(),  # GPU varsa fp16 kullan
    save_total_limit=2,
    predict_with_generate=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"] if "train" in tokenized_dataset else tokenized_dataset,
    tokenizer=tokenizer
)

# Eğitim başlat
trainer.train()

# Modeli kaydet
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Eğitim tamamlandı, model kaydedildi: {output_dir}")


##############################################################################
##############################################################################
##############################################################################
#PROMT TEST 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "C:/huggingface/flan-t5-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

while True:
    prompt = input("\n🧠 Soru: ")
    if prompt.lower() in ["q", "quit", "exit"]:
        break
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"💬 Cevap: {response}")
##############################################################################
##############################################################################
##############################################################################
#SambaLingo-Turkish-Chat ram yetersiz kaldı eğitim için
    import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Ayarlar
base_model = "sambanovasystems/SambaLingo-Turkish-Chat"
dataset_path = "D:/Projeler/PythoneFastApi/kural.jsonl"
output_dir = "C:/huggingface/SambaLingo-Turkish-Chat-Finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ GPU aktif: {device}" if device=="cuda" else "✅ CPU kullanılacak")

# =============================
# Model ve tokenizer
# =============================
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model).to(device)

# Pad token ayarı (Causal LM için genellikle eos_token kullanılır)
tokenizer.pad_token = tokenizer.eos_token

# =============================
# Dataset yükleme
# =============================
dataset = load_dataset("json", data_files=dataset_path)

# =============================
# Tokenizasyon fonksiyonu
# =============================
def tokenize(examples):
    # Prompt ve completion ayrı tutuluyor
    tokenized_inputs = tokenizer(
        examples["prompt"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        examples["completion"],
        max_length=256,
        padding="max_length",
        truncation=True
    )["input_ids"]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)

# =============================
# Training args
# =============================
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=3e-5,
    logging_steps=5,
    save_steps=50,
    fp16=torch.cuda.is_available(),
    save_total_limit=3,
    prediction_loss_only=True
)

# =============================
# Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"] if "train" in tokenized_dataset else tokenized_dataset,
    tokenizer=tokenizer
)

# =============================
# Eğitim başlat
# =============================
trainer.train()

# Modeli kaydet
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Eğitim tamamlandı, model kaydedildi: {output_dir}")
#    pip install transformers peft accelerate bitsandbytes datasets

##############################################################################
##############################################################################
##############################################################################
#SambaLingo-Turkish-Chat sohbet 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "C:/huggingface/SambaLingo-Turkish-Chat-Finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model ve tokenizer yükleme
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

# Test loop
while True:
    prompt = input("\n🧠 Soru: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"💬 Cevap: {response}")
##############################################################################
##############################################################################
##############################################################################




