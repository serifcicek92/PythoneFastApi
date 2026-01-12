import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# -------------------
# 1️⃣ Ayarlar
# -------------------
base_model = "redrussianarmy/gpt2-turkish-cased"
dataset_path = "kural.jsonl"
output_dir = r"C:\huggingface\gpt2-turkish-cased-finetuned"
epochs = 3
batch_size = 1
lr = 1e-4
num_workers = 0  # Windows stabilitesi

print("\n--- Ayarlar ---")
print(f"Model: {base_model}")
print(f"Dataset: {dataset_path}")
print(f"Çıkış klasörü: {output_dir}")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {lr}")
print(f"Num workers: {num_workers}")
print("----------------\n")

# -------------------
# 2️⃣ Model ve Tokenizer
# -------------------
print("📥 Model ve tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# 🔧 GPT2 modelleri pad_token içermez → ekliyoruz:
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Device: {device}")

# -------------------
# 3️⃣ Dataset yükleme ve tokenize
# -------------------
print("📂 Dataset yükleniyor...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize_fn(example):
    text = example["prompt"] + " " + example["completion"]
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_fn)

# -------------------
# 4️⃣ TrainingArguments ve Trainer
# -------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    logging_steps=2,
    save_strategy="epoch",
    report_to="none",
    gradient_accumulation_steps=2,  # küçük batch için GPU doluluğu artırır
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -------------------
# 5️⃣ Eğitim
# -------------------
print("🚀 Eğitim başlıyor...")
trainer.train()

# -------------------
# 6️⃣ Kaydet
# -------------------
print("💾 Model kaydediliyor...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ Fine-tune tamamlandı! Model '{output_dir}' klasörüne kaydedildi.")
