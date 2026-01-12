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