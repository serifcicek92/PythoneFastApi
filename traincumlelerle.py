#pip install transformers datasets accelerate torch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python traincumlelerle.py --mode train
python traincumlelerle.py --mode chat
"""
import argparse, os, torch, textwrap
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from torch.utils.data import Dataset

MODEL_CKPT  = "openai-community/gpt2"
SAVE_DIR    = "otel-clm-final"
DATA_FILE   = "denemedata.txt"
BLOCK_SIZE  = 512
MAX_NEW_TOK = 60

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "chat"], required=True)
    p.add_argument("--data", default=DATA_FILE)
    p.add_argument("--model_out", default=SAVE_DIR)
    return p.parse_args()

class CLMDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

def train(args):
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"{args.data} bulunamadı.")
    print("Model yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    # 1) BOS & EOS ekle
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<|startoftext|>'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_CKPT)
    model.resize_token_embeddings(len(tokenizer))

    print("Metin tokenize ediliyor...")
    with open(args.data, encoding="utf-8") as f:
        text = f.read().replace("\n", " ")
    # 2) Her satırı BOS + text + EOS yap
    lines = [l.strip() for l in text.split(".") if l.strip()]
    all_ids = []
    for line in lines:
        ids = tokenizer.build_inputs_with_special_tokens(
            tokenizer.encode(line, add_special_tokens=False)
        )
        all_ids.extend(ids)

    # 3) Block'la (512)
    block = BLOCK_SIZE
    ds_ids, ds_mask = [], []
    for i in range(0, len(all_ids) - block + 1, block):
        chunk = all_ids[i:i + block]
        ds_ids.append(torch.tensor(chunk))
        ds_mask.append(torch.ones(len(chunk)))
    encodings = {"input_ids": ds_ids, "attention_mask": ds_mask}
    train_ds = CLMDataset(encodings)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.model_out,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="epoch",
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
    )
    print("Eğitim başlıyor...")
    trainer.train()
    trainer.save_model(args.model_out)
    tokenizer.save_pretrained(args.model_out)
    print(f"Model  {args.model_out}  altına kaydedildi.")

# --------------------------------------------------  
# 2) CHAT (soruya göre üretim)
def chat(args):
    if not os.path.isdir(args.model_out):
        raise FileNotFoundError(f"Önce --mode train ile modeli oluşturun: {args.model_out}")
    print("Model yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_out)
    model     = AutoModelForCausalLM.from_pretrained(args.model_out)
    tokenizer.pad_token = tokenizer.eos_token

    max_len = getattr(model.config, 'max_position_embeddings', 1024)
    reserve = 80
    max_doc = max_len - reserve

    # 1) Dokümanı TOKEN olarak kes
    with open(args.data, encoding="utf-8") as f:
        doc = f.read().strip()
    doc_ids  = tokenizer(doc, truncation=True, max_length=max_doc).input_ids
    prefix_ids = tokenizer("Aşağıdaki belgelere sadık kalın.\n\n", add_special_tokens=False).input_ids
    suffix_ids = tokenizer("\n\nSoru: {question}\nYanıt:", add_special_tokens=False).input_ids
    base_ids   = prefix_ids + doc_ids + suffix_ids[:-1]  # {question} yerini tut

    print("\n=== Sohbet başladı (çıkmak için q) ===")
    while True:
        soru = input("\nSorunuz: ").strip()
        if soru.lower() == "q":
            break
        soru_ids   = tokenizer(soru, add_special_tokens=False).input_ids
        prompt_ids = base_ids + soru_ids + [suffix_ids[-1]]  # Soru EKLENDİ
        if len(prompt_ids) >= max_len:
            prompt_ids = prompt_ids[:max_len - 1] + [tokenizer.eos_token_id]

        input_ids = torch.tensor([prompt_ids])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(60, max_len - input_ids.shape[1]),
                do_sample=True,               # Sampling → çeşitli cevap
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        cevap = tokenizer.decode(out_ids[0][len(prompt_ids):], skip_special_tokens=True).strip()
        print("Cevap:", cevap)

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        chat(args)


"""
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

model_name = "t3ai/gpt2-turkish"   # veya "Trendyol/llama-3b-turkish"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name)

# 1) 100 anlatımı birleştir
docs = open("denemedata.txt", encoding="utf-8").read()
docs = docs.replace("\n", " ")  # tek satır

# 2) Tokenize
tokens = tokenizer(docs, return_tensors="pt", truncation=False)
block_size = 512
# 3) Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

# chunk'la
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = group_texts({k: [v.tolist()] for k, v in tokens.items()})
train_ds = Dataset(lm_dataset)

# 4) Trainer
args = TrainingArguments(
    output_dir="llama-hotel-clm",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
trainer.save_model("llama-hotel-clm-final")
"""