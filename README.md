# Persian-To-English-Translator

![Banner](banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen3](https://img.shields.io/badge/Base%20Model-Qwen3--0.6B-blue)](https://huggingface.co/Qwen)
[![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth)
[![Dataset: 300k](https://img.shields.io/badge/Dataset-300k%20Pairs-red)]()

## 📖 Project Overview
**Persian-To-English-Translator** is a specialized, lightweight machine translation model designed to bridge the gap between Persian (Farsi) and English with high speed and accuracy. Built upon the **Qwen3-0.6B** architecture, this project aims to provide an efficient solution for low-resource environments without compromising on translation quality.

### 🚀 Key Goals
- **Low-Resource:** Optimized to run efficiently on consumer-grade hardware.
- **High Speed:** Fast inference suitable for real-time applications.
- **Accuracy:** Finetuned on high-quality, cleaned datasets to handle nuances of Persian-English translation.

---

## 🛠️ Technical Details

### Model Architecture
- **Base Model:** Qwen3-0.6B
- **Fine-Tuning Method:** PEFT LoRA (Low-Rank Adaptation)
- **Optimization Library:** [Unsloth](https://github.com/unslothai/unsloth) (Used for 2x faster training and optimized inference)

### Training Process
The model was trained using **Unsloth**, which significantly reduced VRAM usage and training time. Below is a snapshot of the training initialization:

```text
The model is already on multiple devices. Skipping the move to device specified in `args`.
==================================================
🚀 شروع آموزش...
==================================================
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 290,376 | Num Epochs = 2 | Total steps = 72,594
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
 "-____-"     Trainable parameters = 20,185,088 of 616,235,008 (3.28% trained)
```

### Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Epochs** | 2 |
| **Batch Size (per device)** | 2 |
| **Gradient Accumulation** | 4 |
| **Learning Rate** | 2e-4 |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **LoRA Dropout** | 0.05 |
| **Weight Decay** | 0.01 |
| **Optimizer** | AdamW (8-bit) |
| **Scheduler** | Linear |

---

## 📊 Dataset Information

The model was trained on a carefully curated dataset of **300,000 high-quality Persian-English sentence pairs**. The data underwent a rigorous cleaning process to remove noise, align sentences correctly, and address specific challenges in Persian-to-English translation (e.g., idioms, complex grammar).

### Dataset Comparison Table
We benchmarked our curated dataset against other common open sources:

| Dataset Name | Size (Pairs) | Noise Level | Alignment Quality | Suitability for MT |
| :--- | :--- | :--- | :--- | :--- |
| **Our Curated Dataset** | **300k** | **Very Low** | **High** | **Excellent** |
| Common Crawl Extract | 1M+ | High | Low | Poor |
| OPUS OpenSubtitles | 5M+ | Medium | Medium | Good (Conversational) |
| Mizan | 1M | Low | Medium | Good (Formal) |

---

## 📦 Deliverables & Usage

All model checkpoints and GGUF quantized versions are open-sourced for community use.

### Download Links
- **Hugging Face Hub:** [Link to Model]
- **GGUF (Quantized):** [Link to GGUF]

### How to Run (Inference)
You can load the model using standard Hugging Face `transformers` or `unsloth` for faster inference.

#### Using Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "YourUsername/Persian-To-English-Translator"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

input_text = "مدل‌های زبانی بزرگ دنیا را تغییر داده‌اند."
prompt = f"Translate the following Persian text to English:\n{input_text}\nTranslation:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Using Unsloth (Fast Inference)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "YourUsername/Persian-To-English-Translator",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# ... generation code ...
```

---

## 📄 License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software.

```text
MIT License

Copyright (c) 2024 Independent Developer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```
