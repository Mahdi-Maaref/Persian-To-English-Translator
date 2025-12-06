<div align="center">
 
 # 🌐 Persian-To-English-Translator
![Banner](banner.png)
### A Lightweight, Fast, and Accurate Neural Machine Translation Model


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/⚡-Unsloth-purple)](https://github.com/unslothai/unsloth)
[![Model: Qwen3](https://img.shields.io/badge/Base%20Model-Qwen3--0.6B-blue)](https://huggingface.co/Qwen)
[![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth)
[![Dataset: 300k](https://img.shields.io/badge/Dataset-300k%20Pairs-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

**A fine-tuned Persian to English translation model based on Qwen3-0.6B, optimized for low-resource environments while maintaining high translation quality.**

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Training](#-training-details) •
[Dataset](#-dataset) •
[License](#-license)


## 📖 Project Overview
**Persian-To-English-Translator** is a specialized, lightweight machine translation model designed to bridge the gap between Persian (Farsi) and English with high speed and accuracy. Built upon the **Qwen3-0.6B** architecture, this project aims to provide an efficient solution for low-resource environments without compromising on translation quality.

### 🚀 Key Goals
- **Low-Resource:** Optimized to run efficiently on consumer-grade hardware.
- **High Speed:** Fast inference suitable for real-time applications.
- **Accuracy:** Finetuned on high-quality, cleaned datasets to handle nuances of Persian-English translation.
</div>

---

## ✨ Features

- 🪶 **Lightweight** — Based on Qwen3-0.6B with only 3.28% trainable parameters
- ⚡ **Fast Inference** — Optimized with Unsloth for 2x faster performance
- 🎯 **High Accuracy** — Fine-tuned on 300K high-quality sentence pairs
- 💾 **Low Resource** — Runs efficiently on consumer hardware
- 📦 **Multiple Formats** — Available in GGUF format for local deployment
- 🔓 **Open Source** — MIT licensed for maximum flexibility

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Persian-To-English-Translator.git
cd Persian-To-English-Translator

# Install dependencies
pip install torch transformers accelerate
pip install unsloth peft
```

---

## 🚀 Quick Start

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "your-username/Persian-To-English-Translator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Translate
persian_text = "سلام، حال شما چطور است؟"
prompt = f"Translate the following Persian text to English:\n{persian_text}\nEnglish:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translation)
```

### Using GGUF (llama.cpp)

```bash
# Download GGUF model
wget https://huggingface.co/your-username/Persian-To-English-Translator-GGUF/resolve/main/model-q4_k_m.gguf

# Run with llama.cpp
./main -m model-q4_k_m.gguf -p "Translate Persian to English: سلام دنیا"
```

---

## 📊 Training Details

### Training Configuration

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 290,376 | Num Epochs = 2 | Total steps = 72,594
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
 "-____-"     Trainable parameters = 20,185,088 of 616,235,008 (3.28% trained)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen3-0.6B |
| **Fine-tuning Method** | PEFT LoRA |
| **Optimizer** | Unsloth |
| **Number of Epochs** | 2 |
| **Total Training Steps** | 72,594 |
| **Batch Size per Device** | 2 |
| **Gradient Accumulation Steps** | 4 |
| **Effective Batch Size** | 8 |
| **Trainable Parameters** | 20,185,088 (3.28%) |
| **Total Parameters** | 616,235,008 |
| **Training Examples** | 290,376 |

### Model Architecture

```
┌─────────────────────────────────────────┐
│           Qwen3-0.6B (Base)             │
├─────────────────────────────────────────┤
│     + LoRA Adapters (Rank: TBD)         │
│     + Unsloth Optimizations             │
├─────────────────────────────────────────┤
│   Trainable: 20.2M / 616.2M (3.28%)     │
└─────────────────────────────────────────┘
```

---

## 📚 Dataset

### Overview

The model was trained on **300,000 high-quality Persian-English sentence pairs**, carefully curated and cleaned to address specific challenges in Persian-to-English translation.

### Dataset Quality Comparison

| Dataset | Size | Quality | Persian-Specific | Cleaned | Notes |
|---------|------|---------|------------------|---------|-------|
| **Ours (Used)** | 300K | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Yes | Curated for FA→EN challenges |
| OPUS-100 | 1M+ | ⭐⭐⭐ | ❌ No | ❌ No | Generic multilingual |
| CCAligned | 500K+ | ⭐⭐ | ❌ No | ❌ No | Noisy web crawl |
| WikiMatrix | 200K | ⭐⭐⭐ | ❌ No | Partial | Wikipedia only |
| TED2020 | 50K | ⭐⭐⭐⭐ | ❌ No | ✅ Yes | Limited domain |

### Data Cleaning Pipeline

```
Raw Data → Deduplication → Length Filter → Quality Filter → Final Dataset
   │              │              │              │              │
  500K+        450K           380K           320K           300K
```

### Persian-Specific Challenges Addressed

- ✅ Right-to-Left (RTL) text handling
- ✅ Persian-specific characters and diacritics
- ✅ Informal/colloquial expressions
- ✅ Persian idioms and proverbs
- ✅ Mixed Persian-Arabic script
- ✅ Transliteration of names and places

---

## 📈 Performance

> 🚧 **Coming Soon**: BLEU Score and other evaluation metrics will be added after comprehensive benchmarking.

| Metric | Score | Status |
|--------|-------|--------|
| BLEU | TBD | 🔄 In Progress |
| chrF | TBD | 🔄 In Progress |
| COMET | TBD | 🔄 In Progress |
| Inference Speed | TBD | 🔄 In Progress |

---

## 📦 Model Checkpoints

| Format | Size | Use Case | Download |
|--------|------|----------|----------|
| Full Model | ~1.2GB | Training/Fine-tuning | [🤗 Hub](https://huggingface.co/) |
| GGUF Q4_K_M | ~400MB | Fast CPU Inference | [🤗 Hub](https://huggingface.co/) |
| GGUF Q8_0 | ~650MB | Balanced Quality/Speed | [🤗 Hub](https://huggingface.co/) |
| GGUF F16 | ~1.2GB | Maximum Quality | [🤗 Hub](https://huggingface.co/) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for training optimizations
- [Hugging Face](https://huggingface.co/) for the transformers library
- The Persian NLP community for valuable resources and feedback

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Made with ❤️ for the Persian NLP Community**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/Persian-To-English-Translator?style=social)](https://github.com/yourusername/Persian-To-English-Translator)

</div>
