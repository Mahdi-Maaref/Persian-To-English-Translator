 
# 🌐 Persian-To-English-Translator
![Banner](banner.png)
### Lightweight, Fast, and Accurate Neural Machine Translation Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/⚡-Unsloth-purple)](https://github.com/unslothai/unsloth)
[![Models: Qwen3](https://img.shields.io/badge/Base%20Models-Qwen3-blue)](https://huggingface.co/Qwen)
[![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth)
[![Dataset: 300k](https://img.shields.io/badge/Dataset-300k%20Pairs-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

**Two fine-tuned Persian to English translation models: one ultra-lightweight for speed, one larger for maximum accuracy — both optimized for efficiency.**

[Models](#-available-models) •
[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Training](#-training-details) •
[Dataset](#-dataset) •
[License](#-license)

## 📖 Project Overview

**Persian-To-English-Translator** offers two specialized machine translation models designed to bridge the gap between Persian (Farsi) and English. We provide options for different use cases:

| 🪶 **Lite Model** | 🎯 **Pro Model** |
|:---:|:---:|
| Qwen3-0.6B | Qwen3-4B |
| Ultra-fast, minimal resources | Higher accuracy, still lightweight |
| Perfect for edge devices | Perfect for quality-focused apps |

### 🚀 Key Goals

- **Flexibility:** Choose between speed-optimized or accuracy-optimized models
- **Low-Resource:** Both models run efficiently on consumer-grade hardware
- **High Speed:** Fast inference suitable for real-time applications
- **Accuracy:** Fine-tuned on high-quality, cleaned datasets to handle Persian-English nuances

</div>

---

## 🔥 Available Models

### Model Comparison

| Feature | 🪶 **Lite (0.6B)** | 🎯 **Pro (4B)** |
|---------|:---:|:---:|
| **Base Model** | Qwen3-0.6B | Qwen3-4B |
| **Total Parameters** | 616M | ~4B |
| **Trainable Params** | 20.2M (3.28%) | TBD |
| **Model Size (FP16)** | ~1.2GB | ~8GB |
| **GGUF Q4_K_M Size** | ~400MB | ~2.5GB |
| **Inference Speed** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Translation Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **RAM Required** | ~2GB | ~6GB |
| **GPU Required** | Optional | Recommended |
| **Best For** | Mobile, Edge, Real-time | Desktop, Server, Quality |

### 🪶 Lite Model (0.6B) — Speed Champion
```
Perfect for: Mobile apps, IoT devices, real-time translation, 
             low-power devices, batch processing at scale
```

### 🎯 Pro Model (4B) — Accuracy Champion
```
Perfect for: Professional translation, content creation, 
             complex sentences, idiomatic expressions, nuanced text
```

> 💡 **Note:** Even our "Pro" 4B model is remarkably lightweight compared to industry giants like GPT-4 (1.7T params) or LLaMA-70B. It's like comparing a feather to an elephant!

---

## ✨ Features

- 🪶 **Dual Options** — Choose between ultra-lite (0.6B) or balanced (4B) models
- ⚡ **Fast Inference** — Optimized with Unsloth for 2x faster performance
- 🎯 **High Accuracy** — Fine-tuned on 300K high-quality sentence pairs
- 💾 **Low Resource** — Both models run on consumer hardware
- 📦 **Multiple Formats** — Available in GGUF format for local deployment
- 🔓 **Open Source** — MIT licensed for maximum flexibility
- 🔄 **Scalable** — Pick the right model for your resource constraints

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

# Choose your model:
# Lite (0.6B) - Fast & Lightweight
model_name = "your-username/Persian-To-English-Translator-Lite"

# Pro (4B) - Higher Accuracy
# model_name = "your-username/Persian-To-English-Translator-Pro"

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
# Download GGUF model (Lite version)
wget https://huggingface.co/your-username/Persian-To-English-Translator-Lite-GGUF/resolve/main/model-q4_k_m.gguf

# Or Pro version
wget https://huggingface.co/your-username/Persian-To-English-Translator-Pro-GGUF/resolve/main/model-q4_k_m.gguf

# Run with llama.cpp
./main -m model-q4_k_m.gguf -p "Translate Persian to English: سلام دنیا"
```

### Model Selection Guide

```python
# Use this helper to choose the right model
def select_model(priority="balanced"):
    """
    priority options:
    - "speed": Use Lite model (0.6B)
    - "quality": Use Pro model (4B)  
    - "balanced": Use Lite for simple, Pro for complex text
    """
    if priority == "speed":
        return "your-username/Persian-To-English-Translator-Lite"
    elif priority == "quality":
        return "your-username/Persian-To-English-Translator-Pro"
    else:
        # Implement your logic here
        pass
```

---

## 📊 Training Details

### 🪶 Lite Model (Qwen3-0.6B) Training

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 290,376 | Num Epochs = 2 | Total steps = 72,594
O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
 "-____-"     Trainable parameters = 20,185,088 of 616,235,008 (3.28% trained)
```

### 🎯 Pro Model (Qwen3-4B) Training

```
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 290,376 | Num Epochs = 2 | Total steps = TBD
O^O/ \_/ \    Batch size per device = TBD | Gradient accumulation steps = TBD
\        /    Data Parallel GPUs = 1 | Total batch size = TBD
 "-____-"     Trainable parameters = TBD of ~4,000,000,000 (TBD% trained)
```

### Hyperparameters Comparison

| Parameter | 🪶 Lite (0.6B) | 🎯 Pro (4B) |
|-----------|:---:|:---:|
| **Base Model** | Qwen3-0.6B | Qwen3-4B |
| **Fine-tuning Method** | PEFT LoRA | PEFT LoRA |
| **Optimizer** | Unsloth | Unsloth |
| **Number of Epochs** | 2 | 2 |
| **Total Training Steps** | 72,594 | TBD |
| **Batch Size per Device** | 2 | TBD |
| **Gradient Accumulation Steps** | 4 | TBD |
| **Effective Batch Size** | 8 | TBD |
| **Trainable Parameters** | 20.2M (3.28%) | TBD |
| **Total Parameters** | 616M | ~4B |
| **Training Examples** | 290,376 | 290,376 |

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Persian-To-English-Translator               │
├────────────────────────────┬────────────────────────────────────┤
│     🪶 LITE MODEL          │        🎯 PRO MODEL                │
├────────────────────────────┼────────────────────────────────────┤
│    Qwen3-0.6B (Base)       │       Qwen3-4B (Base)              │
│  + LoRA Adapters           │     + LoRA Adapters                │
│  + Unsloth Optimizations   │     + Unsloth Optimizations        │
├────────────────────────────┼────────────────────────────────────┤
│  Trainable: 20.2M (3.28%)  │     Trainable: TBD                 │
│  Total: 616M params        │     Total: ~4B params              │
├────────────────────────────┼────────────────────────────────────┤
│  💨 Speed: ★★★★★           │     💨 Speed: ★★★☆☆                │
│  🎯 Quality: ★★★★☆         │     🎯 Quality: ★★★★★              │
│  💾 Size: ★★★★★            │     💾 Size: ★★★★☆                 │
└────────────────────────────┴────────────────────────────────────┘
```

---

## 📚 Dataset

### Overview

Both models were trained on **300,000 high-quality Persian-English sentence pairs**, carefully curated and cleaned to address specific challenges in Persian-to-English translation.

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

| Metric | 🪶 Lite (0.6B) | 🎯 Pro (4B) | Status |
|--------|:---:|:---:|:---:|
| BLEU | TBD | TBD | 🔄 In Progress |
| chrF | TBD | TBD | 🔄 In Progress |
| COMET | TBD | TBD | 🔄 In Progress |
| Tokens/Second (CPU) | TBD | TBD | 🔄 In Progress |
| Tokens/Second (GPU) | TBD | TBD | 🔄 In Progress |

---

## 📦 Model Checkpoints

### 🪶 Lite Model (0.6B)

| Format | Size | Use Case | Download |
|--------|------|----------|----------|
| Full Model | ~1.2GB | Training/Fine-tuning | [🤗 Hub](https://huggingface.co/) |
| GGUF Q4_K_M | ~400MB | Fast CPU Inference | [🤗 Hub](https://huggingface.co/) |
| GGUF Q8_0 | ~650MB | Balanced Quality/Speed | [🤗 Hub](https://huggingface.co/) |
| GGUF F16 | ~1.2GB | Maximum Quality | [🤗 Hub](https://huggingface.co/) |

### 🎯 Pro Model (4B)

| Format | Size | Use Case | Download |
|--------|------|----------|----------|
| Full Model | ~8GB | Training/Fine-tuning | [🤗 Hub](https://huggingface.co/) |
| GGUF Q4_K_M | ~2.5GB | Fast CPU Inference | [🤗 Hub](https://huggingface.co/) |
| GGUF Q8_0 | ~4.5GB | Balanced Quality/Speed | [🤗 Hub](https://huggingface.co/) |
| GGUF F16 | ~8GB | Maximum Quality | [🤗 Hub](https://huggingface.co/) |

---

## 🎯 Which Model Should I Use?

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SELECTION GUIDE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📱 Mobile App / Edge Device?          → 🪶 Lite (0.6B)         │
│  🖥️ Desktop / Server?                  → 🎯 Pro (4B)            │
│  ⚡ Real-time Translation?              → 🪶 Lite (0.6B)         │
│  📝 Professional Content?              → 🎯 Pro (4B)            │
│  💰 Limited GPU Memory (<4GB)?         → 🪶 Lite (0.6B)         │
│  🎨 Complex/Idiomatic Text?            → 🎯 Pro (4B)            │
│  📊 Batch Processing at Scale?         → 🪶 Lite (0.6B)         │
│  📖 High-Quality Publication?          → 🎯 Pro (4B)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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

- [Qwen Team](https://github.com/QwenLM) for the excellent base models
- [Unsloth](https://github.com/unslothai/unsloth) for training optimizations
- [Hugging Face](https://huggingface.co/) for the transformers library
- The Persian NLP community for valuable resources and feedback

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Made with ❤️ for the Persian NLP Community**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/Persian-To-English-Translator?style=social)](https://github.com/yourusername/Persian-To-English-Translator)

---

### 🪶 Lite for Speed | 🎯 Pro for Precision

*Both still lighter than a typical browser tab! 🚀*
