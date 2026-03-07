# 💼 FinSight AI — Phi-4 Mini Finance Fine-Tuning

> Fine-tuning Microsoft Phi-4 Mini 3.8B to answer complex SEC 10-K financial 
> questions with 69% better accuracy than the base model — trained on free 
> Kaggle GPUs in under 3 hours.

[![Model](https://img.shields.io/badge/🤗_Model-Emar7/phi4--finance--finetuned-blue)](https://huggingface.co/Emar7/phi4-finance-finetuned)
[![Demo](https://img.shields.io/badge/🌐_Live_Demo-HuggingFace_Spaces-brightgreen)](https://emar7-phi4-finance-demo.hf.space)
[![Notebook](https://img.shields.io/badge/📓_Notebook-Kaggle-orange)](https://www.kaggle.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why This Project Exists

Financial analysts spend hours reading SEC 10-K filings to extract 
insights — revenue trends, risk factors, operational efficiency signals. 
General-purpose LLMs answer these questions poorly because they lack 
deep financial domain grounding.

This project fine-tunes Microsoft's Phi-4 Mini (3.8B parameters) on 
real SEC 10-K Q&A data using QLoRA — a parameter-efficient method that 
trains only ~1% of the model's weights. The result: a compact, 
deployable finance expert that outperforms the base model by 69% on 
ROUGE-L, running on hardware anyone can access for free.

---

## 📊 Results

| Metric   | Base Model | Fine-Tuned | Improvement |
|----------|-----------|------------|-------------|
| ROUGE-1  | 0.4657    | 0.7523     | **+61.6%**  |
| ROUGE-2  | 0.3560    | 0.6106     | **+71.5%**  |
| ROUGE-L  | 0.4242    | 0.7168     | **+69.0%**  |

*Evaluated on held-out SEC 10-K Q&A pairs from the financial-qa-10K dataset.*

---

## 💡 What the Fine-Tuning Actually Changed

**Q: What does it mean when revenue grows but net income declines?**

**Base model:**
> "When revenue grows but net income declines, it means the company's 
> expenses increased at a faster rate than its revenue..."

*(Generic — no financial specificity, no actionable insight)*

**Fine-tuned model:**
> "This pattern indicates margin compression — rising costs outpacing 
> revenue growth. In a 10-K, look at: COGS trends (gross margin erosion), 
> SG&A increases (scaling inefficiency), higher interest expense (leverage 
> risk), and one-time charges. Analysts treat persistent margin compression 
> as a red flag for pricing power and operational efficiency — especially 
> when it occurs despite top-line growth."

*(Domain-specific, cites 10-K sections, analyst-grade reasoning)*

---

## 🔧 Training Configuration
```
Base Model:        microsoft/Phi-4-mini-instruct (3.8B params)
Method:            QLoRA — 4-bit NF4 quantization
LoRA Config:       r=16, alpha=32, dropout=0.05
Trainable Params:  ~40M / 3.8B (≈1% of total)
Dataset:           virattt/financial-qa-10K (6,300 train samples)
Hardware:          Kaggle 2x Tesla T4 (free tier)
Training Time:     ~3 hours | 3 epochs
Final Train Loss:  1.07
```

**Why QLoRA?**
Full fine-tuning a 3.8B model requires 30+ GB of VRAM. QLoRA quantizes 
the base model to 4-bit and trains small low-rank adapter layers, making 
it possible to fine-tune on 2x T4s (16GB each) — hardware available 
for free on Kaggle. This is a practical, reproducible approach for 
domain adaptation on a budget.

---

## 🏗️ Project Structure
```
phi4-finance-finetuning/
├── notebooks/
│   └── phi4_finance_finetuning.ipynb  # Full pipeline: prep → train → eval
├── app/
│   ├── app.py                         # FastAPI inference server
│   ├── requirements.txt
│   └── static/index.html              # FinSight AI dashboard
├── results/
│   └── eval_results.json              # ROUGE scores + evaluation output
└── README.md
```

---

## 🖥️ Run Locally

**Prerequisites:** Python 3.10+, 8GB+ RAM (16GB recommended)
```bash
# Clone the repo
git clone https://github.com/Emart29/phi4-finance-finetuning.git
cd phi4-finance-finetuning

# Install dependencies
cd app
pip install -r requirements.txt

# Start the inference server
uvicorn app:app --host 0.0.0.0 --port 8000

# Open the dashboard
# → http://localhost:8000
```

The app loads the fine-tuned model from Hugging Face automatically 
on first run. Subsequent runs use the cached model.

---

## 📓 Reproduce the Training

The full training pipeline is in `notebooks/phi4_finance_finetuning.ipynb`.

To run it yourself:
1. Open the notebook on [Kaggle](https://www.kaggle.com) (free T4 GPUs)
2. Add your Hugging Face token as a Kaggle secret (`HF_TOKEN`)
3. Run all cells — total time ~3 hours

The notebook covers:
- Dataset loading and prompt formatting
- QLoRA config and LoRA adapter setup
- Training with `trl` SFTTrainer
- ROUGE evaluation against the base model
- Pushing the adapter to Hugging Face Hub

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🤗 Fine-tuned Model | [Emar7/phi4-finance-finetuned](https://huggingface.co/Emar7/phi4-finance-finetuned) |
| 🌐 Live Demo | [HuggingFace Spaces](https://emar7-phi4-finance-demo.hf.space) |
| 📓 Training Notebook | [Kaggle](https://www.kaggle.com) |
| 📦 Base Model | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| 📊 Dataset | [virattt/financial-qa-10K](https://huggingface.co/datasets/virattt/financial-qa-10K) |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Base Model | Microsoft Phi-4 Mini Instruct (3.8B) |
| Fine-tuning | QLoRA, PEFT, trl SFTTrainer |
| Quantization | bitsandbytes (4-bit NF4) |
| Evaluation | ROUGE-1/2/L (rouge-score) |
| Serving | FastAPI, Uvicorn |
| Hardware | Kaggle 2x Tesla T4 (free) |
| Model Hub | Hugging Face Hub |

---

## 👤 Author

**Emmanuel Nwanguma** — ML Engineer & Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/nwangumaemmanuel)
[![GitHub](https://img.shields.io/badge/GitHub-Emart29-black)](https://github.com/Emart29)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:nwangumaemmanuel29@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
