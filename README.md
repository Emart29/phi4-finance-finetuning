# 💼 FinSight AI — Phi-4 Mini Finance Fine-Tuning

![Model](https://img.shields.io/badge/Model-Phi--4%20Mini%203.8B-blue?style=for-the-badge)
![Method](https://img.shields.io/badge/Method-QLoRA-orange?style=for-the-badge)
![ROUGE-L](https://img.shields.io/badge/ROUGE--L-%2B69%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

Fine-tuning Microsoft Phi-4 Mini 3.8B on SEC 10-K financial Q&A data using QLoRA.
Full pipeline: data prep → training → evaluation → FastAPI → live dashboard.

[🤗 Model](https://huggingface.co/Emar7/phi4-finance-finetuned) · [🌐 Live Demo](https://emar7-phi4-finance-demo.hf.space) · [📓 Notebook](https://www.kaggle.com/code/nwangumaemmanuel/phi4-finance-finetuning)

## 📊 Results

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|-----------|------------|-------------|
| ROUGE-1 | 0.4657 | 0.7523 | **+61.6%** |
| ROUGE-2 | 0.3560 | 0.6106 | **+71.5%** |
| ROUGE-L | 0.4242 | 0.7168 | **+69.0%** |

## 🏗️ Structure
```
phi4-finance-finetuning/
├── notebooks/phi4_finance_finetuning.ipynb
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── static/index.html
├── results/eval_results.json
└── README.md
```

## 🔧 Training Config
```
Base:         microsoft/Phi-4-mini-instruct (3.8B)
Method:       QLoRA — 4-bit NF4 + LoRA (r=16, alpha=32)
Data:         virattt/financial-qa-10K (6,300 train samples)
Hardware:     Kaggle 2x Tesla T4 (free)
Time:         ~3 hours | 3 epochs | Train loss: 1.07
Trainable:    ~40M / 3.8B params (≈1%)
```

## 💡 Example

**Q:** What does it mean when revenue grows but net income declines?

**Base model:** Generic explanation with no financial specificity.

**Fine-tuned:** This pattern indicates margin compression — rising costs outpacing revenue. Key 10-K areas: COGS trends, SG&A increases, higher interest expense, one-time charges. Analysts treat this as a red flag for pricing power and operational efficiency.

## 🖥️ Run Locally
```bash
cd app
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# Dashboard → http://localhost:8000
```

## 🔗 Links
| | |
|--|--|
| 🤗 Model | [Emar7/phi4-finance-finetuned](https://huggingface.co/Emar7/phi4-finance-finetuned) |
| 🌐 Demo | [emar7-phi4-finance-demo.hf.space](https://emar7-phi4-finance-demo.hf.space) |
| 📓 Notebook | [Kaggle](https://www.kaggle.com/code/nwangumaemmanuel/phi4-finance-finetuning) |
| 📦 Base model | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| 📊 Dataset | [virattt/financial-qa-10K](https://huggingface.co/datasets/virattt/financial-qa-10K) |

## 👤 Author
**Emmanuel Nwanguma** — ML Engineer

## 📄 License
MIT
