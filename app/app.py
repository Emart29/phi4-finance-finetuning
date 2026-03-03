"""
Phi-4 Mini Finance Assistant — FastAPI Backend
CPU-compatible version for HF Spaces free tier
No bitsandbytes / no 4-bit quantization
"""

import os
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="Phi-4 Finance Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_MODEL_ID = "microsoft/Phi-4-mini-instruct"
ADAPTER_ID    = "Emar7/phi4-finance-finetuned"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model (CPU, float32)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()
print("Model ready!")

# ── Schemas ───────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    context: str = ""

class AnswerResponse(BaseModel):
    answer: str
    question: str

# ── Inference ─────────────────────────────────────────────────────────────────
def generate_answer(question: str, context: str = "") -> str:
    user_content = f"{question}\n\nContext: {context}" if context.strip() else question

    prompt = f"""<|system|>
You are an expert financial analyst. Answer questions about company financials, earnings reports, and business performance accurately and concisely.<|end|>
<|user|>
{user_content}<|end|>
<|assistant|>
"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": ADAPTER_ID}

@app.post("/predict", response_model=AnswerResponse)
def predict(req: QuestionRequest):
    return AnswerResponse(
        answer=generate_answer(req.question, req.context),
        question=req.question
    )

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
