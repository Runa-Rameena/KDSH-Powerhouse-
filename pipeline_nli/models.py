"""
Helper utilities for model loading and batched inference for NLI.
"""
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


def load_nli_model(model_name: str, device: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def run_nli_batch(tokenizer, model, device, premises: List[str], hypotheses: List[str]) -> List[Tuple[str, float]]:
    # Returns list of (label, confidence)
    assert len(premises) == len(hypotheses)
    outputs = []
    batch = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
        probs = F.softmax(logits, dim=-1).cpu()
    id2label = {int(k): v.upper() for k, v in model.config.id2label.items()} if hasattr(model.config, 'id2label') else None
    for p in probs:
        top_idx = int(p.argmax().item())
        label = id2label[top_idx] if id2label is not None else str(top_idx)
        confidence = float(p[top_idx].item())
        outputs.append((label, confidence))
    return outputs
