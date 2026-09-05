import re
from typing import List, Tuple

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

def load_nli_model(model_name: str = None, device: str = None):
    if TRANSFORMERS_AVAILABLE and model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            return tokenizer, model, device
        except Exception:
            pass
    return None, None, "cpu"

def _heuristic_nli_single(premise: str, hypothesis: str) -> Tuple[str, float]:
    negation_tokens = {"not", "n't", "never", "no", "none", "without", "neither", "denied", "refused", "untrue", "falsely"}
    stopwords = {"the", "a", "an", "and", "or", "in", "on", "of", "for", "to", "by", "with", "is", "was", "were", "it", "his", "her", "he", "she", "they"}

    def tokenize(s):
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return [t for t in s.split() if len(t) > 2 and t not in stopwords]

    h_tokens = set(tokenize(hypothesis))
    if not h_tokens:
        return "NEUTRAL", 0.5

    sentences = re.split(r'(?<=[.!?])\s+', premise)
    best_overlap = 0
    contradiction_found = False
    support_found = False

    for sent in sentences:
        sent_tokens = set(tokenize(sent))
        overlap = len(h_tokens & sent_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
        sent_words = set(re.findall(r'[a-z0-9]+', sent.lower()))
        has_neg = any(neg in sent_words for neg in negation_tokens)
        if overlap >= 2 and has_neg:
            contradiction_found = True
            break
        elif overlap >= 2:
            support_found = True

    if contradiction_found:
        return "CONTRADICTION", 0.85
    if support_found or best_overlap >= 3:
        return "ENTAILMENT", 0.80
    return "NEUTRAL", 0.50

def run_nli_batch(tokenizer, model, device, premises: List[str], hypotheses: List[str]) -> List[Tuple[str, float]]:
    assert len(premises) == len(hypotheses)
    if TRANSFORMERS_AVAILABLE and tokenizer is not None and model is not None:
        try:
            batch = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
                probs = F.softmax(logits, dim=-1).cpu()
            id2label = {int(k): v.upper() for k, v in model.config.id2label.items()} if hasattr(model.config, 'id2label') else None
            outputs = []
            for p in probs:
                top_idx = int(p.argmax().item())
                label = id2label[top_idx] if id2label is not None else str(top_idx)
                confidence = float(p[top_idx].item())
                outputs.append((label, confidence))
            return outputs
        except Exception:
            pass
    return [_heuristic_nli_single(p, h) for p, h in zip(premises, hypotheses)]
