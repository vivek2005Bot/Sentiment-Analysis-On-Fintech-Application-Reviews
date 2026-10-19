"""Explainability (plan section 38): token attribution for a trained model.

Gradient-times-embedding attribution for the predicted class. Attention
weights are NOT used as explanations (plan Rule s.38: attention alone does
not prove model reasoning).
"""

from __future__ import annotations

import argparse


from src import model_registry as mr
from src.utils import setup_logging

logger = setup_logging("explainability")


def explain_text(
    model_dir, text: str, device: str | None = None, top_k: int = 12
) -> list[dict]:
    """Return top tokens with signed attribution for the predicted class."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or mr.get_device()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    embeddings = model.get_input_embeddings()(enc["input_ids"]).detach()
    embeddings.requires_grad_(True)

    kwargs = {"inputs_embeds": embeddings}
    if "attention_mask" in enc:
        kwargs["attention_mask"] = enc["attention_mask"]
    out = model(**kwargs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    score = out.logits[0, pred_id]
    model.zero_grad()
    score.backward()

    grads = embeddings.grad[0]  # [seq, hidden]
    attributions = (grads * embeddings.detach()[0]).sum(-1)  # signed
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    mask = enc["attention_mask"][0].bool()

    rows = [
        {"token": t, "attribution": float(a), "position": i}
        for i, (t, a) in enumerate(zip(tokens, attributions))
        if mask[i]
    ]
    rows = [r for r in rows if r["token"] not in ("[CLS]", "[SEP]", "[PAD]")]
    rows.sort(key=lambda r: abs(r["attribution"]), reverse=True)
    top = rows[:top_k]
    for r in top:
        r["attribution"] = round(r["attribution"], 5)
    return [
        {"prediction_id": pred_id, "confidence": round(float(probs[pred_id]), 4)}
    ] + top


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    for row in explain_text(args.model_dir, args.text):
        print(row)
