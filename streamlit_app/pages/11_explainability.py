"""Page: Model Explainability (plan s.49, s.38)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from components.common import get_filtered, pii, section_header


def render() -> None:
    st.title("🔎 Model Explainability")
    st.caption("Attribution is computed with gradient-times-embedding for the "
               "predicted class. Attention weights alone are NOT treated as "
               "proof of reasoning (plan s.38).")
    df = get_filtered()

    models = {}
    try:
        from src import model_registry as mr

        for task in mr.LABEL_MAPS:
            d = mr.model_dir(task, "distilbert-base-uncased") / "model"
            has_cfg = (d / "config.json").exists()
            has_weights = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
            if has_cfg:
                models[task] = {"dir": d, "has_weights": has_weights}
    except Exception as e:
        st.warning(f"Could not inspect model artifacts: {e}")

    if not models:
        st.info("No trained model artifacts found yet.")
        st.markdown("""
Train one first, e.g.:

```bash
.venv/bin/python -m src.train --task sentiment --model distilbert-base-uncased --smoke
```

Until then this page explains how attribution works:

1. The review is tokenised and passed through the fine-tuned encoder.
2. The predicted class logit is back-propagated to the input embedding layer.
3. Each token's attribution = (gradient x embedding) summed over the hidden
   dimension, reported signed (supporting vs opposing the prediction).
4. The top-k absolute attributions are surfaced with the model's confidence.
""")
        return

    task = st.selectbox("Task", list(models.keys()))
    selected_meta = models[task]
    model_dir = selected_meta["dir"]
    has_weights = selected_meta["has_weights"]

    if has_weights:
        st.success(f"🟢 Active Model: Fine-tuned DistilBERT checkpoint for **{task}** loaded from disk.")
    else:
        st.info(f"ℹ️ Fine-tuned weights not saved on disk for **{task}**; attribution will run using the base **distilbert-base-uncased** architecture.")

    section_header("Select a review")
    
    # Filter for meaningful, multi-word reviews for better attribution insights
    clean_reviews = df.dropna(subset=["review_text_model"])
    meaningful = clean_reviews[clean_reviews["review_text_model"].astype(str).str.strip().str.len() > 15]
    sample = meaningful.head(200) if not meaningful.empty else clean_reviews.head(50)

    if not sample.empty:
        options = [f"{i}: {str(t)[:90]}" for i, t in enumerate(sample["review_text_model"].tolist())]
        pick = st.selectbox("Sample reviews (from current filter)", options)
        idx = int(pick.split(":")[0])
        default_text = sample["review_text_model"].iloc[idx]
    else:
        default_text = "Unable to login after recent update, OTP not coming and money got debited."

    text = st.text_area("Review text to explain", value=default_text, height=120)
    st.caption("PII-masked preview: " + pii(pd.Series([text]))[0])

    if st.button("Explain prediction", type="primary"):
        with st.spinner("Computing gradient-times-embedding attribution..."):
            try:
                from src.explainability import explain_text

                rows = explain_text(model_dir, text)
                head, tokens = rows[0], rows[1:]
                c1, c2 = st.columns(2)
                c1.metric("Predicted class ID", head["prediction_id"])
                c2.metric("Confidence", f"{head['confidence']:.3%}")
                
                token_df = pd.DataFrame(tokens)
                st.dataframe(token_df, use_container_width=True)
                
                pos = [r for r in tokens if r["attribution"] > 0]
                neg = [r for r in tokens if r["attribution"] < 0]
                if pos:
                    st.success("Supporting tokens (+): " + ", ".join(f"`{r['token']}`" for r in pos[:8]))
                if neg:
                    st.error("Opposing tokens (-): " + ", ".join(f"`{r['token']}`" for r in neg[:8]))
                st.caption("Attributions are local sensitivity scores for this model instance, reported signed.")
            except Exception as e:
                st.error(f"Attribution failed: {e}")


if __name__ == "__main__":
    render()
else:
    render()
