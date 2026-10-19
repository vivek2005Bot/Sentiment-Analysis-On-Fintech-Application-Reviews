"""Page: Model Explainability (plan s.49, s.38)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import get_filtered, pii


def render() -> None:
    st.title("🔎 Model Explainability")
    st.caption("Attribution is computed with gradient-times-embedding for the "
               "predicted class. Attention weights alone are NOT treated as "
               "proof of reasoning (plan s.38).")
    df = get_filtered()

    models = []
    try:
        from src import model_registry as mr

        for task in mr.LABEL_MAPS:
            d = mr.model_dir(task, "distilbert-base-uncased") / "model"
            if (d / "config.json").exists():
                models.append((task, d))
    except Exception as e:
        st.warning(f"Could not inspect model artifacts: {e}")

    if not models:
        st.info("No trained model artifacts found yet.")
        st.markdown("""
Train one first, e.g.:

```bash
.venv/bin/python -m src.train --task sentiment --model distilbert-base-uncased
# then a fuller run (no --smoke) to save the artifact under models/<task>/
```

Until then this page explains how attribution works:

1. The review is tokenised and passed through the fine-tuned encoder.
2. The predicted class logit is back-propagated to the input embedding layer.
3. Each token's attribution = (gradient x embedding) summed over the hidden
   dimension, reported signed (supporting vs opposing the prediction).
4. The top-k absolute attributions are surfaced with the model's confidence.

Known limits: attribution explains *this* model's local sensitivity, not
ground truth reasoning; it is not a causal explanation of the phenomenon.
""")
        return

    task = st.selectbox("Task", [t for t, _ in models])
    model_dir = dict(models)[task]

    st.subheader("Select a review")
    sample = df.dropna(subset=["review_text_model"]).head(200)
    options = [f"{i}: {str(t)[:90]}" for i, t in
               enumerate(sample["review_text_model"].tolist())]
    pick = st.selectbox("Sample reviews (from current filter)", options)
    idx = int(pick.split(":")[0])
    default_text = sample["review_text_model"].iloc[idx]

    text = st.text_area("Review text", value=default_text, height=120)
    st.caption("PII-masked preview: " +
               pii(pd.Series([text]))[0])

    if st.button("Explain prediction", type="primary"):
        with st.spinner("Running attribution..."):
            try:
                from src.explainability import explain_text

                rows = explain_text(model_dir, text)
                head, tokens = rows[0], rows[1:]
                st.metric("Predicted class id", head["prediction_id"])
                st.metric("Confidence", f"{head['confidence']:.3f}")
                st.dataframe(tokens, width="stretch")
                pos = [r for r in tokens if r["attribution"] > 0]
                neg = [r for r in tokens if r["attribution"] < 0]
                if pos:
                    st.success("Supporting tokens: " +
                               ", ".join(r["token"] for r in pos[:8]))
                if neg:
                    st.error("Opposing tokens: " +
                             ", ".join(r["token"] for r in neg[:8]))
                st.caption("Attributions are local sensitivity scores for this "
                           "model instance, reported signed.")
            except Exception as e:
                st.error(f"Attribution failed: {e}")


if __name__ == "__main__":
    render()
else:
    render()
