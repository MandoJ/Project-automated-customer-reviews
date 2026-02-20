from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]

HF_MODEL_ID = "YOUR_USERNAME/YOUR_MODEL_REPO"

THEME_CSV = ROOT / "reports" / "artifacts" / "theme_dashboard_with_meta.csv"
THEME_CSV_FALLBACK = ROOT / "reports" / "artifacts" / "theme_dashboard.csv"
UMAP_IMG = ROOT / "reports" / "figures" / "umap_clusters.png"

ARTICLES_JSON = ROOT / "reports" / "artifacts" / "recommendation_articles.json"
ARTICLES_MD = ROOT / "reports" / "artifacts" / "recommendation_articles.md"

LABELS = ["negative", "neutral", "positive"]


def load_sentiment_model():
    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    mdl.eval()
    return tok, mdl


tokenizer, model = load_sentiment_model()


def predict_sentiment(text: str):
    text = (text or "").strip()
    if not text:
        return "", 0.0

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])


def load_theme_dashboard() -> pd.DataFrame:
    if THEME_CSV.exists():
        df = pd.read_csv(THEME_CSV)
    elif THEME_CSV_FALLBACK.exists():
        df = pd.read_csv(THEME_CSV_FALLBACK)
    else:
        return pd.DataFrame(columns=["meta_category", "sentiment", "cluster_id", "size", "top_phrases"])

    keep = [c for c in ["meta_category", "sentiment", "cluster_id", "size", "top_phrases"] if c in df.columns]
    df = df[keep].copy()

    if "size" in df.columns:
        df = df.sort_values(["meta_category", "size"], ascending=[True, False], kind="mergesort")
    return df


def get_top_clusters(n: int = 25) -> pd.DataFrame:
    df = load_theme_dashboard()
    return df.head(n) if not df.empty else df


def load_articles():
    if ARTICLES_JSON.exists():
        with open(ARTICLES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data

    if ARTICLES_MD.exists():
        return {"All writeups": ARTICLES_MD.read_text(encoding="utf-8")}

    return {}


ARTICLES = load_articles()
ARTICLE_KEYS = sorted(ARTICLES.keys()) if ARTICLES else []


def show_article(key: str) -> str:
    if not ARTICLES:
        return "No writeups found. Make sure `reports/artifacts/recommendation_articles.json` exists."
    return ARTICLES.get(key, "No content for that selection.")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Automated Customer Reviews") as demo:
        gr.Markdown(
            "# Automated Customer Reviews\n"
            "This demo runs our fine-tuned DistilBERT sentiment model (loaded from Hugging Face), "
            "plus the clustering + recommendation outputs."
        )

        with gr.Tabs():
            with gr.TabItem("1) Sentiment (Our DistilBERT)"):
                review = gr.Textbox(label="Paste a review", lines=5)
                with gr.Row():
                    pred = gr.Textbox(label="Predicted sentiment", interactive=False)
                    conf = gr.Number(label="Confidence", interactive=False)
                btn = gr.Button("Predict")

                gr.Examples(
                    examples=[
                        ["This product is awesome. Setup was easy and it works perfectly."],
                        ["It arrived on time and matches the description. I haven’t used it enough to judge yet."],
                        ["Terrible quality. Stopped working after two days and support was useless."],
                    ],
                    inputs=review,
                )

                btn.click(fn=predict_sentiment, inputs=review, outputs=[pred, conf])

            with gr.TabItem("2) Themes (Clusters)"):
                if UMAP_IMG.exists():
                    gr.Image(value=str(UMAP_IMG), interactive=False)
                clusters_table = gr.Dataframe(value=get_top_clusters(25), interactive=False, wrap=True)
                refresh = gr.Button("Refresh table")
                refresh.click(fn=lambda: get_top_clusters(25), inputs=None, outputs=clusters_table)

            with gr.TabItem("3) Recommendation writeups"):
                if ARTICLE_KEYS:
                    pick = gr.Dropdown(choices=ARTICLE_KEYS, value=ARTICLE_KEYS[0], label="Meta-category")
                    out = gr.Markdown(value=show_article(ARTICLE_KEYS[0]))
                    pick.change(fn=show_article, inputs=pick, outputs=out)
                else:
                    gr.Markdown("No writeups found.")

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)