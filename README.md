# Automated Customer Reviews

This project builds an NLP pipeline that turns raw product reviews into practical insights.
It covers three core capabilities:

1) **Sentiment classification** — label reviews as **negative / neutral / positive**
2) **Category clustering** — group products into **4–6 meta-categories**
3) **Generative summaries** — generate short “recommendation-style” writeups per category

A lightweight web demo is included to interact with all three components.

---

## Project structure

- `notebooks/` — experiments and results (preprocessing, modeling, evaluation)
- `data/raw/` — original dataset (ignored by git)
- `data/processed/` — cleaned train/test splits (ignored by git)
- `src/` — reusable Python modules (training/inference utilities)
- `app/` — Gradio demo app
- `reports/` — report assets / final writeup
- `slides/` — presentation assets

---

## Dataset

Kaggle: “Consumer Reviews of Amazon Products”  
File used: `1429_1.csv`

> Note: raw and processed data files are intentionally not committed to this repository.

---

## How to run

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
