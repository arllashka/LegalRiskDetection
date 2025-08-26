# LegalRisk: Multi-class Legal-Risk Detection on Social-Media Posts

Detect and triage **Defamation**, **Harassment**, and **Misleading advertising** in short-form posts (with **Normal** as the fourth class). We compare classic ML baselines with transformer models (DistilBERT and LegalBERT), and release an end-to-end pipeline for dataset construction, training, and evaluation.

> ⚠️ **Disclaimer:** This project is a research prototype. It does **not** provide legal advice, and model outputs must be reviewed by humans before decisions are made.

---

## Contents

- [Overview](#overview)  
- [Labels](#labels)  
- [Data Pipeline](#data-pipeline)  
- [Repository Structure](#repository-structure)  
- [Setup](#setup)  
- [Quickstart](#quickstart)

---

## Overview

- **Task:** 4-class text classification on social-media / short news:
  - `0 = Normal`, `1 = Harassment`, `2 = Defamation`, `3 = Misleading`
- **Total size:** 3,690 posts  
  - Normal: 1,747 • Harassment: 1,261 • Defamation: 249 • Misleading: 433
- **Models:** Logistic Regression, Linear SVM, Random Forest; DistilBERT; LegalBERT
- **Extras:** Optional supervised contrastive warm-up, class-balanced losses, thorough EDA

---

## Labels

- **Normal (0):** Benign posts/news and brand updates  
- **Harassment (1):** Targeted insults, threats, abuse (mapped from toxic tags & ASA “offensive” cases)  
- **Defamation (2):** Allegations or false statements harming reputation (curated from a celebrity rumor corpus; true counterparts → Normal)  
- **Misleading (3):** Promotional claims likely to mislead (ASA “misleading” rulings; e.g., unsubstantiated environmental claims, non-disclosed ads)

---

## Data Pipeline

1. **ASA rulings scraping (Misleading & Offensive):**
   - **Stage 1:** harvest ruling links from topic pages into a CSV index  
   - **Stage 2:** fetch each page (with on-disk HTML cache + 1s delay), normalize sections (`title`, `ad_description`, `assessment`, etc.), build labeled texts
2. **Harassment source:** toxic comment corpus (6 tags collapsed into a single Harassment label) + ASA “offensive” cases  
   - Filter with **NER** to keep posts mentioning a person/organization (targeted abuse)
3. **Defamation source:** “fake” rumor articles (celebrity news) → **Defamation**; matched “true” article → **Normal**
4. **Normal source:** true news, non-bullying comments, plus ~300 neutral brand tweets (Twitter API v2)
5. **Cleaning:** deduplication, lowercasing, URL normalization; truncation at model max length during training

> Scraping & parsing code: `data/asa_scrape_index.py` and `data/asa_scrape_cases.py`.  
> Synthetic “defamation-style” augmentation with **fictional placeholders** is included (optional).

---


## Repository Structure


```
.
├─ data/
│ └─ finalMainDataset/ # Final dataset (incl. normal tweets & synthetic additions)
│ ├─ final_dataset_with_normal.csv # Main CSV (text,label,source,split,...) ← expected filename
│ └─ README.md # Data cards / provenance notes (recommended)
├─ MainCode/
│ ├─ distillExperiments.py # DistilBERT runner
│ ├─ legalBertRunner.py # LegalBERT runner
│ ├─ scrapper.py # ASA rulings scraper (BeautifulSoup)
│ └─ classicModels.ipynb # Classic ML baselines (LR/SVM/RF)
├─ results/ # Saved reports/metrics/checkpoints
│
└─ README.md
```


---

## Setup

**Python:** 3.10+ recommended

**Install (CPU-only minimal):**
```bash
python -m pip install -U pip
pip install -U pandas numpy scikit-learn matplotlib tqdm beautifulsoup4 requests gensim nltk stanza evaluate
pip install -U transformers datasets accelerate sentencepiece safetensors
```

Install (PyTorch + CUDA): visit https://pytorch.org and install a build matching your CUDA/driver. Example:

```
# Example for CUDA 12.x (adjust per your system)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Optional model/data assets:
```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
python -c "import stanza; stanza.download('en')"
```

Silence tokenizers warning (optional):
```
export TOKENIZERS_PARALLELISM=false
```
