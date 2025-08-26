# import argparse, os, sys, pandas as pd
# from tqdm import tqdm
# import nltk
# from nltk.tag import StanfordNERTagger
# from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab')

# # ----------------------------- CLI ------------------------------------
# ap = argparse.ArgumentParser()
# ap.add_argument("--csv", required=True, help="Input CSV with harassment rows")
# ap.add_argument("--out_har", default="harassment_filtered.csv",
#                 help="Output CSV for harassment rows (label 1)")
# ap.add_argument("--out_norm", default="normal_from_har.csv",
#                 help="Output CSV for non-bullying rows (label 0)")
# args = ap.parse_args()

# jar   = os.path.join("stanford-ner-2020-11-17/stanford-ner-4.2.0.jar")
# model = os.path.join("stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz")

# if not (os.path.isfile(jar) and os.path.isfile(model)):
#     sys.exit("✖︎  stanford-ner.jar or model not found—check --ner_dir path.")

# # ------------------------- Load dataset -------------------------------
# df = pd.read_csv(args.csv)
# print(f"Loaded {len(df):,} rows from {args.csv}")

# # change these to your column names if different
# text_col   = "comment_text"
# toxic_cols = ["malignant","highly_malignant","rude","threat","abuse","loathe"]

# if not all(c in df.columns for c in toxic_cols+[text_col]):
#     sys.exit("✖︎  Expected toxicity columns not found—inspect CSV header.")

# df["is_toxic"]     = df[toxic_cols].any(axis=1)
# df["is_nonbully"]  = ~df["is_toxic"]

# # ---------------------- Stanford NER setup ----------------------------
# print("Initialising StanfordNERTagger …")
# tagger = StanfordNERTagger(model, jar, encoding="utf-8")

# def has_person_or_org(text:str) -> bool:
#     tokens = word_tokenize(str(text))
#     tags   = tagger.tag(tokens)
#     return any(t[1] in ("PERSON","ORGANIZATION") for t in tags)

# tqdm.pandas(desc="NER filter")
# df["keep"] = df[text_col].progress_apply(has_person_or_org)

# # ---------------------- Split & save ----------------------------------
# harassment = df[df["is_toxic"] & df["keep"]][[text_col]].copy()
# harassment["label"] = 1
# harassment.to_csv(args.out_har, index=False)
# print(f"Harassment rows saved → {args.out_har}  ({len(harassment):,})")

# normal = df[df["is_nonbully"]][[text_col]].copy()
# normal["label"] = 0
# normal.to_csv(args.out_norm, index=False)
# print(f"Non-bullying rows saved → {args.out_norm}  ({len(normal):,})")

# print("✅  Done.")


import argparse, os, sys, pandas as pd, stanza, torch
from tqdm import tqdm

# ---------------- CLI -----------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="Input CSV (harassment dataset)")
ap.add_argument("--max_har", type=int, default=2000,
                help="Stop after this many harassment rows (default 2000)")
ap.add_argument("--out_har", default="harassment_filtered.csv",
                help="Output CSV for harassment rows (label 1)")
ap.add_argument("--out_norm", default="normal_from_har.csv",
                help="Output CSV for non-bullying rows (label 0)")
args = ap.parse_args()

# ------------- Load dataset ------------
text_col   = "comment_text"
toxic_cols = ["malignant","highly_malignant","rude","threat","abuse","loathe"]

df = pd.read_csv(args.csv)
if not all(c in df.columns for c in toxic_cols+[text_col]):
    sys.exit("❌ Column names differ—edit toxic_cols or text_col in script.")

df["is_toxic"]    = df[toxic_cols].any(axis=1)
df["is_nonbully"] = ~df["is_toxic"]

print(f"Rows total: {len(df):,}  •  toxic: {df.is_toxic.sum():,}  •  non_bully: {df.is_nonbully.sum():,}")

# ------------- Stanza pipeline ----------
print("⏳ Downloading Stanza model (first run only)…")
stanza.download("en", processors="tokenize,ner", verbose=False)

print("🚀 Initialising GPU NER pipeline…" if torch.cuda.is_available()
      else "⚠️  CUDA not available, running on CPU…")
nlp = stanza.Pipeline(
    "en",
    processors="tokenize,ner",
    use_gpu=torch.cuda.is_available(),
    verbose=False
)

def has_person_org(text: str) -> bool:
    doc = nlp(str(text))
    return any(ent.type in ("ORG") for ent in doc.ents)

# ------------- Filter loop --------------
har_rows = []
har_indices = []
with tqdm(total=args.max_har, desc="Collecting harassment rows") as pbar:
    for i, row in df[df.is_toxic].iterrows():
        if has_person_org(row[text_col]):
            har_rows.append(row[text_col])
            har_indices.append(i)
            pbar.update()
            if len(har_rows) >= args.max_har:
                break

harassment = pd.DataFrame({text_col: har_rows})
harassment["label"] = 1
print(f"Collected {len(harassment):,} harassment rows (limit {args.max_har}).")

normal = df[df.is_nonbully][[text_col]].copy()
normal["label"] = 0
print(f"Non-bullying rows: {len(normal):,}")

# ------------- Save ---------------------
harassment.to_csv(args.out_har, index=False)
normal.to_csv(args.out_norm, index=False)
print(f"✅  Saved:\n  {args.out_har}\n  {args.out_norm}")