#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_supcon_pipeline.py
---------------------------------------------
Supervised-contrastive warm-up → cross-entropy fine-tune for DistilBERT (or any BERT-like HF model).

Usage:
  python run_supcon_pipeline.py \
    --csv /path/to/final_dataset_with_normal.csv \
    --outdir /path/to/outputs \
    --model distilbert-base-uncased \
    --lengths 256,320,384 \
    --supcon_epochs 1 --ce_epochs 4 \
    --temperature 0.1 --proj_dim 128 \
    --bsz_train 16 --bsz_eval 32 \
    --seed 42

Notes:
- CSV must have columns: text, label  (labels in {0,1,2,3})
- By default, checkpoints DURING training are disabled to save disk space.
- Final model + tokenizer are saved in each run directory.
"""

import os, json, argparse, numpy as np, pandas as pd, torch, evaluate
from datetime import datetime
from collections import Counter
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from transformers import __version__ as hf_version

# Quiet the tokenizer "fork" warning; also safer with 0 workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------- Metrics -----------------------------
acc = evaluate.load("accuracy")
f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# ------------------------ Tokenization utils -----------------------
from transformers import DataCollatorWithPadding

def tokenize_for(model_name, ds: Dataset, max_length=256):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def _tok(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    cols_to_remove = [c for c in ds.column_names if c not in ("text","label")]
    return ds.map(_tok, batched=True, remove_columns=cols_to_remove).with_format("torch")

# 2) SupCon model: encoder + projection head
class SupConEncoder(nn.Module):
    def __init__(self, model_name, proj_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)  # e.g., distilbert-base-uncased
        hid = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, proj_dim)
        )
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)                         # DistilBERT has no pooler
        cls = out.last_hidden_state[:, 0, :]                 # [B, H]
        z = self.proj(cls)                                   # [B, D]
        z = F.normalize(z, dim=-1)
        return z

# 3) SupCon loss (Khosla et al. 2020)
def supervised_contrastive_loss(z, labels, temperature=0.1):
    # z: [N, D], labels: [N]
    sim = torch.matmul(z, z.T) / temperature                 # [N, N]
    # mask to remove self-comparisons
    self_mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, float('-inf'))

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & (~self_mask)           # positives = same label, not self
    # log-softmax over rows
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # for rows with at least one positive, average log-prob over positives
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob.masked_fill(~pos_mask, 0).sum(dim=1) / pos_counts).mean()
    return loss

# 4) Trainer for SupCon: forward twice (dropout noise gives two "views")
class SupConTrainer(Trainer):
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # labels are used
        labels = inputs["labels"]
        # two stochastic passes through dropout = two views
        z1 = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        z2 = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        z = torch.cat([z1, z2], dim=0)                       # [2B, D]
        y = torch.cat([labels, labels], dim=0)               # [2B]
        loss = supervised_contrastive_loss(z, y, self.temperature)
        return (loss, {"z": z}) if return_outputs else loss

# ---------------------------- Orchestrator -------------------------
def supcon_then_ce(
    train_ds, val_ds,
    model_name="distilbert-base-uncased",
    output_dir="./outputs/run",
    max_length=320,
    supcon_epochs=1,
    supcon_lr=2e-5,
    temperature=0.1,
    proj_dim=128,
    ce_epochs=4,
    ce_lr=2e-5,
    bsz_train=16, bsz_eval=32,
    seed=42,
    early_stop=False,
    cache_dir=None,
):
    torch.manual_seed(seed); np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Tokenize once
    tok_train = tokenize_for(model_name, train_ds, max_length=max_length)
    tok_val     = tokenize_for(model_name, val_ds,   max_length=max_length)

    # -------------- Stage A: SupCon warm-up (no checkpoints) --------------
    if supcon_epochs > 0:
        supcon_model = SupConEncoder(model_name, proj_dim=proj_dim)
        supcon_args = TrainingArguments(
            output_dir=output_dir + "_supcon",
            eval_strategy="no",
            save_strategy="epoch",
            learning_rate=supcon_lr,
            per_device_train_batch_size=bsz_train,
            per_device_eval_batch_size=bsz_eval,
            num_train_epochs=supcon_epochs,
            warmup_ratio=0.06,
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
            seed=seed,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
        )
        supcon_trainer = SupConTrainer(
            model=supcon_model,
            args=supcon_args,
            train_dataset=tok_train,
            eval_dataset=None,
            compute_metrics=None,
            temperature=temperature
        )
        supcon_trainer.train()
    else:
        supcon_model = SupConEncoder(model_name, proj_dim=proj_dim)  # untrained encoder

    # -------------- Stage B: Cross-entropy fine-tune ----------------------
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4, cache_dir=cache_dir
    )
    # Robustly access base encoder and load warmed weights
    base_attr = getattr(clf_model, "base_model_prefix", None) or getattr(clf_model.config, "model_type", None)
    if base_attr and hasattr(clf_model, base_attr):
        getattr(clf_model, base_attr).load_state_dict(supcon_model.encoder.state_dict(), strict=False)
    else:
        # Fallback (works for many models): try .base_model if present
        if hasattr(clf_model, "base_model"):
            clf_model.base_model.load_state_dict(supcon_model.encoder.state_dict(), strict=False)

    ce_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no" if not early_stop else "epoch",
        save_total_limit=1,
        load_best_model_at_end=early_stop,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        learning_rate=ce_lr,
        per_device_train_batch_size=bsz_train,
        per_device_eval_batch_size=bsz_eval,
        num_train_epochs=ce_epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,

        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    ce_trainer = Trainer(
        model=clf_model,
        args=ce_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        compute_metrics=compute_metrics,
    )

    ce_trainer.train()
    metrics = ce_trainer.evaluate()

    # -------------------------- Save artifacts ---------------------------
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    preds = ce_trainer.predict(tok_val)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)

    label_names = {0:"Normal", 1:"Harassment", 2:"Defamation", 3:"Misleading"}
    report_txt = classification_report(
        y_true, y_pred, digits=2,
        target_names=[label_names[i] for i in [0,1,2,3]]
    )
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    pd.DataFrame(
        cm,
        index=[f"true_{label_names[i]}" for i in [0,1,2,3]],
        columns=[f"pred_{label_names[i]}" for i in [0,1,2,3]],
    ).to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    # Save model + tokenizer for deployment
    ce_trainer.save_model(output_dir)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.save_pretrained(output_dir)

    # Save run config
    cfg = dict(model_name=model_name, max_length=max_length, supcon_epochs=supcon_epochs,
               supcon_lr=supcon_lr, temperature=temperature, proj_dim=proj_dim,
               ce_epochs=ce_epochs, ce_lr=ce_lr, bsz_train=bsz_train, bsz_eval=bsz_eval,
               seed=seed, early_stop=early_stop, hf_version=hf_version)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n✅ Finished: {output_dir}")
    print(metrics)
    print(report_txt)

    return {
        "run_dir": output_dir,
        "accuracy": float(metrics.get("eval_accuracy", np.nan)),
        "f1_macro": float(metrics.get("eval_f1_macro", np.nan)),
        "max_length": max_length,
        "temperature": temperature,
        "supcon_epochs": supcon_epochs,
        "ce_epochs": ce_epochs,
        "bsz_train": bsz_train,
        "bsz_eval": bsz_eval,
        "seed": seed,
    }

# ------------------------------- Main --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: text,label")
    ap.add_argument("--outdir", required=True, help="Base output directory")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--lengths", default="320", help="Comma-separated max lengths, e.g., 256,320,384")
    ap.add_argument("--temperature", default="0.1", help="Comma-separated temps, e.g., 0.05,0.1,0.2")
    ap.add_argument("--supcon_epochs", type=int, default=1)
    ap.add_argument("--ce_epochs", type=int, default=4)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--bsz_train", type=int, default=16)
    ap.add_argument("--bsz_eval", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--early_stop", action="store_true", help="Enable epoch checkpoints + early best-model reloading")
    ap.add_argument("--cache_dir", default=None, help="Optional HF cache dir on a roomy disk")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"🔹 Transformers: {hf_version}")
    print(f"🔹 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load data
    df = pd.read_csv(args.csv)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Split (stratified)
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, stratify=df["label"], random_state=args.seed
    )
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    # Sweep
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    temps   = [float(x) for x in args.temperature.split(",") if x.strip()]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir  = os.path.join(args.outdir, f"supcon_runs_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    rows = []
    for L in lengths:
        for T in temps:
            run_dir = os.path.join(base_dir, f"len{L}_temp{T}")
            row = supcon_then_ce(
                train_ds=train_ds, val_ds=val_ds,
                model_name=args.model,
                output_dir=run_dir,
                max_length=L,
                supcon_epochs=args.supcon_epochs,
                supcon_lr=2e-5,
                temperature=T,
                proj_dim=args.proj_dim,
                ce_epochs=args.ce_epochs,
                ce_lr=2e-5,
                bsz_train=args.bsz_train, bsz_eval=args.bsz_eval,
                seed=args.seed,
                early_stop=args.early_stop,
                cache_dir=args.cache_dir,
            )
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(base_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n🏁 Summary written to: {summary_path}")

if __name__ == "__main__":
    main()
