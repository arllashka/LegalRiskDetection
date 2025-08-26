#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, numpy as np, pandas as pd, torch, evaluate
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

# ------------------------- Metrics -------------------------
acc = evaluate.load("accuracy")
f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# ---------------------- Tokenization ----------------------
def tokenize_for(model_name, ds: Dataset, max_length=256):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def _tok(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    cols_to_remove = [c for c in ds.column_names if c not in ("text","label")]
    return ds.map(_tok, batched=True, remove_columns=cols_to_remove).with_format("torch")

# ------------------ Class weights / Focal -----------------
def get_class_weights(train_ds, num_labels=4):
    counts = Counter(train_ds["label"])
    total  = sum(counts.values())
    freqs  = np.array([counts.get(i,0) for i in range(num_labels)], dtype=np.float32)
    weights = total / (num_labels * np.maximum(freqs, 1))
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss

class LossTrainer(Trainer):
    """Trainer with weighted CE or focal loss via kwargs."""
    def __init__(self, loss_type="ce", class_weights=None, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.focal = FocalLoss(self.class_weights, gamma) if loss_type == "focal" else None
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        if self.loss_type == "focal":
            loss = self.focal(logits, labels)
        elif self.loss_type == "weighted_ce" and self.class_weights is not None:
            loss = CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        else:
            loss = CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -------------------- Train/Eval routine -------------------
def train_and_eval(model_name, output_dir,
                   train_ds, val_ds,
                   max_length=256, lr=2e-5, epochs=3,
                   bsz_train=16, bsz_eval=32,
                   loss_type="ce", gamma=2.0, seed=42,
                   warmup_ratio=0.06, weight_decay=0.01,
                   early_stop=True):

    torch.manual_seed(seed); np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    tok_train = tokenize_for(model_name, train_ds, max_length=max_length)
    tok_val   = tokenize_for(model_name, val_ds,   max_length=max_length)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    if hasattr(model.config, "seq_classif_dropout"):
        model.config.seq_classif_dropout = 0.2

    class_weights = get_class_weights(train_ds) if loss_type in {"weighted_ce", "focal"} else None

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch" if early_stop else "no",
        save_total_limit=1,
        load_best_model_at_end=early_stop,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        learning_rate=lr,
        per_device_train_batch_size=bsz_train,
        per_device_eval_batch_size=bsz_eval,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,

        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if early_stop else None

    trainer = LossTrainer(
        model=model, args=args,
        train_dataset=tok_train, eval_dataset=tok_val,
        compute_metrics=compute_metrics,
        loss_type=loss_type, class_weights=class_weights, gamma=gamma,
        callbacks=callbacks
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # Predictions & reports
    preds = trainer.predict(tok_val)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)

    # Save classification report
    label_names = {0:"Normal", 1:"Harassment", 2:"Defamation", 3:"Misleading"}
    report_txt = classification_report(y_true, y_pred, digits=2, target_names=[label_names[i] for i in [0,1,2,3]])
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    pd.DataFrame(cm, index=[f"true_{label_names[i]}" for i in [0,1,2,3]],
                    columns=[f"pred_{label_names[i]}" for i in [0,1,2,3]]) \
      .to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    # Save model
    trainer.save_model(output_dir)

    print(f"\n✅ {model_name} :: len={max_length}, loss={loss_type}, lr={lr}, epochs={epochs}, seed={seed}")
    print(metrics)
    print(report_txt)
    return {"run_dir": output_dir, "seed": seed, "max_len": max_length, "loss": loss_type,
            "gamma": gamma, "lr": lr, "epochs": epochs,
            "accuracy": float(metrics.get("eval_accuracy", np.nan)),
            "f1_macro": float(metrics.get("eval_f1_macro", np.nan))}

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: text,label")
    ap.add_argument("--outdir", required=True, help="Base output directory")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"🔹 Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    df = pd.read_csv(args.csv)
    if not {"text","label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df[["text","label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Split
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, stratify=df["label"], random_state=args.seed
    )
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    # Define experiment grid
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join(args.outdir, f"distil_exps_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    exps = [
        dict(name="base_ce_len384_synt",      max_length=384, loss_type="ce",          lr=2e-5, epochs=4, seed=args.seed),
        dict(name="base_ce_len512_synt",      max_length=384, loss_type="ce",          lr=2e-5, epochs=4, seed=args.seed)
    ]
    
    length_grid = [128, 384, 512]

    def pick_bsz(L):
        # larger max_length → fewer examples per batch
        if L <= 192: return 32, 64
        if L <= 320: return 16, 32
        if L <= 384: return 12, 24
        return 8, 16

    for L in length_grid:
        btr, bev = pick_bsz(L)
        exps.append(dict(
            name=f"weighted_ce_len{L}",
            max_length=L,
            loss_type="ce",   # or "ce" to compare
            lr=2e-5,
            epochs=4,
            seed=args.seed,
            bsz_train=btr,
            bsz_eval=bev
        ))

    # Run
    all_rows = []
    for e in exps:
        outdir = os.path.join(base_dir, e["name"])
        row = train_and_eval(
            model_name=args.model,
            output_dir=outdir,
            train_ds=train_ds, val_ds=val_ds,
            max_length=e.get("max_length", 256),
            lr=e.get("lr", 2e-5),
            epochs=e.get("epochs", 3),
            bsz_train=16, bsz_eval=32,
            loss_type=e.get("loss_type", "ce"),
            gamma=e.get("gamma", 2.0),
            seed=e.get("seed", args.seed),
            warmup_ratio=0.06, weight_decay=0.01,
            early_stop=True
        )
        all_rows.append({**e, **row})

    # Save summary CSV
    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(base_dir, "summary.csv"), index=False)
    print("\n🏁 Finished. Summary written to:", os.path.join(base_dir, "summary.csv"))

if __name__ == "__main__":
    main()
