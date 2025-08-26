
# -*- coding: utf-8 -*-
"""
A complete script to run a series of comparative experiments fine-tuning transformer
models for multi-class legal risk classification.

This script compares a general-purpose model (DistilBERT) against a domain-specific
model (Legal-BERT) across various training configurations, including different
sequence lengths and loss functions.
"""
import os
import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from collections import Counter

# --- Metrics ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Computes accuracy and macro F1-score from model predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# --- Tokenization ---
def tokenize_for(model_name, ds: Dataset, max_length=256):
    """Tokenizes a dataset for a given model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    cols_to_remove = [c for c in ds.column_names if c not in ("text", "label")]
    return ds.map(_tokenize, batched=True, remove_columns=cols_to_remove).with_format("torch")

# --- Custom Loss Functions ---
def get_class_weights(ds, num_labels=4):
    """Computes class weights for handling imbalanced datasets."""
    counts = Counter(ds["label"])
    total = sum(counts.values())
    freqs = np.array([counts.get(i, 0) for i in range(num_labels)], dtype=np.float32)
    weights = total / (num_labels * np.maximum(freqs, 1))
    return torch.tensor(weights, dtype=torch.float)

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class LossTrainer(Trainer):
    def __init__(self, loss_type="ce", class_weights=None, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.focal_loss_fn = FocalLoss(self.class_weights, gamma) if loss_type == "focal" else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.loss_type == "focal":
            loss = self.focal_loss_fn(logits, labels)
        elif self.loss_type == "weighted_ce" and self.class_weights is not None:
            loss = CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        else:
            loss = CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

# --- Main Experiment Function ---
def run_experiment(config, train_ds, val_ds):
    """Main function to run a single training and evaluation pipeline."""
    output_dir = config['output_dir']
    model_name = config['model_name']
    
    print("\n" + "="*80)
    print(f"🚀 STARTING EXPERIMENT: {os.path.basename(output_dir)}")
    print(f"    Config: {config}")
    print("="*80 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Tokenize datasets
    tok_train = tokenize_for(model_name, train_ds, max_length=config.get('max_length', 128))
    tok_val = tokenize_for(model_name, val_ds, max_length=config.get('max_length', 128))

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    if hasattr(model.config, "seq_classif_dropout"):
        model.config.seq_classif_dropout = 0.2

    # Configure training
    loss_type = config.get('loss_type', 'ce')
    class_weights = get_class_weights(train_ds) if loss_type in {"weighted_ce", "focal"} else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.get('lr', 2e-5),
        per_device_train_batch_size=config.get('batch', 16),
        per_device_eval_batch_size=config.get('batch', 16) * 2,
        num_train_epochs=config.get('epochs', 3),
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = LossTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        compute_metrics=compute_metrics,
        loss_type=loss_type,
        class_weights=class_weights,
        gamma=config.get('gamma', 2.0),
        callbacks=callbacks,
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()
    
    print(f"\n✅ FINAL METRICS for {os.path.basename(output_dir)}:")
    print(metrics)

    # Final report and save
    preds = trainer.predict(tok_val)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)
    
    label_names = ["Normal", "Harassment", "Defamation", "Misleading"]
    report = classification_report(y_true, y_pred, target_names=label_names, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    print("\nPer-class report on validation set:\n", report)
    print("Confusion matrix:\n", cm)

    # Save results to a text file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"Experiment Configuration:\n{config}\n\n")
        f.write(f"Final Validation Metrics:\n{metrics}\n\n")
        f.write("Per-class Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print(f"🔹 Saving final model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✨ Experiment {os.path.basename(output_dir)} Done.")

if __name__ == "__main__":
    # --- 1. Load and Split Data (once for all experiments) ---
    CSV_PATH = "data/defamation_synthetic_97.csv" # <--- IMPORTANT: SET YOUR FILE PATH HERE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Data file not found at {CSV_PATH}. Please update the path.")

    print(f"🔹 Loading and splitting data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    print("Data ready.")

    # --- 2. Define All Experiment Configurations ---
    experiments = [
        # --- DistilBERT Experiments ---
        
        # --- Legal-BERT Experiments ---
        {'model_name': 'nlpaueb/legal-bert-base-uncased', 'output_dir': 'results/legalbert_base', 'max_length': 512, 'epochs': 4, 'loss_type': 'ce'},
        {'model_name': 'nlpaueb/legal-bert-base-uncased', 'output_dir': 'results/legalbert_len320_wce', 'max_length': 320, 'epochs': 4, 'loss_type': 'weighted_ce'},
        {'model_name': 'nlpaueb/legal-bert-base-uncased', 'output_dir': 'results/legalbert_len320_focal', 'max_length': 320, 'epochs': 4, 'loss_type': 'focal'},
    ]

    # --- 3. Run All Experiments ---
    for config in experiments:
        run_experiment(config, train_ds, val_ds)

    print("\n\n🎉 All experiments complete.")


# import argparse, os, numpy as np, pandas as pd, torch, seaborn as sns, matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
#                           TrainingArguments, Trainer)
# import evaluate
# from datasets import Dataset

# # ───────────────────────────── helper ─────────────────────────────
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
#     f1  = f1_macro.compute(predictions=preds, references=labels, average="macro")["f1"]
#     return {"accuracy": acc, "f1_macro": f1}

# # ───────────────────────────── main ───────────────────────────────
# def main(args):
#     print("🔹 Loading CSV  …")
#     df = pd.read_csv(args.csv).loc[:, ["text", "label"]]

#     X_train, X_val, y_train, y_val = train_test_split(
#         df["text"], df["label"], test_size=0.20, stratify=df["label"], random_state=42
#     )

#     train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
#     val_ds   = Dataset.from_dict({"text": X_val.tolist(),   "label": y_val.tolist()})

#     print("🔹 Tokenising …")
#     tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

#     def tok(batch):
#         return tokenizer(
#             batch["text"],
#             truncation=True, padding="max_length",
#             max_length=128
#         )

#     train_ds = train_ds.map(tok, batched=True)
#     val_ds   = val_ds.map(tok,   batched=True)

#     train_ds.set_format(type="torch",
#                         columns=["input_ids", "attention_mask", "label"])
#     val_ds.set_format(type="torch",
#                       columns=["input_ids", "attention_mask", "label"])

#     print("🔹 Model init  …")
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "nlpaueb/legal-bert-base-uncased", num_labels=4
#     )

#     training_args = TrainingArguments(
#         output_dir          = args.out,
#         num_train_epochs    = args.epochs,
#         per_device_train_batch_size = args.batch,
#         per_device_eval_batch_size  = args.batch*2,
#         evaluation_strategy = "epoch",
#         save_strategy       = "no",
#         learning_rate       = 2e-5,
#         weight_decay        = 0.01,
#         logging_steps       = 100,
#         report_to="none"
#     )

#     trainer = Trainer(
#         model          = model,
#         args           = training_args,
#         train_dataset  = train_ds,
#         eval_dataset   = val_ds,
#         compute_metrics= compute_metrics
#     )

#     print("🔹 Training …")
#     trainer.train()

#     print("🔹 Evaluating …")
#     logits = trainer.predict(val_ds).predictions
#     preds  = np.argmax(logits, axis=1)
#     print(classification_report(y_val, preds,
#                                 target_names=["Normal","Harassment","Defamation","Misleading"],
#                                 digits=2))
#     cm = confusion_matrix(y_val, preds, labels=[0,1,2,3])
#     print("Confusion-matrix:\n", cm)

#     # optional heat-map
#     try:
#         sns.heatmap(cm, annot=True, fmt='d',
#                     cmap="Blues",
#                     xticklabels=["Norm","Harass","Defam","Mislead"],
#                     yticklabels=["Norm","Harass","Defam","Mislead"])
#         plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
#         plt.tight_layout(); plt.savefig(os.path.join(args.out, "confusion_matrix.png"))
#         plt.close()
#     except Exception as e:
#         print("⚠️ Could not plot heat-map:", e)

#     print("🔹 Saving model + tokenizer to", args.out)
#     model.save_pretrained(args.out)
#     tokenizer.save_pretrained(args.out)

# # ───────────────────────────── entry - point ──────────────────────
# if __name__ == "__main__":
#     accuracy  = evaluate.load("accuracy")
#     f1_macro  = evaluate.load("f1")

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv",   required=True, help="Path to CSV with text,label columns")
#     parser.add_argument("--out",   required=True, help="Output directory for model")
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--batch",  type=int, default=16)
#     args = parser.parse_args()

#     os.makedirs(args.out, exist_ok=True)
#     main(args)