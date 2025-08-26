import argparse, os, numpy as np, pandas as pd, torch, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import evaluate
from datasets import Dataset

# ───────────────────────────── helper ─────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_macro.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}

# ───────────────────────────── main ───────────────────────────────
def main(args):
    print("🔹 Loading CSV  …")
    df = pd.read_csv(args.csv).loc[:, ["text", "label"]]

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.20, stratify=df["label"], random_state=42
    )

    train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    val_ds   = Dataset.from_dict({"text": X_val.tolist(),   "label": y_val.tolist()})

    print("🔹 Tokenising …")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True, padding="max_length",
            max_length=128
        )

    train_ds = train_ds.map(tok, batched=True)
    val_ds   = val_ds.map(tok,   batched=True)

    train_ds.set_format(type="torch",
                        columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch",
                      columns=["input_ids", "attention_mask", "label"])

    print("🔹 Model init  …")
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased", num_labels=4
    )

    training_args = TrainingArguments(
        output_dir          = args.out,
        num_train_epochs    = args.epochs,
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size  = args.batch*2,
        evaluation_strategy = "epoch",
        save_strategy       = "no",
        learning_rate       = 2e-5,
        weight_decay        = 0.01,
        logging_steps       = 100,
        report_to="none"
    )

    trainer = Trainer(
        model          = model,
        args           = training_args,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        compute_metrics= compute_metrics
    )

    print("🔹 Training …")
    trainer.train()

    print("🔹 Evaluating …")
    logits = trainer.predict(val_ds).predictions
    preds  = np.argmax(logits, axis=1)
    print(classification_report(y_val, preds,
                                target_names=["Normal","Harassment","Defamation","Misleading"],
                                digits=2))
    cm = confusion_matrix(y_val, preds, labels=[0,1,2,3])
    print("Confusion-matrix:\n", cm)

    # optional heat-map
    try:
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap="Blues",
                    xticklabels=["Norm","Harass","Defam","Mislead"],
                    yticklabels=["Norm","Harass","Defam","Mislead"])
        plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print("⚠️ Could not plot heat-map:", e)

    print("🔹 Saving model + tokenizer to", args.out)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

# ───────────────────────────── entry - point ──────────────────────
if __name__ == "__main__":
    accuracy  = evaluate.load("accuracy")
    f1_macro  = evaluate.load("f1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True, help="Path to CSV with text,label columns")
    parser.add_argument("--out",   required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch",  type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    main(args)