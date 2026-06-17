"""
scripts/train_intent_classifier.py
-----------------------------------
Fine-tune a DistilBERT binary classifier on deepset/prompt-injections
PLUS a curated set of benign coding-request examples (Phase 6.5 fix).

WHY THE CODING-REQUEST DATASET WAS ADDED
-----------------------------------------
Live demo testing found the original classifier (trained on
deepset/prompt-injections alone, 529 examples) BLOCKS ordinary coding
requests at 0.93-0.99 "injection" probability:

  "Write a function to reverse a string in Python"  -> 0.99 INJECTION
  "Can you write me a script that sorts a list?"     -> 0.93 INJECTION
  "Explain how a Python function works"              -> 0.99 INJECTION
  "Write a SQL query to find duplicate rows"          -> 0.99 INJECTION

This is a spurious correlation learned from a training set where
injection-labeled examples are often phrased as "write a script that
does X" (a common jailbreak template), with too few benign counter-
examples of the same surface pattern.

scripts/generate_benign_coding_dataset.py produces ~300 diverse benign
coding-request examples (many languages, many tasks, many phrasings --
including the EXACT phrasings that triggered the false positives above).
This script merges that file with deepset/prompt-injections before
training.

EVALUATION STRATEGY
--------------------
The two sources are split independently (80/20 each), then combined:
  train = train_deepset + train_coding
  eval  = eval_deepset  + eval_coding

This lets us report metrics on each eval subset SEPARATELY:
  - eval_deepset metrics  -> confirms no regression on the original
                              98.3% / F1 0.97-0.98 benchmark
  - eval_coding metrics   -> confirms the coding-request false positives
                              are actually fixed (should be ~100% BENIGN
                              on held-out coding examples)

Run this once. If injectionbench/datasets/benign/coding_requests.json
does not exist, run scripts/generate_benign_coding_dataset.py first.

Usage:
    python scripts/generate_benign_coding_dataset.py   # one-time, if not done
    python scripts/train_intent_classifier.py

Requirements (install with pip install promptgate-llm[intent]):
    transformers>=4.30.0
    datasets
    torch
    scikit-learn
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so promptgate imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models" / "intent_classifier"
CODING_DATASET_PATH = PROJECT_ROOT / "injectionbench" / "datasets" / "benign" / "coding_requests.json"


def main() -> None:
    # ── Dependency check ─────────────────────────────────────────────────────
    try:
        import torch
        from datasets import load_dataset
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        import numpy as np
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with:")
        print("  pip install transformers[torch] datasets scikit-learn accelerate")
        sys.exit(1)

    print("=" * 60)
    print("PromptGate — Intent Classifier Training (with coding-request fix)")
    print("=" * 60)

    # ── Load deepset/prompt-injections ──────────────────────────────────────
    print("\n[1/6] Loading deepset/prompt-injections from HuggingFace...")
    ds = load_dataset("deepset/prompt-injections")

    deepset_texts  = list(ds["train"]["text"])  + list(ds["test"]["text"])
    deepset_labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    print(f"      Total samples: {len(deepset_texts)}")
    print(f"      Injections:    {sum(deepset_labels)}")
    print(f"      Benign:        {len(deepset_labels) - sum(deepset_labels)}")

    train_deepset_texts, eval_deepset_texts, train_deepset_labels, eval_deepset_labels = train_test_split(
        deepset_texts, deepset_labels,
        test_size=0.20,
        random_state=42,
        stratify=deepset_labels,
    )
    print(f"      Train: {len(train_deepset_texts)} | Eval: {len(eval_deepset_texts)}")

    # ── Load coding-request benign dataset ──────────────────────────────────
    print("\n[2/6] Loading benign coding-request dataset...")
    if not CODING_DATASET_PATH.is_file():
        print(f"      NOT FOUND: {CODING_DATASET_PATH}")
        print("      Run scripts/generate_benign_coding_dataset.py first.")
        print("      Falling back to deepset-only training (original Phase 4 behaviour).")
        coding_texts: list[str] = []
        coding_labels: list[int] = []
    else:
        coding_data = json.loads(CODING_DATASET_PATH.read_text(encoding="utf-8"))
        coding_texts  = [ex["text"] for ex in coding_data]
        coding_labels = [ex["label"] for ex in coding_data]
        print(f"      Loaded {len(coding_texts)} benign coding-request examples")

    if coding_texts:
        train_coding_texts, eval_coding_texts, train_coding_labels, eval_coding_labels = train_test_split(
            coding_texts, coding_labels,
            test_size=0.20,
            random_state=42,
        )
        print(f"      Train: {len(train_coding_texts)} | Eval: {len(eval_coding_texts)}")
    else:
        train_coding_texts, eval_coding_texts = [], []
        train_coding_labels, eval_coding_labels = [], []

    # ── Combine ───────────────────────────────────────────────────────────────
    train_texts  = train_deepset_texts + train_coding_texts
    train_labels = train_deepset_labels + train_coding_labels
    # Combined eval set used for the Trainer's during-training metrics.
    # eval_deepset and eval_coding are also kept separately for the final
    # breakdown report below.
    eval_texts  = eval_deepset_texts + eval_coding_texts
    eval_labels = eval_deepset_labels + eval_coding_labels

    print(f"\n      Combined train: {len(train_texts)} | Combined eval: {len(eval_texts)}")

    # ── Tokenise ─────────────────────────────────────────────────────────────
    print("\n[3/6] Tokenising...")
    BASE_MODEL = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenise(texts: list[str]) -> dict:
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

    train_enc = tokenise(train_texts)
    eval_enc  = tokenise(eval_texts)

    class InjectionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: dict, labels: list[int]) -> None:
            self.encodings = encodings
            self.labels    = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = InjectionDataset(train_enc, train_labels)
    eval_dataset  = InjectionDataset(eval_enc,  eval_labels)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[4/6] Loading base model: {BASE_MODEL}")
    id2label = {0: "BENIGN", 1: "INJECTION"}
    label2id = {"BENIGN": 0, "INJECTION": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[5/6] Fine-tuning...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(
            labels, preds,
            labels=[0, 1],
            target_names=["BENIGN", "INJECTION"],
            output_dict=True,
            zero_division=0,
        )
        return {
            "f1_injection": report["INJECTION"]["f1-score"],
            "f1_macro":     report["macro avg"]["f1-score"],
            "accuracy":     report["accuracy"],
        }

    import inspect
    _trainer_params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_strategy_key = (
        "eval_strategy" if "eval_strategy" in _trainer_params else "evaluation_strategy"
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        **{_eval_strategy_key: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_injection",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Eval report (combined, as before) ───────────────────────────────────
    print("\n[6/6] Evaluation on held-out splits:")
    print("\n--- Combined eval set (deepset + coding-requests) ---")
    preds_output = trainer.predict(eval_dataset)
    preds        = np.argmax(preds_output.predictions, axis=-1)
    print(classification_report(
        eval_labels, preds,
        target_names=["BENIGN", "INJECTION"],
        zero_division=0,
    ))

    # ── Eval report (deepset subset only) — regression check ────────────────
    print("\n--- deepset/prompt-injections eval subset only (regression check) ---")
    deepset_enc = tokenise(eval_deepset_texts)
    deepset_eval_dataset = InjectionDataset(deepset_enc, eval_deepset_labels)
    deepset_preds_output = trainer.predict(deepset_eval_dataset)
    deepset_preds = np.argmax(deepset_preds_output.predictions, axis=-1)
    print(classification_report(
        eval_deepset_labels, deepset_preds,
        target_names=["BENIGN", "INJECTION"],
        zero_division=0,
    ))

    # ── Eval report (coding-requests subset only) — fix verification ────────
    if eval_coding_texts:
        print("\n--- Benign coding-request eval subset only (fix verification) ---")
        coding_enc = tokenise(eval_coding_texts)
        coding_eval_dataset = InjectionDataset(coding_enc, eval_coding_labels)
        coding_preds_output = trainer.predict(coding_eval_dataset)
        coding_preds = np.argmax(coding_preds_output.predictions, axis=-1)
        print(classification_report(
            eval_coding_labels, coding_preds,
            labels=[0, 1],
            target_names=["BENIGN", "INJECTION"],
            zero_division=0,
        ))
        n_correct = sum(1 for p in coding_preds if p == 0)
        print(f"\n      {n_correct}/{len(coding_preds)} held-out coding requests "
              f"correctly classified as BENIGN "
              f"({100 * n_correct / len(coding_preds):.1f}%)")
        if n_correct < len(coding_preds):
            print("      Examples STILL misclassified as INJECTION:")
            for text, label, pred in zip(eval_coding_texts, eval_coding_labels, coding_preds):
                if pred != label:
                    print(f"        - {text!r}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model to: {MODEL_DIR}")
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\nDone. IntentClassifier will load from:", MODEL_DIR)
    print("Next steps:")
    print("  1. python -m pytest tests/ -q")
    print("  2. python -m injectionbench run --source huggingface")
    print("  3. Re-run the 6 regression prompts from the live demo session")
    print("  4. If all good, re-upload to HF Hub (overwrite existing model)")


if __name__ == "__main__":
    main()