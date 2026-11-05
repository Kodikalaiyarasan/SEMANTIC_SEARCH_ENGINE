"""
train_fine_tune.py
==================
Fine-tuning script for SentenceTransformer using MultipleNegativesRankingLoss
on the 100,000 Python Anchor-Positive dataset.

Pre-requisites:
    pip install sentence-transformers datasets torch accelerate

Usage:
    python train_fine_tune.py
"""

import json
from pathlib import Path
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "python_code_search_100k.json"
BASE_MODEL_ID = "Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2-tech-code-search-v3"

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
EVAL_SPLIT_RATIO = 0.05  # 5% for evaluation (~5,000 pairs)


def load_pairs(filepath: Path):
    print(f"Loading dataset from: {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} total pairs.")
    return data


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Please run scripts/generate_100k_dataset.py first."
        )

    # 1. Load data
    raw_data = load_pairs(DATASET_PATH)

    # Format into datasets format
    anchors = [item["anchor"] for item in raw_data]
    positives = [item["positive"] for item in raw_data]

    hf_dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives
    })

    # Train / Test split
    split_dataset = hf_dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train split: {len(train_dataset):,} pairs")
    print(f"Eval split:  {len(eval_dataset):,} pairs")

    # 2. Load Base Model
    print(f"\nLoading base model: {BASE_MODEL_ID}...")
    model = SentenceTransformer(BASE_MODEL_ID)

    # 3. Define Loss Function
    # MultipleNegativesRankingLoss treats other elements in the batch as negative candidates
    loss = losses.MultipleNegativesRankingLoss(model)

    # 4. Training Arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        fp16=True,  # Set to False if running on CPU
    )

    # 5. Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    # 6. Train
    print("\nStarting model fine-tuning...")
    trainer.train()

    # 7. Save Final Model
    print(f"\nSaving fine-tuned model to: {OUTPUT_MODEL_DIR}...")
    model.save_pretrained(str(OUTPUT_MODEL_DIR))
    print("[SUCCESS] Training completed and model saved successfully!")


if __name__ == "__main__":
    main()

