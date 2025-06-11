"""
Script to train ai4bharath\indic-bert (bert-based) to classify LFC using transformer trainer API.
"""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from transformers import Trainer

import evaluate
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import DataCollatorWithPadding
from transformers import get_scheduler
from sklearn.model_selection import train_test_split

# Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "ai4bharat/indic-bert"

data = pd.read_csv(r"data\jur_data+proc.csv")


# Prepare data for training

# Split
train_val_df, test_df = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df['label'], random_state=42)

label_names = sorted(data['label'].unique().tolist())

# Define HF Features schema
features = Features({
    "name": Value("string"),
    "label": ClassLabel(names=[str(label) for label in label_names]),
    "__index_level_0__": Value(dtype='int32', id=None)
})

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df, features=features, preserve_index=None)
val_ds = Dataset.from_pandas(val_df, features=features, preserve_index=None)
test_ds = Dataset.from_pandas(test_df, features=features, preserve_index=None)

# Final DatasetDict
raw_datasets = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    'test': test_ds
})

# Tokenization
def tokenize_function(record):
    return tokenizer(record["name"], truncation=True)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["name", "__index_level_0__"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(label_names))
model.to(device)

training_args = TrainingArguments(
        output_dir = 'lenu_IN',
        eval_strategy = "epoch",
        save_strategy = "epoch",
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        learning_rate = 2e-4,
        optim = "adamw_torch",
        num_train_epochs = 2,
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        run_name = "lenu_indicbert",
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy",
        report_to = "wandb"
    )

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()