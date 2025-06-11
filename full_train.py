"""
Script to train ai4bharath\indic-bert (bert-based) to classify LFC with custom training loop.
"""

import os
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
from torch.utils.data import DataLoader

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

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=32, collate_fn=data_collator
    )
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=32, collate_fn=data_collator
    )

# Training Loop
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(label_names))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
warmup_steps = int(0.1 * num_training_steps)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)
print("Number of training steps: ", num_training_steps)

# Initialize progress bar with metrics tracking
progress_bar = tqdm(
    total=num_training_steps,  # Total number of steps
    desc="Training",  # Description prefix
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"  # Custom format
)


model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0  # Track cumulative loss per epoch
    
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Update metrics
        current_lr = lr_scheduler.get_last_lr()[0]  # Get current learning rate
        step_loss = loss.item()  # Current step loss
        epoch_loss += step_loss
        
        # Update progress bar metrics
        progress_bar.set_postfix({
            "loss": f"{step_loss:.4f}",  # Current step loss
            "avg_loss": f"{(epoch_loss/(step+1)):.4f}",  # Running epoch average
            "lr": f"{current_lr:.2e}",  # Learning rate
            "epoch": f"{epoch+1}/{num_epochs}"  # Epoch progress
        })
        progress_bar.update(1)

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")

progress_bar.close()


metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

results = metric.compute()

print("Evaluation Accuracy: ", results['accuracy'])