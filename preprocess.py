"""
Performs basic data processing operations and creates a json mapping label names to ids.
Presumes the extracted jurisdiction data is present in data folder with naming `jurisdiction_data.csv`
"""

import json
import os
import re

import numpy as np
import pandas as pd

data_dir = "data"
path = os.path.join(data_dir, "jurisdiction_data.csv")
df = pd.read_csv(path)

# Some column names as variables for quick reuse
LFC = 'Entity.LegalForm.EntityLegalFormCode'
LEN = 'Entity.LegalName'


# Ignore less then 100
vals, counts = np.unique(df[LFC], return_counts=True)
values_to_remove = vals[counts<100]
df = df[~df[LFC].isin(values_to_remove)]
df.drop_duplicates(subset=['Entity.LegalName'], inplace=True)

data = df[[LEN, LFC]].copy()

# Clean names
def preprocess_name(name: str) -> str:
    name = name.title()  # Case normalization
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with single
    name = re.sub(r'[^\w\s\.\-&]', '', name)  # Remove special characters except ".", "-", "&"
    return name

data.loc[:, 'name'] = data[LEN].apply(preprocess_name)

# Logic to map labels to ids and vice versa
label_names = sorted(data[LFC].unique().tolist())
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}
data.loc[:, 'label'] = data[LFC].map(label2id)

data.drop([LFC, LEN], axis=1, inplace=True)
data = data.reset_index(drop=True)

# Save files
data.to_csv(os.path.join(data_dir, "jur_data_proc.csv"))

with open(r"assets\labels_ids.json", "w") as fp:
    j = {
        "label2id": label2id,
        "id2label": id2label
    }
    json.dump(j, fp, indent=4)