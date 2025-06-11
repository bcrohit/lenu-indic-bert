import json
import streamlit as st
import pandas as pd
from transformers import pipeline


# Custom model wrapper
class ELFDetectionModel:
    def __init__(self, pipeline, id2label):
        self.pipeline = pipeline
        self.id2label = id2label

    def detect(self, legal_name, top=3):
        elf_probabilities = pd.Series(
            {
                self.id2label.get(res['label'][6:]): res['score']
                for res in self.pipeline(legal_name, top_k=top)
            }
        )
        return elf_probabilities.sort_values(ascending=False)

# Get label to id mapping
with open(r"assets\labels_ids.json") as fp:
    labe_ids = json.load(fp)
    id2label = labe_ids['id2label']

# Load model from Hugging Face
@st.cache_resource
def get_model():
    pipe = pipeline(model="rohitbc/lenu_IN-indic-bert")
    return ELFDetectionModel(pipe, id2label)

model = get_model()

# Streamlit UI
st.title("ELF Detector")

user_input = st.text_input("Enter a legal name:")

if user_input:
    st.write(f"Predictions for: **{user_input}**")
    predictions = model.detect(user_input)
    st.dataframe(predictions.rename("Score"))