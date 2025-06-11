# Entity Legal Form (ELF) Classifier - IN Jurisdiction

This project implements a transformer-based NLP model to classify the **Entity Legal Form (ELF)** of organizations based on their names. It replicates GLEIF's ELF assignment initiative, with a focus on Indian entities.

## Project

To build and fine-tune a machine learning model that can automatically assign the correct ELF code to an entity name using only its textual features — specifically tailored for Indian legal entities.

## Dependencies
- python (>=3.8, <3.10)
- [scikit-learn](https://scikit-learn.org/) - Provides Machine Learning functionality for token based modelling
- [transformers](https://huggingface.co/docs/transformers/index) - Download and applying Neural Network Models
- [pytorch](https://pytorch.org/) - Machine Learning Framework to train Neural Network Models
- [pandas](https://pandas.pydata.org/) - For reading and handling data
- [Typer](https://typer.tiangolo.com/) - Adds the command line interface
- [requests](https://docs.python-requests.org/en/latest/) and [pydantic](https://pydantic-docs.helpmanual.io/) - For downloading LEI data from GLEIF's website

## Dataset

- Source: [GLEIF Golden Copy](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-concatenated-file)
- Filtered to include only entities with **Jurisdiction = IN** (India)
- Required columns:
  - `Entity Legal Name`
  - `Entity Legal Form Code (ELF)`

## Model

- Base model: [`ai4bharat/indic-bert`](https://huggingface.co/ai4bharat/indic-bert)
- Task: Multi-class classification using entity names
- Training framework: Hugging Face Transformers + PyTorch
- Dataset split: Train / Validation (80/20)

## Training

- Batch size: 16
- Epochs: 3
- Optimizer: AdamW
- Scheduler: Linear learning rate decay
- Evaluation metric: Accuracy
- Hardware: Kaggle Notebook with Tesla P100 GPU

## Results

After fine-tuning, the model achieved **Validation Accuracy** of *88%* compared to the base model performance of *82%*


## Usage flow

   ```bash
   pip install -r requirements
   python get_data.py
   python preprocess.py
   python train.py
   python hug.py  
