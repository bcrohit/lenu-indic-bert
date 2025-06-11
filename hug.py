"""
Uploads the trained model to hugging face spaces.
"""
from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

api = HfApi()

token = HfFolder.get_token()

repo_id = "rohitbc/lenu_IN-indic-bert"

api.create_repo(repo_id=repo_id, token=token)

model = AutoModelForSequenceClassification.from_pretrained(r"models\lenu_IN-indic-bert")
tokenizer = AutoTokenizer.from_pretrained(r'models\lenu_IN-indic-bert')

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
