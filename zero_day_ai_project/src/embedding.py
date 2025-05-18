# src/embedding.py
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class BERTEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def encode(self, texts, max_length=128):
        self.model.eval()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

def save_embeddings(embeddings, path):
    torch.save(torch.tensor(embeddings), path)

def load_embeddings(path):
    return torch.load(path)
