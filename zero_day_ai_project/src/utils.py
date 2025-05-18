# src/utils.py
import torch
import os
from datetime import datetime

def save_model_versioned(model, model_dir="results/model_logs"):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"gnn_model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
