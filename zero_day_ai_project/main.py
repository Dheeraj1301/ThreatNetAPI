# File: zero_day_ai_project/main.py

import os
import time
import torch
from sklearn.metrics import classification_report

from src.data_ingestion import load_nvd_cve_data
from src.preprocessing import process_cve_data, stratified_train_test_split
from src.embedding import BERTEncoder, save_embeddings, load_embeddings
from src.graph_builder import build_graph_data
from src.model_gnn import GNNModel, train_model, evaluate_model
from src.utils import save_model_versioned
from src.hyperparameter_tuning import run_optuna_study

# Track total time
start_total = time.time()

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# Step 1: Load Raw Data (limit to 1000 for quick test)
start = time.time()
cve_items = load_nvd_cve_data("data/raw/nvdcve-1.1-2024.json.gz")
cve_items = cve_items[:1000]
print(f"Step 1 done: Loaded {len(cve_items)} CVE items in {time.time() - start:.2f} sec")

# Step 2: Preprocess and Label
start = time.time()
df = process_cve_data(cve_items)
print(f"Step 2 done: Processed data into DataFrame with {df.shape[0]} rows in {time.time() - start:.2f} sec")

# Step 3: Train/Test Split (Stratified)
start = time.time()
train_df, test_df = stratified_train_test_split(df)
print(f"Step 3 done: Stratified split with {len(train_df)} train and {len(test_df)} test rows in {time.time() - start:.2f} sec")

# Step 4: BERT Embedding for Descriptions
start = time.time()
embedding_path = "data/processed/train_embeddings.pt"
if os.path.exists(embedding_path):
    print("📦 Loading cached BERT embeddings...")
    train_embeddings = load_embeddings(embedding_path)
else:
    print("⚙️ Encoding descriptions using BERT...")
    encoder = BERTEncoder()
    train_embeddings = encoder.encode(train_df['description'].tolist())
    save_embeddings(train_embeddings, embedding_path)
print(f"Step 4 done: BERT embeddings completed in {time.time() - start:.2f} sec")

# Step 5: Graph Building
start = time.time()
graph_data = build_graph_data(train_df, train_embeddings)
print(f"Step 5 done: Graph built in {time.time() - start:.2f} sec")

# Step 6: Hyperparameter Tuning (on subset)
start = time.time()
print("🎯 Running Optuna tuning on a small subset...")
best_params = run_optuna_study(graph_data, subset=True)
print(f"Step 6 done: Hyperparameter tuning finished in {time.time() - start:.2f} sec")

# Step 7: Train Final Model
start = time.time()
print("🚀 Training final GNN model...")
# Extract lr and model params separately
lr = best_params.pop('lr', 0.001)  # Default lr=0.001 if not found

model = GNNModel(input_dim=train_embeddings.shape[1], **best_params).to(DEVICE)

# Then pass lr to train_model
train_model(model, graph_data, device=DEVICE, epochs=10, lr=lr)
save_model_versioned(model, model_dir="results/model_logs")
print(f"Step 7 done: Final model trained in {time.time() - start:.2f} sec")

# Step 8: Evaluate
print("🧪 Evaluating model...")
preds, labels = evaluate_model(model, graph_data, device=DEVICE)
print(classification_report(labels, preds))

print(f"🏁 Project completed in {time.time() - start_total:.2f} sec")
