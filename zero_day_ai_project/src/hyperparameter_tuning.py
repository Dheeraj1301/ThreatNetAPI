import optuna
import torch
from sklearn.metrics import f1_score
from src.model_gnn import GNNModel, train_model, evaluate_model

def objective(trial, graph_data):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = GNNModel(input_dim=graph_data.x.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    train_model(model, graph_data, device="cpu", epochs=10, lr=lr)
    preds, labels = evaluate_model(model, graph_data, device="cpu")
    return f1_score(labels, preds, average="macro")

def run_optuna_study(graph_data, subset=False):
    if subset:
        idx = torch.randperm(len(graph_data.x))[:500]
        graph_data.x = graph_data.x[idx]
        graph_data.y = graph_data.y[idx]
        # Remove edges that reference excluded nodes (not recommended for real use)
        graph_data.edge_index = graph_data.edge_index[:, torch.all(graph_data.edge_index < 500, dim=0)]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, graph_data), n_trials=20)
    print(f"🔍 Best Trial: {study.best_trial.params}")
    return study.best_trial.params
