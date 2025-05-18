# src/graph_builder.py
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_graph_data(df, embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    edge_index = []
    threshold = 0.9  # Can be tuned

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i][j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index).t().contiguous()
    x = torch.tensor(embeddings, dtype=torch.float)
    y = torch.tensor(df['severity'].map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}).values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)
