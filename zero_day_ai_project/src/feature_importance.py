# src/feature_importance.py
import torch
from torch_geometric.nn import GNNExplainer
import matplotlib.pyplot as plt

def explain_gnn_model(model, data, node_idx=0):
    """
    Use GNNExplainer to explain model predictions for a node.
    """
    explainer = GNNExplainer(model, epochs=200)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)

    explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y)
    plt.show()
