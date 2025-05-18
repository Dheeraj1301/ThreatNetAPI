# src/model_gnn.py
import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, 4)
        self.dropout = Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc(x)

def train_model(model, data, device, epochs=50, lr=0.001):
    model.train()
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

def evaluate_model(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
    return preds, labels
