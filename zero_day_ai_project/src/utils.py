import torch
from pathlib import Path

def save_graph(graph, path="zero_day_ai_project/data/processed/graph_data.pt"):
    """
    Save the graph object to disk.

    Args:
        graph: PyTorch or PyTorch Geometric graph object.
        path (str or Path): Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    torch.save(graph, path)
    print(f"✅ Graph saved to: {path}")


def load_graph(path="zero_day_ai_project/data/processed/graph_data.pt"):
    """
    Load the graph object from disk.

    Args:
        path (str or Path): Path to the saved graph file.

    Returns:
        The loaded graph object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found at: {path}")
    graph = torch.load(path)
    print(f"📦 Graph loaded from: {path}")
    return graph


# Optional test code block
if __name__ == "__main__":
    # Dummy test: Only works if you've saved a graph earlier
    try:
        graph = load_graph()
        print("🔹 Graph structure:", graph)
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
