from src.data_ingestion import load_nvd_cve_data
from src.preprocessing import extract_descriptions, get_tfidf_features
from src.graph_builder import build_graph
from src.utils import save_graph

def main():
    raw_data = load_nvd_cve_data("data/raw/nvdcve-1.1-2024.json.gz")
    descriptions = extract_descriptions(raw_data)
    features, _ = get_tfidf_features(descriptions)
    
    graph = build_graph(raw_data, features)
    save_graph(graph, "data/processed/graph_data.pt")
    print("Graph saved. Ready for training.")

if __name__ == "__main__":
    main()
