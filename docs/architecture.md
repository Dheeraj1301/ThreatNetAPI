# ThreatNetAPI — Architecture and Experiment Boundaries

ThreatNetAPI is presently a **CVE severity classification experiment**, not a live vulnerability scanner or deployed inference service. It reads a historic NVD feed and uses text similarity to build a graph for learning severity labels.

## Training flow

```mermaid
flowchart TD
    A[("data/raw/nvdcve-1.1-2024.json.gz")] --> B["data_ingestion.load_nvd_cve_data"]
    B --> C["preprocessing.process_cve_data"]
    C --> D["Extract CVE ID, description, CVSS v3 severity"]
    D --> E["Stratified train/test split"]
    E --> F["embedding.BERTEncoder / cached .pt"]
    F --> G["graph_builder.build_graph_data"]
    G --> H["Cosine similarity > 0.9 defines graph edges"]
    H --> I["GCNConv → ReLU/Dropout → GCNConv → classifier"]
    I --> J["Optuna: hidden_dim, dropout, learning rate"]
    J --> K["Train model, save checkpoint"]
    K --> L["Current-graph classification report"]
```

## Experimental UI separation

```mermaid
flowchart LR
    A["Training script: main.py"] --> B[("Saved model checkpoints")]
    C["Streamlit app.py"] --> D["Placeholder dataset summary, loss curve and metrics"]
    B -.->|"not wired"| D
    E["Public HTTP inference endpoint"] -.->|"not yet implemented"| B
```

## Module boundaries

| Module | Role |
| --- | --- |
| `src/data_ingestion.py` | Reads NVD JSON 1.1 from a compressed file |
| `src/preprocessing.py` | Creates a DataFrame of descriptions and severity classes; performs a split |
| `src/embedding.py` | Downloads/uses BERT and computes/caches dense text embeddings |
| `src/graph_builder.py` | Produces PyTorch Geometric feature, edge and label tensors |
| `src/model_gnn.py` | Trains a two-layer graph convolutional classifier |
| `src/hyperparameter_tuning.py` | Uses Optuna to search training-graph macro-F1 |
| `app.py` | Standalone illustrative dashboard rather than integrated predictions |

## Accuracy and reproducibility warnings

- The held-out test DataFrame created in `main.py` is **not evaluated**; training and scoring reuse the graph. Do not claim test accuracy or zero-day detection.
- Filtering unknown labels after computing embeddings can misalign nodes and labels. Verify that rows are reindexed consistently.
- The edge-construction code may produce an empty graph and requires handling for such cases.
- The Streamlit plots are procedurally generated demo content; they do not reflect the GCN's measured learning curves.
- The dependency manifest is incomplete and the NVD 1.1 feed schema is historical. For reliable replication, pin compatible environments, record seed/data provenance and publish a separate, properly held-out evaluation.

[Return to README](../README.md).
