# ThreatNetAPI

## Overview

**ThreatNetAPI** is an AI-powered API for automated vulnerability detection leveraging Graph Neural Networks (GNNs). Designed to analyze complex relationships in software or network data, ThreatNetAPI models security vulnerabilities as graph data, enabling highly accurate identification of potential threats — including zero-day vulnerabilities.

The system ingests raw data and embeddings, builds graph representations, and uses advanced GNN architectures to classify vulnerability severity levels. It exposes a RESTful API allowing seamless integration with security tools, dashboards, and workflows.

---

## Features

- **Graph-based vulnerability modeling:** Converts security-related data into graph structures to capture complex interdependencies.
- **State-of-the-art GNN:** Utilizes multiple GCN layers to effectively learn node representations for classification.
- **Automated hyperparameter tuning:** Uses Optuna to optimize model performance with minimal manual intervention.
- **Efficient training pipeline:** Includes batching, dropout, and Adam optimizer for fast, stable convergence.
- **RESTful API interface:** Provides easy endpoints for uploading data, running inference, and retrieving results.
- **Dashboard UI:** Intuitive web interface with brown-themed design for visualizing model predictions and system status.
- **Detailed logging and error handling:** Ensures robustness and easier troubleshooting.

---

## Installation & Setup

1. Clone the repo:  
git clone ```https://github.com/your-username/ThreatNetAPI.git```
```cd ThreatNetAPI```


2. Create and activate a virtual environment:  
```python -m venv venv```
```source venv/bin/activate # Linux/Mac```
```venv\Scripts\activate # Windows```


3. Install dependencies:  
```pip install -r requirements.txt```


4. Run initial data processing and model training:  
```python main.py```



5. Start the API server:  
```uvicorn api_server:app --reload```


---

## Project Structure
```
- `main.py` — Entry point: orchestrates data loading, graph building, model training, and hyperparameter tuning.  
- `src/graph_builder.py` — Constructs graph data from raw inputs and embeddings.  
- `src/model_gnn.py` — Defines GNN model architecture and training routines.  
- `src/hyperparameter_tuning.py` — Implements Optuna-based automated hyperparameter optimization.  
- `api_server.py` — FastAPI server exposing endpoints for prediction and monitoring.  
- `dashboard/` — Frontend code for the brown-themed UI dashboard.  
- `requirements.txt` — Python dependencies.  
```
---

## Usage

### Individual Component Testing

- **Graph Builder**  
Test graph construction with a sample dataset:  
```python
from src.graph_builder import build_graph_data  
graph = build_graph_data(sample_df, sample_embeddings)  
print(graph)
```
Model Training

Run standalone training on a small dataset to verify convergence:

```
from src.model_gnn import GNNModel, train_model  
model = GNNModel(input_dim=embedding_dim)  
train_model(model, graph_data, device="cpu", epochs=10, lr=0.01)
```
Hyperparameter Tuning

Run Optuna trials and verify improved metrics:
```
python
Copy
Edit
from src.hyperparameter_tuning import run_optuna_study  
best_params = run_optuna_study(graph_data)
print(best_params)
```
API Endpoints
Use curl or Postman to send JSON payloads to the API and check responses. Example:
```
bash
Copy
Edit
POST /predict  
Content-Type: application/json  
{ "data": ... }
```
Dashboard
Run the dashboard frontend and verify UI components load with the brown/off-white color scheme.

Final Pipeline Testing
Load real or synthetic data, build graph representation.

Use optimized hyperparameters from Optuna to train the model fully.

Evaluate model performance on validation/test data: accuracy, F1-score, confusion matrix.

Deploy trained model in the API backend for inference.

Send test requests to API and verify correct predictions.

View real-time predictions and analytics on the dashboard UI.

Effectiveness & Performance
Accuracy: Achieves high classification accuracy due to graph-based relational learning.

Robustness: Model handles various types of vulnerabilities, including unseen patterns (zero-day).

Speed: Uses GPU acceleration and efficient data loaders for fast training and inference.

Automated Tuning: Optuna tuning reduces manual hyperparameter search time drastically.

API Response: Lightweight FastAPI backend ensures low-latency predictions suitable for integration.

Optimizations
Graph Sampling: For large datasets, graph sampling methods reduce memory and computation overhead.

Early Stopping: Stops training when validation loss stagnates, saving time and avoiding overfitting.

Batching: Mini-batch training to handle large graphs effectively.

Dropout Regularization: Prevents overfitting in GNN layers.

Learning Rate Scheduling: Dynamic adjustment of learning rate improves convergence speed.

Parallel API Serving: Deploy with Uvicorn/Gunicorn with multiple workers for scalable inference.

Troubleshooting & FAQs
Error: "Found unmapped severity labels: ['UNKNOWN']"
Ensure all severity labels in your dataset are mapped to known classes before graph building.
Update preprocessing to handle or filter unknown labels.

Error: "Target is out of bounds" during training
Check your target labels are valid class indices [0, num_classes-1] and properly encoded.

Model fails to train or diverges
Lower learning rate, increase dropout, or check data quality. Use Optuna tuning to find better hyperparameters.

API not responding
Check if the FastAPI server is running and accessible on the correct port. Inspect logs for exceptions.

Dashboard UI doesn’t load or is blank
Verify static files are served correctly and dependencies (React/JS) are installed if applicable.

Future Improvements
Integrate more advanced GNN models (e.g., GraphSAGE, GAT) for improved representation learning.

Add real-time vulnerability alerting and automated patch recommendations.

Enhance API security with authentication and rate limiting.

Build CI/CD pipelines for automated model retraining and deployment.

Support multi-modal inputs including text, network logs, and binary code analysis.

Contact & Support
For questions or contributions, please open issues or pull requests on the GitHub repository:
https://github.com/dheeraj1301/ThreatNetAPI

Thank you for using ThreatNetAPI — powering the next generation of intelligent vulnerability detection.
```
yaml
Copy
Edit

---

Would you like me to help generate example test commands or a sample Postman collection for the API?
```









Search

Reason

Deep research

Create image




