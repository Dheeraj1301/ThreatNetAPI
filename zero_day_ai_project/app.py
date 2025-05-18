import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page config and style with brown shades and off-white background
st.set_page_config(page_title="Zero Day AI Dashboard", layout="wide")

# Custom CSS for background and colors
st.markdown(
    """
    <style>
    /* Off-white background */
    .reportview-container {
        background-color: #FAF9F6;
    }
    /* Sidebar background */
    .sidebar .sidebar-content {
        background-color: #A67B5B;
        color: #fff;
    }
    /* Headers */
    h1, h2, h3 {
        color: #5C4033;
    }
    /* Buttons */
    button {
        background-color: #8B5E3C;
        color: white;
    }
    /* Text color */
    .css-18e3th9 {
        color: #5C4033;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🔐 Zero Day AI Project Dashboard")

# Sidebar inputs for hyperparameters (dummy example)
st.sidebar.header("Hyperparameters")
hidden_dim = st.sidebar.selectbox("Hidden Dimension", [32, 64, 128], index=1)
dropout = st.sidebar.slider("Dropout Rate", 0.2, 0.5, 0.3)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=0.001, format="%.5f")
epochs = st.sidebar.slider("Epochs", 5, 100, 20)

# Dummy dataset info section
st.subheader("Dataset Overview")
num_nodes = 1000
num_edges = 2500
num_features = 128
num_classes = 4

st.markdown(f"""
- Number of Nodes: **{num_nodes}**  
- Number of Edges: **{num_edges}**  
- Feature Dimensions: **{num_features}**  
- Number of Classes: **{num_classes}**
""")

# Dummy training loss plot
st.subheader("Training Loss Curve")
epochs_range = np.arange(1, epochs + 1)
loss_values = np.exp(-0.1 * epochs_range) + 0.05 * np.random.rand(epochs)  # Fake exponential decay + noise

fig, ax = plt.subplots()
ax.plot(epochs_range, loss_values, color="#8B5E3C", marker='o')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig)

# Dummy performance metrics
st.subheader("Model Performance")
metrics = {
    "Accuracy": 0.85,
    "Precision": 0.82,
    "Recall": 0.80,
    "F1 Score": 0.81,
}

for metric, val in metrics.items():
    st.write(f"**{metric}:** {val:.2f}")

# Dummy predictions vs labels table
st.subheader("Sample Predictions vs Labels")
data = {
    "Predictions": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "LOW"],
    "Actual Labels": ["LOW", "MEDIUM", "MEDIUM", "CRITICAL", "LOW"]
}
df_preds = pd.DataFrame(data)
st.table(df_preds)

# Add a button to simulate training start (for UX)
if st.button("Start Training"):
    st.info("Training started with the selected hyperparameters.")
    # Here you can hook in actual training code and update the dashboard dynamically

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;color:#5C4033;">
    &copy; 2025 Zero Day AI Project | Developed by You
    </p>
    """,
    unsafe_allow_html=True
)
