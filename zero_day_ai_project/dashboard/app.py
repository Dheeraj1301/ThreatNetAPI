import streamlit as st
import torch
from src.utils import load_graph

st.title("Zero-Day API Vulnerability Risk Viewer")

graph = load_graph("data/processed/graph_data.pt")
st.write("Total Nodes:", graph.num_nodes)
st.write("Total Edges:", graph.num_edges)
