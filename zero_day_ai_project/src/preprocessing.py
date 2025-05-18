# src/preprocessing.py
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np

def stratified_train_test_split(df, stratify_col='severity', test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    return train_df, test_df

def process_cve_data(cve_items):
    rows = []
    for item in cve_items:
        cve_id = item['cve']['CVE_data_meta']['ID']
        description = item['cve']['description']['description_data'][0]['value']
        severity = item.get('impact', {}).get('baseMetricV3', {}).get('cvssV3', {}).get('baseSeverity', 'UNKNOWN')
        rows.append({
            'cve_id': cve_id,
            'description': description,
            'severity': severity
        })

    df = pd.DataFrame(rows)
    return df

def build_graph_data(df):
    # Step 1: Encode text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=300)
    tfidf_matrix = vectorizer.fit_transform(df['description']).toarray()
    x = torch.tensor(tfidf_matrix, dtype=torch.float)

    # Step 2: Create labels
    severity_mapping = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3, 'UNKNOWN': -1}
    y = torch.tensor([severity_mapping.get(sev, -1) for sev in df['severity']], dtype=torch.long)

    # Step 3: Build edge_index using cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    threshold = 0.5  # You can tune this
    edge_index = []

    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if i != j and similarity_matrix[i][j] > threshold:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Step 4: Return PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return data
