from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_descriptions(cve_items):
    return [item['cve']['description']['description_data'][0]['value'] for item in cve_items]

def get_tfidf_features(texts, max_features=100):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray(), vectorizer
