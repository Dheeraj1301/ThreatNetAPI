# src/preprocessing.py
from sklearn.model_selection import train_test_split
import pandas as pd

def stratified_train_test_split(df, stratify_col='severity', test_size=0.2, random_state=42):
    """
    Perform train/test split with stratification to keep label distribution balanced.
    :param df: pandas DataFrame containing the dataset with a stratify_col column
    :param stratify_col: column name to stratify on (e.g. severity)
    :param test_size: fraction for test split
    :param random_state: for reproducibility
    :return: train_df, test_df
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    return train_df, test_df
import pandas as pd

def process_cve_data(cve_items):
    """
    Convert raw CVE JSON items into a pandas DataFrame with severity labels and description.
    """
    rows = []
    for item in cve_items:
        cve_id = item['cve']['CVE_data_meta']['ID']
        description = item['cve']['description']['description_data'][0]['value']
        # Example: Extract severity (use your actual path from JSON)
        severity = item.get('impact', {}).get('baseMetricV3', {}).get('cvssV3', {}).get('baseSeverity', 'UNKNOWN')
        
        rows.append({
            'cve_id': cve_id,
            'description': description,
            'severity': severity
        })

    df = pd.DataFrame(rows)
    return df
