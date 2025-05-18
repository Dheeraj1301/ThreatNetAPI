import json
import gzip
from pathlib import Path

def load_nvd_cve_data(filepath):
    """
    Load CVE data from a gzipped JSON file.

    Args:
        filepath (str or Path): Path to the .json.gz file.

    Returns:
        List[dict]: List of CVE item dictionaries.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['CVE_Items']


if __name__ == "__main__":
    # Path relative to the project root
    file_path = "zero_day_ai_project/data/raw/nvdcve-1.1-2024.json.gz"

    # Load data
    try:
        cve_items = load_nvd_cve_data(file_path)
        print(f"✅ Loaded {len(cve_items)} CVE entries.")
        print("🔹 Sample CVE ID:", cve_items[0]['cve']['CVE_data_meta']['ID'])
    except Exception as e:
        print(f"❌ Error: {e}")
