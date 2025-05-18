import json
import gzip

def load_nvd_cve_data(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded JSON top-level keys: {list(data.keys())}")  # Debug print
    print(f"Number of CVE items: {len(data['CVE_Items'])}")
    return data['CVE_Items']

if __name__ == "__main__":
    filepath = r"C:/Users/disha/OneDrive/Desktop/zero_day/Zero_day_API/zero_day_ai_project/data/raw/nvdcve-1.1-2024.json.gz"
    cve_items = load_nvd_cve_data(filepath)
    print(f"Sample CVE item:\n{cve_items[0]}")
