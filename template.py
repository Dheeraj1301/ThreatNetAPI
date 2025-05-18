import os

# Define your project name
project_name = "zero_day_ai_project"

# Define all file paths to be created
list_of_files = [
    f"{project_name}/data/raw/nvdcve-1.1-2024.json.gz",        # Download manually or keep empty
    f"{project_name}/data/processed/graph_data.pt",            # Will be generated

    f"{project_name}/data/README.md",

    f"{project_name}/src/data_ingestion.py",
    f"{project_name}/src/preprocessing.py",
    f"{project_name}/src/graph_builder.py",
    f"{project_name}/src/model_gnn.py",
    f"{project_name}/src/model_qml.py",                        # Optional
    f"{project_name}/src/utils.py",

    f"{project_name}/notebooks/EDA_and_Feature_Exploration.ipynb",

    f"{project_name}/dashboard/app.py",

    f"{project_name}/results/model_logs/.gitkeep",             # placeholder

    f"{project_name}/requirements.txt",
    f"{project_name}/main.py",
    f"{project_name}/README.md",
]

# Loop through each file path
for filepath in list_of_files:
    filepath = os.path.normpath(filepath)
    filedir, filename = os.path.split(filepath)

    # Create the directory if it doesn't exist
    if filedir and not os.path.exists(filedir):
        os.makedirs(filedir, exist_ok=True)
        print(f"✅ Created directory: {filedir}")

    # Create the file if it doesn't exist or is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "wb" if filename.endswith((".gz", ".pt")) else "w") as f:
            pass  # Creates an empty file (binary if .gz or .pt)
        print(f"📄 Created empty file: {filepath}")
    else:
        print(f"⚠️ File already exists and is not empty: {filepath}")
