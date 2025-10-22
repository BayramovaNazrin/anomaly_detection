import os

for folder in ["artifacts", "artifacts/plots", "artifacts/models", "artifacts/data"]:
    os.makedirs(folder, exist_ok=True)
