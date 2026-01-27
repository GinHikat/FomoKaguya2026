import os
from huggingface_hub import hf_hub_download

BASE_DIR = os.getcwd()

if not BASE_DIR.endswith("FomoKaguya2026"):
    raise ValueError("Please run this script from the FomoKaguya2026 directory")

data_dir = os.path.join(BASE_DIR, "data")

os.makedirs(data_dir, exist_ok=True)

data_original_dir = os.path.join(data_dir, "original")

os.makedirs(data_original_dir, exist_ok=True)

hf_hub_download(repo_id="zinzinmit/Fomokaguya2026", filename="original/test.txt", local_dir=data_dir, repo_type="dataset")

hf_hub_download(repo_id="zinzinmit/Fomokaguya2026", filename="original/train.txt", local_dir=data_dir, repo_type="dataset")

data_processed_dir = os.path.join(data_dir, "processed")

os.makedirs(data_processed_dir, exist_ok=True)

hf_hub_download(repo_id="zinzinmit/Fomokaguya2026", filename="processed/test.csv", local_dir=data_dir, repo_type="dataset")

hf_hub_download(repo_id="zinzinmit/Fomokaguya2026", filename="processed/train.csv", local_dir=data_dir, repo_type="dataset")

hf_hub_download(repo_id="zinzinmit/Fomokaguya2026", filename="processed/for_visualization.csv", local_dir=data_dir, repo_type="dataset")