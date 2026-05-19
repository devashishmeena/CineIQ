import os
import shutil
import kagglehub
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Kagglehub Datasets
DATASETS = [
    ("garymk/movielens-25m-dataset", "MovieLens 25M"),
    ("juzershakir/tmdb-movies-dataset", "TMDB Metadata"),
    ("aleynakorkmaz/imdb-dataset-of-50k-movie-reviews", "IMDB 50K Reviews")
]

def setup_directories():
    print("Ensuring data directories exist...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def copy_files_to_raw(source_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_file = Path(root) / file
            dest_file = RAW_DATA_DIR / file
            if not dest_file.exists():
                shutil.copy2(source_file, dest_file)
                print(f"Copied {file} to data/raw/")
            else:
                print(f"File {file} already exists in data/raw/")

def download_datasets():
    for dataset_id, dataset_name in DATASETS:
        print(f"\nDownloading {dataset_name} ({dataset_id}) via kagglehub...")
        cache_path = kagglehub.dataset_download(dataset_id)
        print(f"Downloaded to cache: {cache_path}")
        print(f"Copying {dataset_name} files to {RAW_DATA_DIR}...")
        copy_files_to_raw(cache_path)

if __name__ == "__main__":
    setup_directories()
    download_datasets()
    print("\nAll datasets downloaded and copied successfully!")
