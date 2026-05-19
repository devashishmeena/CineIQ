import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MOVIES_PATH = RAW_DATA_DIR / "movies.csv"
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"
REVIEWS_PATH = RAW_DATA_DIR / "reviews.csv"

# Model Paths
MODEL_DIR = BASE_DIR / "models"
SVD_MODEL_PATH = MODEL_DIR / "svd_model.pkl"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.pkl"

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR}/mlflow.db"

# Hyperparameters
SVD_N_FACTORS = 50
SVD_N_EPOCHS = 20
HYBRID_WEIGHT_COLLAB = 0.6
HYBRID_WEIGHT_CONTENT = 0.4
