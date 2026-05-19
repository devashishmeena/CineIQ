import mlflow
import time
from .config import MLFLOW_TRACKING_URI, SVD_N_FACTORS
from .data_loader import load_movielens_data, load_tmdb_data
from .models import train_svd_model, build_tfidf_matrix, save_models

def main():
    print("Starting Training Pipeline...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("CineIQ_Hybrid_Models")
    
    with mlflow.start_run():
        start_time = time.time()
        
        # 1. Load Data
        print("\n--- Loading Data ---")
        ratings = load_movielens_data()
        movies = load_tmdb_data()
        
        mlflow.log_metric("num_ratings", len(ratings))
        mlflow.log_metric("num_movies", len(movies))
        
        # 2. Train SVD (Collaborative Filtering)
        print("\n--- Training Collaborative Filtering (SVD) ---")
        mlflow.log_param("svd_n_factors", SVD_N_FACTORS)
        
        svd_model = train_svd_model(ratings)
        
        # 3. Train TF-IDF (Content-Based)
        print("\n--- Training Content-Based Filtering (TF-IDF) ---")
        tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(movies)
        
        mlflow.log_metric("tfidf_features", tfidf_matrix.shape[1])
        
        # 4. Save Models
        print("\n--- Saving Models ---")
        save_models(svd_model, tfidf_vectorizer, tfidf_matrix)
        
        end_time = time.time()
        mlflow.log_metric("training_duration_seconds", end_time - start_time)
        print(f"\nPipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
