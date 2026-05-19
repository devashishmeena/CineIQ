import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path
from .config import SVD_N_FACTORS, MODEL_DIR

def setup_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_svd_model(ratings):
    """
    Train an SVD model using sklearn's TruncatedSVD on user ratings.
    `ratings` must be a DataFrame with ['userId', 'movieId', 'rating']
    """
    print("Preparing data for SVD model...")
    # To save memory, we can limit the data to users/movies with enough interactions
    # or just create a sparse matrix directly.
    # For simplicity, we pivot the dataframe
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    
    print(f"Training TruncatedSVD model (n_components={SVD_N_FACTORS})...")
    svd = TruncatedSVD(n_components=SVD_N_FACTORS, random_state=42)
    # Fit the model to find latent features for users
    user_factors = svd.fit_transform(user_movie_matrix)
    movie_factors = svd.components_
    
    # We will save the SVD object and the original matrix columns to map back to movieIds
    model_data = {
        'svd': svd,
        'user_factors': user_factors,
        'movie_factors': movie_factors,
        'user_index': user_movie_matrix.index,
        'movie_columns': user_movie_matrix.columns
    }
    
    print("SVD Model training complete.")
    return model_data

def build_tfidf_matrix(movies):
    """
    Build a TF-IDF matrix for content-based filtering.
    `movies` must be a DataFrame with an 'overview' or 'genres' column.
    """
    print("Building TF-IDF matrix...")
    
    def extract_genres(genre_str):
        if pd.isna(genre_str):
            return ""
        try:
            import ast
            genres = ast.literal_eval(genre_str)
            return " ".join([g['name'] for g in genres])
        except:
            return str(genre_str)

    if 'genres' in movies.columns:
        movies['genres_clean'] = movies['genres'].apply(extract_genres)
    else:
        movies['genres_clean'] = ""

    if 'overview' not in movies.columns:
        movies['overview'] = ""

    movies['combined_features'] = movies['overview'].fillna("") + " " + movies['genres_clean'].fillna("")
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    
    print(f"TF-IDF Matrix generated with shape: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix

def save_models(svd_model_data, tfidf_vectorizer, tfidf_matrix):
    """Save the models to disk."""
    setup_model_dir()
    
    svd_path = MODEL_DIR / "svd_model.pkl"
    with open(svd_path, 'wb') as f:
        pickle.dump(svd_model_data, f)
    print(f"SVD model saved to {svd_path}")
    
    tfidf_path = MODEL_DIR / "tfidf_model.pkl"
    with open(tfidf_path, 'wb') as f:
        pickle.dump((tfidf_vectorizer, tfidf_matrix), f)
    print(f"TF-IDF model saved to {tfidf_path}")

def load_models():
    """Load models from disk."""
    svd_path = MODEL_DIR / "svd_model.pkl"
    tfidf_path = MODEL_DIR / "tfidf_model.pkl"
    
    with open(svd_path, 'rb') as f:
        svd_model_data = pickle.load(f)
        
    with open(tfidf_path, 'rb') as f:
        tfidf_vectorizer, tfidf_matrix = pickle.load(f)
        
    return svd_model_data, tfidf_vectorizer, tfidf_matrix
