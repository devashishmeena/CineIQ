import pandas as pd
from pathlib import Path
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def get_data_paths():
    """Returns a dictionary of expected file paths."""
    return {
        "ml_ratings": RAW_DATA_DIR / "ratings.csv",
        "ml_movies": RAW_DATA_DIR / "movies.csv",
        "tmdb_movies": RAW_DATA_DIR / "tmdb_movies_data.csv",
        "imdb_reviews": RAW_DATA_DIR / "IMDB Dataset.csv" # Might not exist yet
    }

def load_movielens_data(sample_size=100000):
    """Load MovieLens ratings."""
    paths = get_data_paths()
    ratings_path = paths["ml_ratings"]
    
    print(f"Loading MovieLens ratings from {ratings_path}...")
    # Load only necessary columns and sample to avoid memory overflow during SVD training on 25M rows
    ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
    
    # We sample the data for faster training demonstration. 
    # In production, we'd use PySpark or chunking.
    if len(ratings) > sample_size:
        print(f"Sampling {sample_size} ratings from {len(ratings)} for faster training...")
        ratings = ratings.sample(n=sample_size, random_state=42)
        
    return ratings

def load_tmdb_data():
    """Load TMDB metadata."""
    paths = get_data_paths()
    movies_path = paths["tmdb_movies"]
    
    print(f"Loading TMDB metadata from {movies_path}...")
    movies = pd.read_csv(movies_path)
    if 'overview' in movies.columns:
        movies['overview'] = movies['overview'].fillna('')
    return movies

def load_imdb_reviews():
    """Load IMDB reviews for sentiment model."""
    paths = get_data_paths()
    reviews_path = paths["imdb_reviews"]
    
    if not reviews_path.exists():
        print("IMDB reviews dataset not found. Skipping sentiment data load.")
        return pd.DataFrame()
        
    print(f"Loading IMDB reviews from {reviews_path}...")
    reviews = pd.read_csv(reviews_path)
    return reviews
