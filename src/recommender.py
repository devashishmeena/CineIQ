import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

from .config import HYBRID_WEIGHT_COLLAB, HYBRID_WEIGHT_CONTENT
from .models import load_models
from .data_loader import load_movielens_data, load_tmdb_data, load_imdb_reviews

class CineIQRecommender:
    def __init__(self):
        print("Initializing CineIQ Recommender...")
        self.svd_model_data, self.tfidf_vectorizer, self.tfidf_matrix = load_models()
        self.movies = load_tmdb_data()
        
        # Need links to map TMDB ids to MovieLens ids
        from .config import RAW_DATA_DIR
        links_path = RAW_DATA_DIR / "links.csv"
        if links_path.exists():
            self.links = pd.read_csv(links_path)
            self.tmdb_to_ml = dict(zip(self.links.tmdbId, self.links.movieId))
            self.ml_to_tmdb = dict(zip(self.links.movieId, self.links.tmdbId))
        else:
            self.tmdb_to_ml = {}
            self.ml_to_tmdb = {}

        self.analyzer = SentimentIntensityAnalyzer()
        
        # Create a reverse map of movie indices in the dataframe for TF-IDF
        self.indices = pd.Series(self.movies.index, index=self.movies['id']).drop_duplicates()
        print("Initialization complete.")

    def get_content_based_scores(self, tmdb_id):
        """Get movie similarity scores using TF-IDF."""
        if tmdb_id not in self.indices:
            return np.zeros(len(self.movies))
            
        idx = self.indices[tmdb_id]
        # Get cosine similarity of this movie with all others
        cosine_sim = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        return cosine_sim

    def get_collaborative_score(self, user_id, ml_movie_id):
        """Predict user rating for a movie using SVD."""
        svd = self.svd_model_data['svd']
        user_index = self.svd_model_data['user_index']
        movie_columns = self.svd_model_data['movie_columns']
        user_factors = self.svd_model_data['user_factors']
        movie_factors = self.svd_model_data['movie_factors']
        
        if user_id not in user_index or ml_movie_id not in movie_columns:
            return 3.0 # Default average rating
            
        u_idx = user_index.get_loc(user_id)
        m_idx = movie_columns.get_loc(ml_movie_id)
        
        # Reconstruct the prediction: dot product of user and movie factors
        pred = np.dot(user_factors[u_idx], movie_factors[:, m_idx])
        return pred

    def get_sentiment_penalty(self, original_title):
        """Basic sentiment analysis to penalize movies with negative IMDB reviews."""
        # For prototype, we just return a 0 penalty. 
        # In full production, we'd query the IMDB dataset reviews for this title.
        return 0.0

    def recommend(self, user_id, target_tmdb_id, top_n=10):
        """Hybrid Recommendation combining SVD, TF-IDF and Sentiment."""
        print(f"Generating recommendations for User {user_id} based on Movie {target_tmdb_id}...")
        
        # 1. Get Content Scores
        content_scores = self.get_content_based_scores(target_tmdb_id)
        
        # Normalize content scores to 0-5 scale to match collaborative
        if content_scores.max() > 0:
            content_scores = (content_scores / content_scores.max()) * 5.0
            
        recommendations = []
        
        for i, row in self.movies.iterrows():
            tmdb_id = row['id']
            title = row.get('original_title', f"Movie {tmdb_id}")
            if tmdb_id == target_tmdb_id:
                continue
                
            c_score = content_scores[i]
            
            # 2. Get Collaborative Score
            ml_id = self.tmdb_to_ml.get(tmdb_id, -1)
            collab_score = self.get_collaborative_score(user_id, ml_id)
            
            # 3. Sentiment Re-ranking
            sentiment_penalty = self.get_sentiment_penalty(title)
            
            # Hybrid Formula
            final_score = (HYBRID_WEIGHT_CONTENT * c_score) + \
                          (HYBRID_WEIGHT_COLLAB * collab_score) - \
                          sentiment_penalty
                          
            recommendations.append({
                'tmdb_id': tmdb_id,
                'title': title,
                'content_score': c_score,
                'collab_score': collab_score,
                'final_score': final_score,
                'explanation': f"Content: {c_score:.2f}, Collab: {collab_score:.2f}"
            })
            
        # Sort by final score
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        return recommendations[:top_n]

if __name__ == "__main__":
    recommender = CineIQRecommender()
    
    # Example: User 1, Movie 135397 (Jurassic World)
    recs = recommender.recommend(user_id=1, target_tmdb_id=135397)
    
    print("\n--- Top Recommendations ---")
    for r in recs:
        print(f"{r['title']} | Score: {r['final_score']:.2f} | ({r['explanation']})")
