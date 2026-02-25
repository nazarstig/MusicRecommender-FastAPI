from fastapi.params import Depends
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.rating import Rating
from app.models.song import Song
from typing import List, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class RecommendationService:
    def __init__(self):
        self.predictions_df = None
        self.db = get_db()
    
    @staticmethod
    def loadRatings(db: Session) -> List[Rating]:
        """Load ratings from database"""
        ratings = db.query(Rating).limit(1000000).all()
        return ratings
    
    @staticmethod
    def create_user_item_matrix(ratings: List[Rating]) -> pd.DataFrame:
        data = [{'UserId': r.UserId, 'TrackId': r.TrackId, 'Rating': r.Rating} for r in ratings]
        
        df = pd.DataFrame(data)
        
        matrix = df.pivot_table(index='UserId', columns='TrackId', values='Rating', fill_value=0)
        
        #delete
        print(matrix.shape)
        
        return matrix
    
    def create_recommendations(self, ratings: List[Rating]):
        user_item_matrix = RecommendationService.create_user_item_matrix(ratings)
        sparse_matrix = csr_matrix(user_item_matrix.values)
        u, s, vt = svds(sparse_matrix, k=50)

        #s comes as a list of values; need convert to diagonal matrix
        sigma = np.diag(s)

        # Reconstruct the predicted ratings
        all_user_predicted_ratings = np.dot(np.dot(u, sigma), vt)

        self.predictions_df = pd.DataFrame(all_user_predicted_ratings, 
                              columns=user_item_matrix.columns, 
                              index=user_item_matrix.index)
        
        #delete
        print(self.predictions_df.shape)
        print('recommendations created')
        
        return self.predictions_df.head()
    
    def recommend_songs_for_user(self, user_id: str, num_recommendations: int = 5) -> List[Dict]:
        if self.predictions_df is None:
            raise ValueError("Predictions not created yet. Call create_recommendations() first.")
        
        user_predictions = self.predictions_df.loc[user_id].sort_values(ascending=False)
        
        user_rated_songs = set(self.get_already_rated_songs(user_id))

        recommendations = [song for song in user_predictions.index if song not in user_rated_songs]
        
        return [{"TrackId": track_id, "PredictedRating": float(user_predictions[track_id])} 
                for track_id in recommendations[:num_recommendations]]
    
    def get_already_rated_songs(self, user_id: str, db: Session = Depends(get_db)) -> List[str]:
        rated_tracks = (
            db.query(Rating.TrackId)
            .filter(Rating.UserId == user_id)
            .distinct()
            .all()
        )

        return [track_id for (track_id,) in rated_tracks]