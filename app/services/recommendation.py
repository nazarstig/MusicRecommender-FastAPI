import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from app.models.rating import Rating
from typing import List, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class RecommendationService:
    def __init__(self):
        self.predictions_df = None
        self.V = None
        self.Sigma = None
    
    def get_ratings(self, db: Session) -> List[Rating]:
        ratings = db.query(Rating).all()
       
        return ratings
    
    def create_user_item_matrix(self, db: Session) -> pd.DataFrame:
        ratings = self.get_ratings(db)
        data = [{'UserId': r.UserId, 'TrackId': r.TrackId, 'Rating': r.Rating} for r in ratings]    
        df = pd.DataFrame(data)
        matrix = df.pivot_table(index='UserId', columns='TrackId', values='Rating', fill_value=0)
        
        return matrix
    
    def create_recommendations(self, db: Session):
        user_item_matrix = self.create_user_item_matrix(db)
        sparse_matrix = csr_matrix(user_item_matrix.values)
        u, s, vt = svds(sparse_matrix, k=100)
        sigma = np.diag(s)

        self.V = vt.T
        self.Sigma = sigma

        all_user_predicted_ratings = np.dot(np.dot(u, sigma), vt)
        self.predictions_df = pd.DataFrame(all_user_predicted_ratings, 
                              columns=user_item_matrix.columns, 
                              index=user_item_matrix.index)
                
        return self.predictions_df.head()
    
    def get_recommended_songs_for_user(self, db: Session, user_id: str, num_recommendations: int = 5) -> List[Dict]:
        if self.predictions_df is None:
            raise ValueError("Predictions not created yet. Call create_recommendations() first.")
        
        user_predictions = self.predictions_df.loc[user_id].sort_values(ascending=False)
        user_rated_songs = set(self.get_already_rated_songs(db, user_id))
        recommendations = [song for song in user_predictions.index if song not in user_rated_songs]
        
        return [{"TrackId": track_id, "PredictedRating": float(user_predictions[track_id])} 
                for track_id in recommendations[:num_recommendations]]
    
    def get_recommended_songs_from_input(self, input_songs: List[str], num_recommendations: int = 5) -> List[Dict]:
        if self.predictions_df is None:
            raise ValueError("Predictions not created yet. Call create_recommendations() first.")
        
        input_songs = set(input_songs)
        # validation to ensure input songs are in the dataset
        valid_input_songs = input_songs.intersection(set(self.predictions_df.columns))
         
        r_new = np.zeros(self.predictions_df.shape[1])
        for song in valid_input_songs:
            r_new[self.predictions_df.columns.get_loc(song)] = 50.0
        r_new = r_new.reshape(1, -1)
        u_new = np.dot(r_new, np.dot(self.V, np.linalg.inv(self.Sigma)))
        r = np.dot(u_new, np.dot(self.Sigma, self.V.T))
        song_scores = pd.Series(r.flatten(), index=self.predictions_df.columns).sort_values(ascending=False)
        recommendations = [song for song in song_scores.index if song not in input_songs]
        
        return [{"TrackId": track_id, "PredictedRating": float(song_scores[track_id])} 
                for track_id in recommendations[:num_recommendations]]
    
    def get_already_rated_songs(self, db: Session, user_id: str) -> List[str]:
        rated_tracks = (
            db.query(Rating.TrackId)
            .filter(Rating.UserId == user_id)
            .distinct()
            .all()
        )

        return [track_id for (track_id,) in rated_tracks]