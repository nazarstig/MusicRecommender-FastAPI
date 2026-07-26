import os
from typing import List, Dict, Tuple, Optional

import boto3
import botocore.exceptions
from fastapi.params import Depends
from app.core.config import settings
import io

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from app.models.rating import Rating
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class RecommendationService:
    def __init__(self):
        self.predictions_df = None
        self.U = None
        self.V_T = None
        self.Sigma = None

        self.recommendations_filepath = "recommendations_manual.npz"
        self.s3_client = boto3.client(
                        "s3",
                        endpoint_url="http://localhost:9000",  # Адреса MinIO
                        aws_access_key_id="minioadmin",
                        aws_secret_access_key="minioadmin"
                        #region_name="us-east-1",
                    )
        self.s3_bucket_name = "recommender-matrices"
        self.s3_key = "models/latest/recommendations_matrices.npz"
    
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
        #self.load_prediction_matrices(self.recommendations_filepath)
        
        # Stop if predictions are already defined
        if self.predictions_df is not None:
            return self.predictions_df.head()
        
        user_item_matrix = self.create_user_item_matrix(db)
        sparse_matrix = csr_matrix(user_item_matrix.values)
        u, s, vt = svds(sparse_matrix, k=200)
        sigma = np.diag(s)

        self.U = u
        self.V_T = vt
        self.Sigma = sigma

        self.create_predictions_df(u, sigma, vt, user_item_matrix)
        # all_user_predicted_ratings = np.dot(np.dot(u, sigma), vt)
        # self.predictions_df = pd.DataFrame(all_user_predicted_ratings, 
        #                       columns=user_item_matrix.columns, 
        #                       index=user_item_matrix.index)
        
        #self.persist_predictions(self.recommendations_filepath)
        self.save_matrices_to_s3()
        return self.predictions_df.head()
    
    def create_predictions_df(self, u, sigma, vt, user_item_matrix):
        all_user_predicted_ratings = np.dot(np.dot(u, sigma), vt)
        self.predictions_df = pd.DataFrame(all_user_predicted_ratings, 
                                    columns=user_item_matrix.columns, 
                                    index=user_item_matrix.index)
    def persist_predictions(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        if self.predictions_df is None:
            raise ValueError("Predictions not created yet. Call create_recommendations() first.")

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "recommendations_matrices.npz")

        np.savez(
            output_path,
            U=self.U,
            V_T=self.V_T,
            Sigma=self.Sigma,
        )

        return output_path
    
    def s3_matrices_exists(self) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket_name, Key=self.s3_key)
            return True
        except botocore.exceptions.ClientError as e:
            return False

    def save_matrices_to_s3(
        self,
        bucket_name: str = "recommender-matrices",
        s3_key: str = "models/latest/recommendations_matrices.npz",
        s3_client=None,
    ):
        if self.predictions_df is None:
            raise ValueError(
                "Predictions not created yet. Call create_recommendations() first."
            )

        # 1. Ініціалізуємо S3 клієнт, якщо його не передали
        if s3_client is None:
            s3_client = boto3.client(
                "s3",
                endpoint_url="http://localhost:9000",  # Адреса MinIO
                aws_access_key_id="minioadmin",
                aws_secret_access_key="minioadmin"
                #region_name="us-east-1",
            )

        # 2. Перевіряємо/створюємо бакет
        existing_buckets = [
            b["Name"] for b in s3_client.list_buckets().get("Buckets", [])
        ]
        if bucket_name not in existing_buckets:
            s3_client.create_bucket(Bucket=bucket_name)

        # 3. Заковуємо матриці в буфер в ОЗП (In-Memory)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            U=self.U,
            V_T=self.V_T,
            Sigma=self.Sigma,
        )
        buffer.seek(0)

        # 4. Відправляємо в S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
        )

    def load_matrices_from_s3(
        self,
        db: Session = None,
        bucket_name: str = "recommender-matrices",
        s3_key: str = "models/latest/recommendations_matrices.npz",
        s3_client=None,
    ):
        # 1. Ініціалізуємо S3 клієнт, якщо його не передали
        if s3_client is None:
            s3_client = boto3.client(
                "s3",
                endpoint_url="http://localhost:9000",  # Адреса MinIO
                aws_access_key_id="minioadmin",
                aws_secret_access_key="minioadmin",
                region_name="us-east-1",
            )

        try:
            # 2. Отримуємо об'єкт із S3/MinIO
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            file_bytes = response["Body"].read()

            # 3. Зчитуємо npz-матриці напряму з байтового буфера в ОЗП
            with np.load(io.BytesIO(file_bytes)) as matrices:
                required_keys = ["V_T", "Sigma", "U"]

                if not all(key in matrices for key in required_keys):
                    raise ValueError(
                        f"S3 file 's3://{bucket_name}/{s3_key}' is missing required keys: {required_keys}"
                    )

                # 4. Присвоюємо матриці полям класу
                self.U = matrices["U"]
                self.V_T = matrices["V_T"]
                self.Sigma = matrices["Sigma"]

            user_item_matrix = self.create_user_item_matrix(db)
            self.create_predictions_df(self.U, self.Sigma, self.V_T, user_item_matrix)
            print(
                f"✓ Матриці U, V_T, Sigma успішно завантажено з S3: s3://{bucket_name}/{s3_key}"
            )

        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(
                f"Файл не знайдено в S3 за шляхом: s3://{bucket_name}/{s3_key}"
            )
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження матриць з S3: {str(e)}")

    def load_prediction_matrices(
        self,
        file_path: Optional[str] = None,
        db: Session = None
    ) -> str:
        if self.predictions_df is not None:
            return
        if file_path is None:
            raise ValueError("File path is not provided")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction matrices file not found: {file_path}")

        user_item_matrix = self.create_user_item_matrix(db)
        matrices = np.load(file_path)
        if matrices is None or not all(key in matrices for key in ["V_T", "Sigma", "U"]):
            return
            raise ValueError(f"Prediction matrices file is missing required keys: {file_path}")

        self.U = matrices["U"]
        self.V_T = matrices["V_T"]
        self.Sigma = matrices["Sigma"]

        self.create_predictions_df(self.U, self.Sigma, self.V_T, user_item_matrix)
        # all_user_predicted_ratings = np.dot(np.dot(self.U, self.Sigma), self.V_T)
        # self.predictions_df = pd.DataFrame(all_user_predicted_ratings, 
        #                               columns=user_item_matrix.columns, 
        #                               index=user_item_matrix.index)

    def _power_method(
        self,
        matrix: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> Tuple[float, np.ndarray]:
        # Explicit power iteration for dominant eigenpair of a symmetric matrix.
        n = matrix.shape[0]
        rng = np.random.default_rng(42)
        v = rng.normal(size=n)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            raise ValueError("Initial vector norm is zero in power method.")
        v = v / v_norm

        prev_lambda = 0.0
        for _ in range(max_iter):
            w = matrix @ v
            w_norm = np.linalg.norm(w)
            if w_norm < tol:
                return 0.0, v

            v_next = w / w_norm
            lambda_est = float(v_next.T @ matrix @ v_next)

            if abs(lambda_est - prev_lambda) < tol and np.linalg.norm(v_next - v) < tol:
                v = v_next
                prev_lambda = lambda_est
                break

            v = v_next
            prev_lambda = lambda_est

        return prev_lambda, v

    def _manual_svds(
        self,
        matrix: np.ndarray,
        k: int,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute top-k singular values/vectors via eigen-decomposition of A^T A.
        m, n = matrix.shape
        if m == 0 or n == 0:
            raise ValueError("Cannot compute SVD on an empty matrix.")

        max_k = min(m, n)
        k = max(1, min(k, max_k))

        gram = matrix.T @ matrix
        gram = (gram + gram.T) / 2.0

        singular_values = []
        right_vectors = []

        for _ in range(k):
            eigval, v = self._power_method(gram, max_iter=max_iter, tol=tol)
            if eigval <= tol:
                break

            sigma = float(np.sqrt(eigval))
            singular_values.append(sigma)
            right_vectors.append(v)

            # Deflation removes the extracted dominant component.
            gram = gram - eigval * np.outer(v, v)
            gram = (gram + gram.T) / 2.0

        if not singular_values:
            raise ValueError("No singular values were extracted by manual SVD.")

        s = np.array(singular_values)
        vt = np.array(right_vectors)

        u_columns = []
        for i, sigma in enumerate(s):
            v = vt[i]
            u_col = matrix @ v
            u_norm = np.linalg.norm(u_col)
            if u_norm > tol:
                u_col = u_col / u_norm
            else:
                u_col = np.zeros_like(u_col)
            u_columns.append(u_col)

        u = np.column_stack(u_columns)
        return u, s, vt
    
    def create_recommendations_2(self, db: Session):
        user_item_matrix = self.create_user_item_matrix(db)
        dense_matrix = user_item_matrix.values.astype(float)

        k = min(200, dense_matrix.shape[0], dense_matrix.shape[1])
        u, s, vt = self._manual_svds(dense_matrix, k=k)
        sigma = np.diag(s)

        self.V_T = vt
        self.Sigma = sigma

        all_user_predicted_ratings = np.dot(np.dot(u, sigma), vt)
        self.predictions_df = pd.DataFrame(
            all_user_predicted_ratings,
            columns=user_item_matrix.columns,
            index=user_item_matrix.index,
        )

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
        u_new = np.dot(r_new, np.dot(self.V_T.T, np.linalg.inv(self.Sigma)))
        r = np.dot(u_new, np.dot(self.Sigma, self.V_T))
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