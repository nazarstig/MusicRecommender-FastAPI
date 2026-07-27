import io
import boto3
import botocore.exceptions
import numpy as np
from sqlalchemy.orm import Session
from app.core.config import settings

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=settings.AWS_S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        self.s3_bucket_name = settings.AWS_S3_BUCKET
        self.s3_key = settings.AWS_S3_BUCKET_KEY

    def matrices_exist(self) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket_name, Key=self.s3_key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def matrices_bucket_exists(self) -> bool:
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket_name)
            return True
        except botocore.exceptions.ClientError:
            return False

    def save_matrices(
        self,
        U,
        V_T,
        Sigma
    ):
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            U=U,
            V_T=V_T,
            Sigma=Sigma,
        )
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket=self.s3_bucket_name,
            Key=self.s3_key,
            Body=buffer.getvalue(),
        )

    def load_matrices(
        self
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        response = self.s3_client.get_object(Bucket=self.s3_bucket_name, Key=self.s3_key)
        file_bytes = response["Body"].read()
        U = V_T = Sigma = None

        with np.load(io.BytesIO(file_bytes)) as matrices:
            required_keys = ["V_T", "Sigma", "U"]

            if not all(key in matrices for key in required_keys):
                raise ValueError(
                    f"S3 file 's3://{self.s3_bucket_name}/{self.s3_key}' is missing required keys: {required_keys}"
                )
                
            U = matrices["U"]
            V_T = matrices["V_T"]
            Sigma = matrices["Sigma"]

            print(
                f"✓ Матриці U, V_T, Sigma успішно завантажено з S3: s3://{self.s3_bucket_name}/{self.s3_key}"
            )
        
        return U, V_T, Sigma
