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

    def create_matrices_bucket(self):
        self.s3_client.create_bucket(Bucket=self.s3_bucket_name)

    def save_matrices(
        self,
        U,
        V_T,
        Sigma
    ):
        self.save_matrices_to_key(U, V_T, Sigma, self.s3_key)

    def save_matrices_to_key(self, U, V_T, Sigma, key: str):
        """Same as save_matrices, but to an arbitrary key (e.g. a versioned path)."""
        if not self.matrices_bucket_exists():
            self.create_matrices_bucket()

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
            Key=key,
            Body=buffer.getvalue(),
        )

    def object_exists(self, key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket_name, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def download_bytes(self, key: str) -> bytes:
        response = self.s3_client.get_object(Bucket=self.s3_bucket_name, Key=key)
        return response["Body"].read()

    def upload_bytes(self, data: bytes, key: str):
        """Upload arbitrary bytes to the same bucket under a given key (e.g. evaluation results)."""
        if not self.matrices_bucket_exists():
            self.create_matrices_bucket()

        self.s3_client.put_object(
            Bucket=self.s3_bucket_name,
            Key=key,
            Body=data,
        )

    def list_keys(self, prefix: str = ""):
        """Returns [(key, size, last_modified), ...] for every object under prefix."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.s3_bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append((obj["Key"], obj["Size"], obj["LastModified"]))
        return keys

    def copy_object(self, source_key: str, dest_key: str):
        """Server-side copy within the same bucket (no re-upload of the bytes)."""
        self.s3_client.copy_object(
            Bucket=self.s3_bucket_name,
            CopySource={"Bucket": self.s3_bucket_name, "Key": source_key},
            Key=dest_key,
        )

    def delete_object(self, key: str):
        self.s3_client.delete_object(Bucket=self.s3_bucket_name, Key=key)

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
