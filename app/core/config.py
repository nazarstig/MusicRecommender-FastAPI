from pydantic_settings import BaseSettings
from typing import Optional
import urllib.parse


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Music Recommender"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A FastAPI-based music recommender system"
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Database Configuration
    DB_SERVER: str = "DESKTOP-8OOAIKI"
    DB_NAME: str = "LastFmDb"
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"

    AWS_S3_BUCKET: Optional[str] = "recommender-matrices"
    AWS_S3_BUCKET_KEY: str = "models/latest/recommendations_matrices.npz"
    AWS_S3_REGION: Optional[str] = None
    AWS_S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
    AWS_ACCESS_KEY_ID: Optional[str] = "minioadmin"
    AWS_SECRET_ACCESS_KEY: Optional[str] = "minioadmin"
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct SQL Server connection string"""
        params = urllib.parse.quote_plus(
            f"DRIVER={{{self.DB_DRIVER}}};SERVER={self.DB_SERVER};DATABASE={self.DB_NAME};Trusted_Connection=yes;"
        )
        return f"mssql+pyodbc:///?odbc_connect={params}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
