"""
Recompute the SVD recommendation matrices (U, V_T, Sigma) from the current
ratings data and upload them to S3/MinIO.

Meant to run standalone, independent of the API process - e.g. on a schedule
(cron, Windows Task Scheduler, etc.) - so the API only ever loads precomputed
matrices instead of computing them under request/startup load.

Usage:
    python build_recommendation_matrices.py
"""
from app.core.database import SessionLocal
from app.services.recommendation_service import RecommendationService
from app.services.s3_service import S3Service


def main():
    db = SessionLocal()
    try:
        service = RecommendationService()
        user_item_matrix = service.create_user_item_matrix(db)
        if user_item_matrix.empty:
            raise RuntimeError("No ratings found to build the user-item matrix.")

        U, V_T, Sigma = service.count_recommendation_matrices(user_item_matrix)

        s3_service = S3Service()
        s3_service.save_matrices(U, V_T, Sigma)
        print(
            f"✓ Rebuilt matrices from {user_item_matrix.shape[0]} users x "
            f"{user_item_matrix.shape[1]} tracks and uploaded to "
            f"s3://{s3_service.s3_bucket_name}/{s3_service.s3_key}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
