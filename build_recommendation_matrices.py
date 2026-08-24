"""
Recompute the SVD recommendation matrices (U, V_T, Sigma) from the current
ratings data and upload them to S3/MinIO.

Meant to run standalone, independent of the API process - e.g. on a schedule
(cron, Windows Task Scheduler, etc.) - so the API only ever loads precomputed
matrices instead of computing them under request/startup load.

The SVD rank (k) is picked up from the "best_svd_rank.json" pointer that
evaluate_recommendations.py writes to S3 after each of its runs (see that
script). If the pointer is missing, unreachable, or malformed, this falls
back to DEFAULT_SVD_RANK - so evaluation is optional, not a hard dependency.

Usage:
    python build_recommendation_matrices.py
"""
import json

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.recommendation_service import RecommendationService
from app.services.s3_service import S3Service

DEFAULT_SVD_RANK = 200
BEST_POINTER_KEY_SUFFIX = "best_svd_rank.json"


def get_svd_rank_from_pointer(s3_service: S3Service, default_rank: int) -> int:
    pointer_key = f"{settings.AWS_S3_EVAL_PREFIX}/{BEST_POINTER_KEY_SUFFIX}"

    if not s3_service.object_exists(pointer_key):
        print(f"No evaluation pointer at s3://{s3_service.s3_bucket_name}/{pointer_key}; using default SVD rank {default_rank}.")
        return default_rank

    try:
        pointer = json.loads(s3_service.download_bytes(pointer_key).decode("utf-8"))
        rank = int(pointer["svd_rank"])
        print(
            f"Using SVD rank {rank} from evaluation pointer "
            f"(mean NDCG@{pointer.get('ndcg_k')}={pointer.get('mean_ndcg')}, evaluated at {pointer.get('evaluated_at')})."
        )
        return rank
    except Exception as e:
        print(f"! Could not read evaluation pointer ({e}); using default SVD rank {default_rank}.")
        return default_rank


def main():
    db = SessionLocal()
    try:
        service = RecommendationService()
        user_item_matrix = service.create_user_item_matrix(db)
        if user_item_matrix.empty:
            raise RuntimeError("No ratings found to build the user-item matrix.")

        svd_rank = get_svd_rank_from_pointer(service.s3_service, DEFAULT_SVD_RANK)
        svd_rank = min(svd_rank, user_item_matrix.shape[0] - 1, user_item_matrix.shape[1] - 1)

        U, V_T, Sigma = service.count_recommendation_matrices(user_item_matrix, k=svd_rank)

        service.s3_service.save_matrices(U, V_T, Sigma)
        print(
            f"✓ Rebuilt matrices (rank={svd_rank}) from {user_item_matrix.shape[0]} users x "
            f"{user_item_matrix.shape[1]} tracks and uploaded to "
            f"s3://{service.s3_service.s3_bucket_name}/{service.s3_service.s3_key}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
