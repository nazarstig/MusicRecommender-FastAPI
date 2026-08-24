"""
Offline evaluation of the SVD recommender using NDCG@k, with a small search
over candidate SVD ranks to find which one generalizes best.

For each user with at least --min-ratings ratings, holds out --test-fraction
of their ratings as a test set (the same split is reused for every candidate
rank, so the comparison is apples-to-apples), trains the SVD at each of
--svd-ranks on the remaining ("train") ratings, and scores each model's
top-k recommendations against the held-out ("test") tracks with NDCG@k.

Results are printed to stdout and appended as a CSV row under eval_results/,
so a run can be compared against earlier ones later.

Standalone: not part of the API, not run on a schedule. Run manually against
the configured DB whenever you want to sanity-check ranking quality or pick
an SVD rank (e.g. after changing the "active track" filter or fold-in math).

Usage:
    python evaluate_recommendations.py
    python evaluate_recommendations.py --svd-ranks 50,100,200 --ndcg-k 10 --test-fraction 0.2 --min-ratings 5 --seed 42
"""
import argparse
import csv
import io
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.recommendation_service import RecommendationService
from app.services.s3_service import S3Service

DEFAULT_SVD_RANKS = [50, 100, 200]
RESULTS_DIR = "eval_results"
BEST_POINTER_FILENAME = "best_svd_rank.json"
RESULTS_FIELDNAMES = [
    "requested_svd_rank", "svd_rank_used", "mean_ndcg", "median_ndcg", "min_ndcg", "max_ndcg",
    "users_evaluated", "skipped_no_row", "skipped_no_overlap",
    "train_users", "train_tracks", "ndcg_k", "test_fraction", "min_ratings", "seed",
]


def split_train_test(ratings_df: pd.DataFrame, test_fraction: float, min_ratings: int, seed: int):
    """Per user: hold out test_fraction of ratings as test if they have >= min_ratings,
    otherwise keep all of that user's ratings in train (nothing to evaluate for them)."""
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for _, group in ratings_df.groupby("UserId"):
        if len(group) < min_ratings:
            train_parts.append(group)
            continue

        shuffled = group.sample(frac=1.0, random_state=rng)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings_df.iloc[0:0]
    return train_df, test_df


def dcg(relevances: List[float]) -> float:
    """Linear-gain DCG: sum(rel_i / log2(rank_i + 1)). Linear rather than the
    exponential (2^rel - 1) variant, since Rating isn't a bounded 0-5 scale here."""
    return sum(rel / np.log2(pos + 2) for pos, rel in enumerate(relevances))


def ndcg_at_k_for_user(
    predictions_row: pd.Series,
    already_seen: List[str],
    test_relevance: Dict[str, float],
    k: int,
) -> Optional[float]:
    candidates = predictions_row.drop(labels=already_seen, errors="ignore")
    top_k = candidates.sort_values(ascending=False).index[:k]

    actual_dcg = dcg([test_relevance.get(track_id, 0.0) for track_id in top_k])
    ideal_dcg = dcg(sorted(test_relevance.values(), reverse=True)[:k])

    if ideal_dcg == 0:
        return None

    return actual_dcg / ideal_dcg


def load_ratings():
    db = SessionLocal()
    try:
        service = RecommendationService()
        ratings = service.get_ratings(db)
        ratings_df = pd.DataFrame(
            [{"UserId": r.UserId, "TrackId": r.TrackId, "Rating": r.Rating} for r in ratings]
        )
        return ratings_df, service
    finally:
        db.close()


def evaluate_svd_rank(
    service: RecommendationService,
    train_matrix: pd.DataFrame,
    train_seen_by_user: Dict[str, List[str]],
    test_relevance_by_user: Dict[str, Dict[str, float]],
    requested_rank: int,
    ndcg_k: int,
) -> Dict:
    svd_rank = min(requested_rank, train_matrix.shape[0] - 1, train_matrix.shape[1] - 1)
    U, V_T, Sigma = service.count_recommendation_matrices(train_matrix, k=svd_rank)
    service.create_predictions_df(U, Sigma, V_T, train_matrix)
    predictions_df = service.predictions_df

    scores = []
    skipped_no_row = 0
    skipped_no_overlap = 0

    for user_id, test_relevance in test_relevance_by_user.items():
        if user_id not in predictions_df.index:
            skipped_no_row += 1
            continue

        overlapping_relevance = {
            track_id: rel for track_id, rel in test_relevance.items() if track_id in predictions_df.columns
        }
        if not overlapping_relevance:
            skipped_no_overlap += 1
            continue

        already_seen = train_seen_by_user.get(user_id, [])
        score = ndcg_at_k_for_user(predictions_df.loc[user_id], already_seen, overlapping_relevance, ndcg_k)
        if score is not None:
            scores.append(score)

    return {
        "requested_svd_rank": requested_rank,
        "svd_rank_used": svd_rank,
        "mean_ndcg": float(np.mean(scores)) if scores else None,
        "median_ndcg": float(np.median(scores)) if scores else None,
        "min_ndcg": float(np.min(scores)) if scores else None,
        "max_ndcg": float(np.max(scores)) if scores else None,
        "users_evaluated": len(scores),
        "skipped_no_row": skipped_no_row,
        "skipped_no_overlap": skipped_no_overlap,
    }


def run_search(svd_ranks: List[int], ndcg_k: int, test_fraction: float, min_ratings: int, seed: int) -> List[Dict]:
    ratings_df, service = load_ratings()
    if ratings_df.empty:
        raise RuntimeError("No ratings found to evaluate.")

    train_df, test_df = split_train_test(ratings_df, test_fraction, min_ratings, seed)
    if test_df.empty:
        raise RuntimeError(
            f"No user had >= {min_ratings} ratings, so no test set could be held out. "
            "Lower --min-ratings."
        )

    train_matrix = train_df.pivot_table(index="UserId", columns="TrackId", values="Rating", fill_value=0)
    train_seen_by_user = train_df.groupby("UserId")["TrackId"].apply(list).to_dict()
    test_relevance_by_user = (
        test_df.groupby("UserId")
        .apply(lambda g: dict(zip(g["TrackId"], g["Rating"])), include_groups=False)
        .to_dict()
    )

    print(f"Train matrix: {train_matrix.shape[0]} users x {train_matrix.shape[1]} tracks")

    results = []
    seen_effective_ranks = set()
    for requested_rank in svd_ranks:
        effective_rank = min(requested_rank, train_matrix.shape[0] - 1, train_matrix.shape[1] - 1)
        if effective_rank in seen_effective_ranks:
            print(f"skip svd_rank={requested_rank} (clamps to {effective_rank}, already evaluated)")
            continue
        seen_effective_ranks.add(effective_rank)

        result = evaluate_svd_rank(
            service, train_matrix, train_seen_by_user, test_relevance_by_user, requested_rank, ndcg_k
        )
        result.update(
            {
                "train_users": train_matrix.shape[0],
                "train_tracks": train_matrix.shape[1],
                "ndcg_k": ndcg_k,
                "test_fraction": test_fraction,
                "min_ratings": min_ratings,
                "seed": seed,
            }
        )
        results.append(result)

        if result["mean_ndcg"] is None:
            print(f"svd_rank={result['svd_rank_used']} (requested {requested_rank}): no users could be evaluated")
        else:
            print(
                f"svd_rank={result['svd_rank_used']} (requested {requested_rank}): "
                f"NDCG@{ndcg_k} mean={result['mean_ndcg']:.4f} median={result['median_ndcg']:.4f} "
                f"(users={result['users_evaluated']}, skipped_no_row={result['skipped_no_row']}, "
                f"skipped_no_overlap={result['skipped_no_overlap']})"
            )

    return results


def build_results_csv_text(results: List[Dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RESULTS_FIELDNAMES)
    writer.writeheader()
    writer.writerows(results)
    return buffer.getvalue()


def write_results_csv_local(csv_text: str, timestamp: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"k_search_{timestamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_text)
    return path


def upload_results_csv_to_s3(csv_text: str, timestamp: str) -> str:
    s3_service = S3Service()
    key = f"{settings.AWS_S3_EVAL_PREFIX}/k_search_{timestamp}.csv"
    s3_service.upload_bytes(csv_text.encode("utf-8"), key)
    return f"s3://{s3_service.s3_bucket_name}/{key}"


def build_best_pointer_json(best: Dict, timestamp: str) -> str:
    """The single 'current best' pointer build_recommendation_matrices.py reads to
    pick its SVD rank. Overwritten every run - it's a pointer, not history."""
    payload = {
        "svd_rank": best["svd_rank_used"],
        "requested_svd_rank": best["requested_svd_rank"],
        "mean_ndcg": best["mean_ndcg"],
        "ndcg_k": best["ndcg_k"],
        "evaluated_at": timestamp,
        "train_users": best["train_users"],
        "train_tracks": best["train_tracks"],
    }
    return json.dumps(payload, indent=2)


def write_best_pointer_local(pointer_json: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, BEST_POINTER_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(pointer_json)
    return path


def upload_best_pointer_to_s3(pointer_json: str) -> str:
    s3_service = S3Service()
    key = f"{settings.AWS_S3_EVAL_PREFIX}/{BEST_POINTER_FILENAME}"
    s3_service.upload_bytes(pointer_json.encode("utf-8"), key)
    return f"s3://{s3_service.s3_bucket_name}/{key}"


def parse_svd_ranks(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the SVD recommender with NDCG@k across candidate SVD ranks.")
    parser.add_argument(
        "--svd-ranks", type=parse_svd_ranks, default=DEFAULT_SVD_RANKS,
        help=f"Comma-separated SVD truncation ranks to try (default: {','.join(map(str, DEFAULT_SVD_RANKS))})",
    )
    parser.add_argument(
        "--ndcg-k", type=int, default=10,
        help="Cutoff for NDCG, i.e. how many top recommendations are scored (default: 10)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of each eligible user's ratings held out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--min-ratings", type=int, default=5,
        help="Minimum ratings a user must have to be included in the test set (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/test split (default: 42)")
    args = parser.parse_args()

    results = run_search(args.svd_ranks, args.ndcg_k, args.test_fraction, args.min_ratings, args.seed)

    scored = [r for r in results if r["mean_ndcg"] is not None]
    if not scored:
        print("No candidate rank could be evaluated - try lowering --min-ratings.")
        return

    best = max(scored, key=lambda r: r["mean_ndcg"])
    print(
        f"\nBest: svd_rank={best['svd_rank_used']} (requested {best['requested_svd_rank']}) "
        f"NDCG@{args.ndcg_k}={best['mean_ndcg']:.4f}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_text = build_results_csv_text(results)

    local_path = write_results_csv_local(csv_text, timestamp)
    print(f"Results written to {local_path}")

    try:
        s3_uri = upload_results_csv_to_s3(csv_text, timestamp)
        print(f"Results uploaded to {s3_uri}")
    except Exception as e:
        print(f"! Could not upload results to S3 ({e}) - local CSV above is still available.")

    pointer_json = build_best_pointer_json(best, timestamp)
    local_pointer_path = write_best_pointer_local(pointer_json)
    print(f"Best-rank pointer written to {local_pointer_path}")

    try:
        pointer_uri = upload_best_pointer_to_s3(pointer_json)
        print(f"Best-rank pointer uploaded to {pointer_uri} - build_recommendation_matrices.py will use it next run.")
    except Exception as e:
        print(f"! Could not upload best-rank pointer to S3 ({e}) - build_recommendation_matrices.py will keep using its previous/default rank.")


if __name__ == "__main__":
    main()
