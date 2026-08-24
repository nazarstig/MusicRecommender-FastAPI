"""
Recompute the SVD recommendation matrices (U, V_T, Sigma) from the current
ratings data, register a new MLflow Model Registry version, and - if it beats
the currently deployed model - promote it via the "champion" alias.

Meant to run standalone, independent of the API process - e.g. on a schedule
(cron, Windows Task Scheduler, etc.).

The SVD rank is picked up from the best evaluation run logged by
evaluate_recommendations.py (the child run with the highest ndcg_at_10 metric
in MLFLOW_EXPERIMENT_NAME) - falls back to DEFAULT_SVD_RANK if no such run
exists, so evaluation stays optional, not a hard dependency.

Every trained model is registered as a new version of
MLFLOW_REGISTERED_MODEL_NAME, tagged "challenger". It's promoted to
"champion" only if its NDCG (inherited from the evaluation run that picked
its rank) is not worse than the NDCG of the run behind the current champion -
a candidate with nothing to compare against (no eval run, or nothing
deployed yet) is always promoted. A rejected candidate stays registered as
"challenger" for inspection, and the run exits non-zero so a scheduler can
flag it. Roll back to an older version with rollback_model.py.

Temporary bridge: until the API is migrated to read from the MLflow registry
directly (a later phase), every promotion also copies the model's artifact
to the old fixed S3 key (models/latest/recommendations_matrices.npz) the API
currently loads from - so production serving keeps working during the
transition. Remove this once that migration lands.

Usage:
    python build_recommendation_matrices.py
"""
import os
import sys
import tempfile
from pathlib import Path

# Must happen before anything touches stdout - see evaluate_recommendations.py
# for why (mlflow's own status messages include emoji that crash on Windows'
# default cp1252 console encoding).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.recommendation_service import RecommendationService

DEFAULT_SVD_RANK = 200
NDCG_METRIC_NAME = "ndcg_at_10"
ARTIFACT_FILENAME = "recommendations_matrices.npz"

# mlflow's S3 artifact repo talks to S3/MinIO directly (not proxied through the
# tracking server), so it needs these as real env vars, same as the server itself.
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.AWS_S3_ENDPOINT_URL or "")
os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.AWS_ACCESS_KEY_ID or "")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.AWS_SECRET_ACCESS_KEY or "")

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)


class SvdMatricesModel(mlflow.pyfunc.PythonModel):
    """Thin wrapper so the U/V_T/Sigma artifact can be registered as a proper
    MLflow Model (register_model needs a logged model, not a loose file).
    predict() is intentionally unimplemented - nothing calls it yet; matrices
    are loaded directly from the artifact via download_artifacts()."""

    def predict(self, context, model_input, params=None):
        raise NotImplementedError("matrices are loaded directly from the artifact, not via predict()")


def find_best_evaluated_rank(client: MlflowClient):
    """Uses the *most recent* evaluate_recommendations.py search, not the
    best NDCG ever recorded - ratings change over time (this runs weekly),
    so an old high score from stale data must never outrank a fresh one."""
    experiment = client.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return None, None, None

    latest_searches = client.search_runs(
        [experiment.experiment_id],
        filter_string="params.best_svd_rank != ''",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not latest_searches:
        return None, None, None

    latest_search = latest_searches[0]
    svd_rank = int(latest_search.data.params["best_svd_rank"])
    ndcg = latest_search.data.metrics.get(f"best_{NDCG_METRIC_NAME}")
    return svd_rank, ndcg, latest_search.info.run_id


def get_champion_ndcg(client: MlflowClient):
    try:
        champion_version = client.get_model_version_by_alias(settings.MLFLOW_REGISTERED_MODEL_NAME, "champion")
    except MlflowException:
        return None
    champion_run = client.get_run(champion_version.run_id)
    return champion_run.data.metrics.get(NDCG_METRIC_NAME)


def bridge_to_legacy_latest(s3_service, run_id: str):
    """Temporary: keep the not-yet-migrated API working by also writing the
    promoted model to the old fixed S3 key it still reads from directly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"model/artifacts/{ARTIFACT_FILENAME}", dst_path=tmp_dir
        )
        data = Path(local_path).read_bytes()
    s3_service.upload_bytes(data, s3_service.s3_key)


def main():
    db = SessionLocal()
    try:
        service = RecommendationService()
        client = MlflowClient()

        user_item_matrix = service.create_user_item_matrix(db)
        if user_item_matrix.empty:
            raise RuntimeError("No ratings found to build the user-item matrix.")

        svd_rank, candidate_ndcg, source_run_id = find_best_evaluated_rank(client)
        if svd_rank is None:
            svd_rank = DEFAULT_SVD_RANK
            print(f"No usable evaluation run found; using default SVD rank {DEFAULT_SVD_RANK}. Promotion gate will be skipped (nothing to compare against).")
        else:
            print(f"Using SVD rank {svd_rank} from evaluation run {source_run_id} (mean {NDCG_METRIC_NAME}={candidate_ndcg}).")

        svd_rank = min(svd_rank, user_item_matrix.shape[0] - 1, user_item_matrix.shape[1] - 1)

        U, V_T, Sigma = service.count_recommendation_matrices(user_item_matrix, k=svd_rank)

        with mlflow.start_run(run_name=f"train_svd_rank_{svd_rank}") as run:
            mlflow.log_param("svd_rank", svd_rank)
            mlflow.log_param("train_users", user_item_matrix.shape[0])
            mlflow.log_param("train_tracks", user_item_matrix.shape[1])
            if source_run_id:
                mlflow.log_param("source_eval_run_id", source_run_id)
            if candidate_ndcg is not None:
                mlflow.log_metric(NDCG_METRIC_NAME, candidate_ndcg)

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, ARTIFACT_FILENAME)
                np.savez_compressed(local_path, U=U, V_T=V_T, Sigma=Sigma)
                model_info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=SvdMatricesModel(),
                    artifacts={"matrices": local_path},
                )

            run_id = run.info.run_id

        registered = mlflow.register_model(model_info.model_uri, settings.MLFLOW_REGISTERED_MODEL_NAME)
        client.set_registered_model_alias(settings.MLFLOW_REGISTERED_MODEL_NAME, "challenger", registered.version)
        print(f"Registered version {registered.version} of '{settings.MLFLOW_REGISTERED_MODEL_NAME}', tagged challenger.")

        previous_ndcg = get_champion_ndcg(client)
        promoted = candidate_ndcg is None or previous_ndcg is None or candidate_ndcg >= previous_ndcg

        if promoted:
            client.set_registered_model_alias(settings.MLFLOW_REGISTERED_MODEL_NAME, "champion", registered.version)
            bridge_to_legacy_latest(service.s3_service, run_id)
            print(
                f"✓ Promoted version {registered.version} (rank={svd_rank}, "
                f"{user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} tracks) to champion."
            )
        else:
            print(
                f"✗ NOT promoted: candidate {NDCG_METRIC_NAME}={candidate_ndcg:.4f} is worse than "
                f"current champion's {previous_ndcg:.4f}. Version {registered.version} stays challenger only."
            )

        if not promoted:
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
