"""
Roll back the production model by moving the "champion" alias in the MLflow
Model Registry to an earlier registered version, and re-syncing the legacy
S3 key (models/latest/recommendations_matrices.npz) the not-yet-migrated API
still reads from.

Usage:
    python rollback_model.py --list
    python rollback_model.py 3
"""
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os

import mlflow
from mlflow.tracking import MlflowClient

from app.core.config import settings
from app.services.s3_service import S3Service

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.AWS_S3_ENDPOINT_URL or "")
os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.AWS_ACCESS_KEY_ID or "")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.AWS_SECRET_ACCESS_KEY or "")

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def list_versions(client: MlflowClient):
    versions = client.search_model_versions(f"name='{settings.MLFLOW_REGISTERED_MODEL_NAME}'")
    versions.sort(key=lambda v: int(v.version), reverse=True)
    return versions


def main():
    parser = argparse.ArgumentParser(description="Roll back the production model to an earlier registered version.")
    parser.add_argument("version", nargs="?", type=int, help="Registered model version number to promote (see --list)")
    parser.add_argument("--list", action="store_true", help="List available versions and exit")
    args = parser.parse_args()

    client = MlflowClient()

    if args.list or not args.version:
        versions = list_versions(client)
        if not versions:
            print("(no versions found)")

        # search_model_versions doesn't populate .aliases (MLflow quirk) - get it
        # from the registered model instead: {alias_name: version_number}.
        alias_by_version = {}
        try:
            for alias_name, version_number in client.get_registered_model(settings.MLFLOW_REGISTERED_MODEL_NAME).aliases.items():
                alias_by_version.setdefault(version_number, []).append(alias_name)
        except Exception:
            pass

        for v in versions:
            run = client.get_run(v.run_id)
            ndcg = run.data.metrics.get("ndcg_at_10")
            aliases = ", ".join(alias_by_version.get(v.version, [])) or "-"
            print(f"version={v.version}  ndcg_at_10={ndcg}  aliases=[{aliases}]  run_id={v.run_id}")
        if not args.version:
            return

    from build_recommendation_matrices import bridge_to_legacy_latest

    version = client.get_model_version(settings.MLFLOW_REGISTERED_MODEL_NAME, str(args.version))

    client.set_registered_model_alias(settings.MLFLOW_REGISTERED_MODEL_NAME, "champion", args.version)
    bridge_to_legacy_latest(S3Service(), version.run_id)

    print(f"✓ Rolled back production model to version {args.version} (run_id={version.run_id})")


if __name__ == "__main__":
    main()
