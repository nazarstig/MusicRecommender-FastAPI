import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.services.recommendation_service import RecommendationService
from app.core.config import settings


class RecommendationServicePersistenceTests(unittest.TestCase):
    def test_persist_predictions_uploads_to_s3_when_bucket_configured(self):
        service = RecommendationService()
        service.predictions_df = pd.DataFrame(
            [[1.1, 2.2]],
            index=["user_1"],
            columns=["song_a", "song_b"],
        )
        service.U = np.array([[1.0, 0.0], [0.0, 1.0]])
        service.V_T = np.array([[1.0, 0.0], [0.0, 1.0]])
        service.Sigma = np.diag([1.0, 2.0])

        original_bucket = settings.AWS_S3_BUCKET
        original_prefix = settings.AWS_S3_PREFIX
        settings.AWS_S3_BUCKET = "test-bucket"
        settings.AWS_S3_PREFIX = "recommendations"

        with tempfile.TemporaryDirectory() as tmp_dir, patch("app.services.recommendation.boto3.client") as mock_boto_client:
            output_path = os.path.join(tmp_dir, "recommendations_matrices.npz")
            mock_s3_client = mock_boto_client.return_value

            saved_path = service.persist_predictions(output_path=output_path)

            self.assertEqual(saved_path, "s3://test-bucket/recommendations/recommendations_matrices.npz")
            self.assertTrue(os.path.exists(output_path))
            mock_s3_client.upload_file.assert_called_once_with(
                output_path,
                "test-bucket",
                "recommendations/recommendations_matrices.npz",
            )
        settings.AWS_S3_BUCKET = original_bucket
        settings.AWS_S3_PREFIX = original_prefix


if __name__ == "__main__":
    unittest.main()
