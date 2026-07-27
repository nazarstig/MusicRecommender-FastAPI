import os

import numpy as np

class FileService:
    def __init__(self):
        self.filepath = "recommendations_manual.npz"

    def save_matrices(self):
        np.savez(
            self.filepath,
            U=self.U,
            V_T=self.V_T,
            Sigma=self.Sigma,
        )

    def load_matrices(
            self
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.filepath is None:
            raise ValueError("File path is not provided")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Prediction matrices file not found: {self.filepath}"
            )

        matrices = np.load(self.filepath)
        if matrices is None or not all(
            key in matrices for key in ["V_T", "Sigma", "U"]
        ):
            return None, None, None
        
        return matrices["U"], matrices["V_T"], matrices["Sigma"]
