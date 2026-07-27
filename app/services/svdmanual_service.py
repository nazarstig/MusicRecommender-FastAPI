from click import Tuple
import numpy as np
import pandas as pd

class SVDManualService:
    def power_method(
            self,
            matrix: np.ndarray,
            max_iter: int = 1000,
            tol: float = 1e-8,
        ) -> Tuple[float, np.ndarray]:
        # Explicit power iteration for dominant eigenpair of a symmetric matrix.
        n = matrix.shape[0]
        rng = np.random.default_rng(42)
        v = rng.normal(size=n)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            raise ValueError("Initial vector norm is zero in power method.")
        v = v / v_norm
    
        prev_lambda = 0.0
        for _ in range(max_iter):
            w = matrix @ v
            w_norm = np.linalg.norm(w)
            if w_norm < tol:
                return 0.0, v
    
            v_next = w / w_norm
            lambda_est = float(v_next.T @ matrix @ v_next)
    
            if abs(lambda_est - prev_lambda) < tol and np.linalg.norm(v_next - v) < tol:
                v = v_next
                prev_lambda = lambda_est
                break
    
            v = v_next
            prev_lambda = lambda_est
    
        return prev_lambda, v
    
    def manual_svds(
        self,
        matrix: np.ndarray,
        k: int,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute top-k singular values/vectors via eigen-decomposition of A^T A.
        m, n = matrix.shape
        if m == 0 or n == 0:
            raise ValueError("Cannot compute SVD on an empty matrix.")
    
        max_k = min(m, n)
        k = max(1, min(k, max_k))
    
        gram = matrix.T @ matrix
        gram = (gram + gram.T) / 2.0
    
        singular_values = []
        right_vectors = []
    
        for _ in range(k):
            eigval, v = self.power_method(gram, max_iter=max_iter, tol=tol)
            if eigval <= tol:
                break
    
            sigma = float(np.sqrt(eigval))
            singular_values.append(sigma)
            right_vectors.append(v)
    
            # Deflation removes the extracted dominant component.
            gram = gram - eigval * np.outer(v, v)
            gram = (gram + gram.T) / 2.0
    
        if not singular_values:
            raise ValueError("No singular values were extracted by manual SVD.")
    
        s = np.array(singular_values)
        vt = np.array(right_vectors)
    
        u_columns = []
        for i, sigma in enumerate(s):
            v = vt[i]
            u_col = matrix @ v
            u_norm = np.linalg.norm(u_col)
            if u_norm > tol:
                u_col = u_col / u_norm
            else:
                u_col = np.zeros_like(u_col)
            u_columns.append(u_col)
    
            u = np.column_stack(u_columns)
            return u, s, vt
        
    def create_recommendations_2(self, db: Session):
        user_item_matrix = self.create_user_item_matrix(db)
        dense_matrix = user_item_matrix.values.astype(float)
    
        k = min(200, dense_matrix.shape[0], dense_matrix.shape[1])
        u, s, vt = self.manual_svds(dense_matrix, k=k)
        sigma = np.diag(s)
    
        self.V_T = vt
        self.Sigma = sigma
    
        all_user_predicted_ratings = np.dot(np.dot(u, sigma), vt)
        self.predictions_df = pd.DataFrame(
            all_user_predicted_ratings,
            columns=user_item_matrix.columns,
            index=user_item_matrix.index,
        )
    
        return self.predictions_df.head()