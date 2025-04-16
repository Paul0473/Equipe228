import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time

def tridiagonal(D, I, S, b):
    """Résout Ax = b pour A tridiagonale. Retourne x."""
    N = len(D)
    A_sparse = scipy.sparse.diags([I, D, S], offsets=[-1, 0, 1], shape=(N, N), format='csr')
    start_time = time.time()
    x = scipy.sparse.linalg.spsolve(A_sparse, b)
    print(f"[Tridiagonal] Temps (sparse) : {time.time() - start_time:.4f} sec")
    return x

if __name__ == "__main__":
    N = 15000
    D = 4 * np.ones(N)
    I = np.ones(N-1)
    S = np.ones(N-1)
    b = np.random.rand(N)

    # Dense
    A_dense = np.diag(D) + np.diag(I, k=-1) + np.diag(S, k=1)
    start = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    print(f"Dense: {time.time() - start:.3f} sec")

    # Sparse
    start = time.time()
    x_sparse = tridiagonal(D, I, S, b)
    print(f"Sparse: {time.time() - start:.3f} sec")

    # Comparaison dense/sparse
    erreur = np.max(np.abs(x_dense - x_sparse))
    print(f"Erreur max entre dense/sparse: {erreur:.2e}")
