import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time

def tridiagonal(D, I, S, b):
    """
    Résout le système linéaire Ax = b où A est une matrice tridiagonale.

    D : Diagonale principale
    I : Diagonale inférieure
    S : Diagonale supérieure
    b : Vecteur second membre
    """
    N = len(D)
    A_sparse = scipy.sparse.diags([I, D, S], offsets=[-1, 0, 1], shape=(N, N), format='csr')

    start_time = time.time()
    x = scipy.sparse.linalg.spsolve(A_sparse, b)
    end_time = time.time()

    print(f"Temps de résolution (sparse) : {end_time - start_time:.4f} secondes")
    return x

# Test demandé : comparer avec et sans sparse
if __name__ == "__main__":
    N = 15000
    D = 4 * np.ones(N)
    I = 1 * np.ones(N - 1)
    S = 1 * np.ones(N - 1)
    b = np.random.rand(N)

    # ----- Test sans sparse -----
    print("Test sans sparse (matrice dense)")
    A_dense = np.diag(D) + np.diag(I, k=-1) + np.diag(S, k=1)

    start_time = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    end_time = time.time()
    print(f"Temps de résolution (dense) : {end_time - start_time:.4f} secondes\n")

    # ----- Test avec sparse -----
    print("Test avec sparse")
    tridiagonal(D, I, S, b)