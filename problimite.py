import numpy as np
from tridiagonal import tridiagonal

def problimite(h, P, Q, R, a, b, alpha, beta):
    """
    Résout le problème à valeurs aux limites en construisant et en résolvant
    le système tridiagonal.
    
    h : Taille des sous-intervalles
    P, Q, R : Vecteurs des fonctions p(x), q(x), r(x) aux noeuds xi
    a, b : Bornes de l'intervalle [a, b]
    alpha, beta : Conditions aux limites
    """
    N = len(P)
    assert len(Q) == N, "Les vecteurs P, Q, R doivent avoir la même longueur"
    assert len(R) == N, "Les vecteurs P, Q, R doivent avoir la même longueur"
    assert h > 0, "Le pas h doit être positif"
    assert b > a, "L'intervalle [a,b] doit être valide"
    
    D = 2 + Q * h**2
    I = -1 - P[1:] * h / 2
    S = -1 + P[:-1] * h / 2
    
    b_vec = -R * h**2
    b_vec[0] += (1 + P[0] * h / 2) * alpha
    b_vec[-1] += (1 - P[-1] * h / 2) * beta

    y_interieur = tridiagonal(D, I, S, b_vec)

    y = np.zeros(N + 2)
    y[0] = alpha
    y[-1] = beta
    y[1:-1] = y_interieur

    x = np.linspace(a, b, N + 2)
    
    return y, x