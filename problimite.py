
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
    x = np.linspace(a, b, N+2)
    
    # Construction des vecteurs D, I, S et b
    D = 2 + Q * h**2
    I = -1 + P * h / 2
    S = -1 - P * h / 2
    b = R * h**2
    b[0] += (1 + P[0] * h / 2) * alpha
    b[-1] += (1 - P[-1] * h / 2) * beta
    
    # Résolution du système
    y_interieur = tridiagonal(D, I, S, b)
    
    y = np.zeros(N+2)
    y[0] = alpha
    y[-1] = beta
    y[1:-1] = y_interieur

    return y, x