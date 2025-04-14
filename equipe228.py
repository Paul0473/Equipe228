import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

# Solution exacte
def y_exact(x, c, d):
    return (c - 0.4 / x**2) - (c - 0.4 / d) * np.log(x) / np.log(0.9)

# Trouver c et d de la solution exacte
def find_c_d():
    # Utiliser les conditions aux limites pour déterminer c et d
    # y(0.9) = 0, y(1) = 0
    c = 0.4 / 0.9**2
    d = 0.9
    return c, d

# Tracer la solution
def plot_solution(h_values):
    c, d = find_c_d()
    x_exact = np.linspace(0.9, 1, 1000)
    y_exact_vals = y_exact(x_exact, c, d)
    
    plt.plot(x_exact, y_exact_vals, label='Solution exacte', color='black')
    
    for h in h_values:
        P = 1.0 / np.arange(0.9, 1 + h, h)  # Exemple pour P
        Q = np.ones_like(P)  # Exemple pour Q
        R = np.ones_like(P)  # Exemple pour R
        
        x, y = problimite(h, P, Q, R, 0.9, 1, 0, 0)  # Conditions aux limites: alpha = beta = 0
        plt.plot(x, y, label=f'h = {h}')
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.legend()
    plt.title('Solutions approximées vs Solution exacte')
    plt.show()

# Calcul de l'erreur
def compute_error(h_values):
    c, d = find_c_d()
    x_exact = np.linspace(0.9, 1, 1000)
    y_exact_vals = y_exact(x_exact, c, d)
    
    errors = []
    
    for h in h_values:
        P = 1.0 / np.arange(0.9, 1 + h, h)  # Exemple pour P
        Q = np.ones_like(P)  # Exemple pour Q
        R = np.ones_like(P)  # Exemple pour R
        
        x, y = problimite(h, P, Q, R, 0.9, 1, 0, 0)  # Conditions aux limites: alpha = beta = 0
        y_exact_interp = np.interp(x, x_exact, y_exact_vals)
        error = np.max(np.abs(y - y_exact_interp))
        errors.append(error)
    
    return errors

# Tracer l'erreur
def plot_error():
    h_values = [10**(-i) for i in range(2, 6)]
    errors = compute_error(h_values)
    
    plt.loglog(h_values, errors, marker='o')
    plt.xlabel('h')
    plt.ylabel('E(h)')
    plt.title('Erreur maximale en fonction de h')
    plt.grid(True)
    plt.show()

# Exécution
h_values = [1/30, 1/100]
plot_solution(h_values)

# Tracer l'erreur
plot_error()
