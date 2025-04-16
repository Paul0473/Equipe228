import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

# Solution exacte corrigée
def exact_solution(x):
    c = 0.4
    d = 1
    return (c - 0.4 / x**2) - (c - 0.4 / d) * (np.log(x) / np.log(0.9))

# Fonctions du problème
def p(x): return -1 / x
def q(x): return 0 * x
def r(x): return -1.6 / x**4

# Paramètres
a, b = 0.9, 1.0
alpha, beta = 0.0, 0.0

# --- a) Solutions numériques et tracé ---
h_values = [1/30, 1/100]
plt.figure(figsize=(10, 6))

x_exact = np.linspace(a, b, 1000)
y_exact = exact_solution(x_exact)
plt.plot(x_exact, y_exact, 'k-', label='Solution exacte')

for h in h_values:
    N = int((b - a)/h) - 1
    x_interior = np.linspace(a + h, b - h, N)
    P, Q, R = p(x_interior), q(x_interior), r(x_interior)
    y, x_full = problimite(h, P, Q, R, a, b, alpha, beta)
    plt.plot(x_full, y, 'o--', label=f'Solution numérique h={h:.4f}')

plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Solution du problème aux limites')
plt.grid()
plt.legend()
plt.show()

# --- b) Erreurs pour différentes valeurs de h ---
hs = [1e-2, 1e-3, 1e-4, 1e-5]
errors = []

for h in hs:
    N = int((b - a)/h) - 1
    x_interior = np.linspace(a + h, b - h, N)
    P, Q, R = p(x_interior), q(x_interior), r(x_interior)
    y, x_full = problimite(h, P, Q, R, a, b, alpha, beta)
    y_exact = exact_solution(x_full)
    err = np.max(np.abs(y - y_exact))
    errors.append(err)

# Tracé de l'erreur en log-log
plt.figure(figsize=(10, 6))
plt.loglog(hs, errors, 'o-', label='Erreur maximale')
plt.xlabel('h')
plt.ylabel('Erreur E(h)')
plt.title('Erreur max vs pas de discrétisation (log-log)')
plt.grid()
plt.legend()
plt.show()
