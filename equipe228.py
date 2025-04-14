import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

# Définir les bornes et les CL
a, b = 0.9, 1.0
alpha, beta = 0, 0

# Solution exacte
def y_exact(x):
    d = 1.0
    c = 0.4 / d
    return (c - 0.4 / x**2) - (c - 0.4 / d) * (np.log(x) / np.log(0.9))

# Fonctions p(x), q(x), r(x) pour le problème
def generate_PQR(x):
    P = 1 / x
    Q = 0 * x
    R = -1.6 / x**4
    return P, Q, R

# a) Comparaison pour h = 1/30 et h = 1/100
for h in [1/30, 1/100]:
    x = np.arange(a + h, b, h)
    P, Q, R = generate_PQR(x)
    y_num = problimite(h, P, Q, R, a, b, alpha, beta)
    y_ex = y_exact(x)

    plt.plot(x, y_num, label=f"Approx. h={h:.4f}")
    
# Tracer la solution exacte
x_exact = np.linspace(a, b, 1000)
y_exact_vals = y_exact(x_exact)
plt.plot(x_exact, y_exact_vals, 'k--', label="Solution exacte")

plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Comparaison des solutions numériques et exactes")
plt.legend()
plt.grid(True)
plt.show()

# b) Calcul de l'erreur E(h) pour h = 10^-2, ..., 10^-5
hs = [10**(-i) for i in range(2, 6)]
errors = []

for h in hs:
    x = np.arange(a + h, b, h)
    P, Q, R = generate_PQR(x)
    y_num = problimite(h, P, Q, R, a, b, alpha, beta)
    y_ex = y_exact(x)
    E = np.max(np.abs(y_num - y_ex))
    errors.append(E)

# Tracer E(h) en échelle log-log
plt.loglog(hs, errors, 'o-', label="Erreur E(h)")
plt.xlabel("h")
plt.ylabel("Erreur maximale E(h)")
plt.title("Erreur en fonction de h (log-log)")
plt.grid(True, which='both')
plt.legend()
plt.show()
