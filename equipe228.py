import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite


#fcts pour fg 1
def exact_solution(x):
    c = 0.4
    d = 0.81
    return (c - 0.4 / x**2) - (c - 0.4 / d) * (np.log(x) / np.log(0.9))

def p(x): return -1 / x
def q(x): return 0 * x
def r(x): return -1.6 / x**4

a, b = 0.9, 1.0
alpha, beta = 0.0, 0.0

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
    plt.plot(x_full, y, 'o--', markersize=4, label=f'Solution numérique h={h:.4f}')

# Figure 1
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Figure 1: Solution du problème aux limites')
plt.grid()
plt.legend()
plt.show()

#fcts pour fg 2
hs = np.array([1e-2, 1e-3, 1e-4, 1e-5])
errors = []

for h in hs:
    N = int((b - a)/h) - 1
    x_interior = np.linspace(a + h, b - h, N)
    P, Q, R = p(x_interior), q(x_interior), r(x_interior)
    y, x_full = problimite(h, P, Q, R, a, b, alpha, beta)
    y_exact_values = exact_solution(x_full)
    err = np.max(np.abs(y - y_exact_values))
    errors.append(err)

errors = np.array(errors)

# Figure 2 
plt.figure(figsize=(10, 6))

plt.loglog(hs, errors, 'bo-', linewidth=2, markersize=8, 
           label='Erreur maximale $E(h)$')

plt.loglog(hs, 10*errors[0]*(hs/hs[0])**2, 'r--', 
           label='Référence $O(h^2)$')

plt.xlabel('Pas de discrétisation $h$', fontsize=12)
plt.ylabel('Erreur $E(h)$', fontsize=12)
plt.title('Figure 2: Convergence de la méthode : $E(h) = max|y_i - y(x_i)|$', 
          fontsize=14)

for i, (h, err) in enumerate(zip(hs, errors)):
    plt.text(h, err, f'h={h:.0e}\nE={err:.1e}', 
             ha='center', va='bottom' if i%2 else 'top')

plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()