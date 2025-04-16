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

#fcts fig 2
h_values = [10**(-k) for k in range(2, 6)]  # h = 1e-2 à 1e-5
errors = []

for h in h_values:
    N = int((b - a)/h) - 1
    x_interior = np.linspace(a + h, b - h, N)
    P = -1 / x_interior
    Q = np.zeros(N)
    R = -1.6 / x_interior**4
    y_num, x_full = problimite(h, P, Q, R, a, b, alpha, beta)
    y_exact = exact_solution(x_full[1:-1])
    error = np.max(np.abs(y_num[1:-1] - y_exact))
    errors.append(error)


log_h = np.log10(h_values)
log_E = np.log10(errors)
slopes = (log_E[1:] - log_E[:-1]) / (log_h[1:] - log_h[:-1])

print("Pentes de convergence entre les points:")
for i, slope in enumerate(slopes):
    print(f"Entre h={h_values[i]:.0e} et h={h_values[i+1]:.0e}: pente = {slope:.4f}")

# Figure 2
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, 'bo-', label='Erreur observée')
plt.loglog(h_values, [errors[0]*(h/h_values[0])**2 for h in h_values], 
         'r--', label='Référence pente')

plt.xlabel('Pas h (échelle log)')
plt.ylabel('Erreur maximale E(h) (échelle log)')
plt.title('Figure 2: Convergence de la méthode (ordre 2 attendu)')
plt.grid(True, which="both", ls="--")
plt.legend()

for i in range(len(slopes)):
    plt.text((h_values[i] + h_values[i+1])/2, (errors[i] + errors[i+1])/2, 
             f'pente={slopes[i]:.2f}', ha='center')

plt.show()
