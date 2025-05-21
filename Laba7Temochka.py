import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt

# === Параметры ===
n0       = 4
eps_list = [1e-2, 1e-3, 1e-4]

# --- меню задач ---
print("Выберите задачу:")
print("1) y' = x - y/4,         y(1.4)=2.2")
print("2) y' = cos(y)/(2+x) + 0.3*y^2,   y(0)=0")
choice = input("Номер задачи [1-2]: ").strip()

if choice == "1":
    a, y0 = 1.4, 2.2
    b = 4.0
    def f(x, y): return x - y/4
    C = (y0 - (4*a - 16))*math.exp(a/4)
    def y_exact(x): return 4*x - 16 + C * np.exp(-x/4)
else:
    a, y0 = 0.0, 0.0
    b = float(input("Введите правую границу b: "))
    def f(x, y): return math.cos(y)/(2+x) + 0.3*y*y
    y_exact = None

# === 1) Явный метод Эйлера (p=1) ===
def euler_explicit(f, a, b, y0, n):
    """
    Явный Эйлер:
      y_{i+1} = y_i + h * f(x_i, y_i)
    """
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0
    for i in range(n):
        ys[i+1] = ys[i] + h * f(xs[i], ys[i])
    return xs, ys

# === 2) RK5 (p=5) ===
def runge_kutta5(f, a, b, y0, n):
    """
    Рунге–Кутта 5-го порядка (6 стадий)
    """
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0
    for i in range(n):
        xi, yi = xs[i], ys[i]
        k1 = h*f(xi, yi)
        k2 = h*f(xi+0.5*h, yi+0.5*k1)
        k3 = h*f(xi+0.5*h, yi+0.25*(k1+k2))
        k4 = h*f(xi+  h,   yi   -  k2 +2*k3)
        k5 = h*f(xi+2/3*h, yi + (7*k1+10*k2+k4)/27)
        k6 = h*f(xi+0.2*h, yi + (28*k1-125*k2+546*k3+54*k4-378*k5)/625)
        ys[i+1] = yi + (1/24)*k1 + (5/48)*k4 + (27/56)*k5 + (125/336)*k6
    return xs, ys

# === 3) Явный Adams–Bashforth 2-step (p=2) по формуле (4.17) ===
def adams_bashforth2(f, a, b, y0, n):
    """
    Явный AB2:
      y_{i+1} = y_i + h*(3/2 f_i - 1/2 f_{i-1})
    """
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0
    # старт: один шаг RK5 для y1
    _, y1 = runge_kutta5(f, a, a+h, y0, 1)
    ys[1] = y1[-1]
    for i in range(1, n):
        ys[i+1] = ys[i] + h*(1.5*f(xs[i], ys[i]) - 0.5*f(xs[i-1], ys[i-1]))
    return xs, ys

# === 4) Неявный Adams–Moulton 2-step (p=2) по формуле (4.17) + секущие ===
def adams_moulton2_secant(f, a, b, y0, n, eps):
    """
    Неявный AM2:
      y_{i+1} = y_i + (h/12)*(5 f_{i+1} + 8 f_i - f_{i-1})
    решается методом секущих с tol=eps/10
    """
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0

    # старт: два шага RK5 для y1, y2
    _, y_init = runge_kutta5(f, a, a+2*h, y0, 2)
    ys[1], ys[2] = y_init[1], y_init[2]

    tol = eps * 0.1
    for i in range(2, n):
        x_nm1, x_n, x_np1 = xs[i-1], xs[i], xs[i+1]
        y_nm1, y_n       = ys[i-1], ys[i]
        f_nm1 = f(x_nm1, y_nm1)
        f_n   = f(x_n,   y_n)

        def G(Y):
            return Y - y_n - (h/12)*(5*f(x_np1, Y) + 8*f_n - f_nm1)

        y_pred = y_n + h*(1.5*f_n - 0.5*f_nm1)
        x0, x1 = y_n, y_pred
        f0, f1 = G(x0), G(x1)

        # секущие
        for _ in range(50):
            if abs(f1 - f0) < 1e-16:
                break
            x2 = x1 - f1*(x1 - x0)/(f1 - f0)
            if abs(x2 - x1) < tol:
                x1 = x2
                break
            x0, f0, x1, f1 = x1, f1, x2, G(x2)

        ys[i+1] = x1

    return xs, ys

# === Поиск минимального n по правилу Рунге ===
def find_min_n(method, order, eps):
    n = n0
    while True:
        if method is adams_moulton2_secant:
            _, y_n  = method(f, a, b, y0, n,   eps)
            _, y_2n = method(f, a, b, y0, 2*n, eps)
        else:
            _, y_n  = method(f, a, b, y0, n)
            _, y_2n = method(f, a, b, y0, 2*n)
        err = abs(y_n[-1] - y_2n[-1])/(2**order - 1)
        if err <= eps:
            return n
        n *= 2

# === Список методов ===
methods = [
    ("Эйлер явный (p=1)",     euler_explicit,     1),
    ("РК5         (p=5)",     runge_kutta5,       5),
    ("AB2 явный  (p=2)",      adams_bashforth2,   2),
    ("AM2 неявный(p=2)",       adams_moulton2_secant, 2),
]

# --- 1) Таблица минимальных n ---
table = []
for eps in eps_list:
    row = [eps]
    for name, meth, ord_ in methods:
        row.append(find_min_n(meth, ord_, eps))
    table.append(row)

print(tabulate(table, headers=["ε"] + [m[0] for m in methods], tablefmt="grid"))

# --- 2) График: Exact vs Numerical ---
eps_last = eps_list[-1]
name, meth, ord_ = methods[0]  # по умолчанию первый метод
n_fin = find_min_n(meth, ord_, eps_last)
xs_fin, ys_fin = (meth(f,a,b,y0,n_fin,eps_last)
                  if meth is adams_moulton2_secant
                  else meth(f,a,b,y0,n_fin))

plt.figure()
if y_exact is not None:
    X = np.linspace(a, b, 500)
    plt.plot(X, y_exact(X), color="tab:blue", label="Аналитическое", linewidth=2)
plt.plot(xs_fin, ys_fin, 'o-', color="tab:orange", label=f"{name}, n={n_fin}")
plt.title(f"Exact vs {name}, ε={eps_last}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

# --- 3) График: n0 vs n_final ---
plt.figure()
xs0, ys0 = (meth(f,a,b,y0,n0,eps_last)
            if meth is adams_moulton2_secant
            else meth(f,a,b,y0,n0))
plt.plot(xs0, ys0, 's--', color="tab:green", label=f"{name}, n0={n0}")
plt.plot(xs_fin, ys_fin, 'o-', color="tab:red",   label=f"{name}, n_final={n_fin}")
plt.title(f"{name}: n0 vs n_final")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

plt.show()

# --- 4) Графики для RK5 (аналогично Эйлеру) ---
eps_last = eps_list[-1]
name_rk, meth_rk, ord_rk = methods[1]            # RK5 тот же второй в списке
# находим n_final для RK5
n_fin_rk = find_min_n(meth_rk, ord_rk, eps_last)

# 4.1) Exact vs RK5 (n_final)
xs_rk_fin, ys_rk_fin = meth_rk(f, a, b, y0, n_fin_rk)
plt.figure()
if y_exact is not None:
    X = np.linspace(a, b, 500)
    plt.plot(X, y_exact(X),      color="tab:blue",   label="Аналитич.", linewidth=2)
plt.plot(xs_rk_fin, ys_rk_fin,  color="tab:purple", marker="D", label=f"{name_rk}, n={n_fin_rk}")
plt.title(f"Exact vs {name_rk}, ε={eps_last}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

# 4.2) Сравнение сеток для RK5 (n0 vs n_final)
xs_rk0, ys_rk0 = meth_rk(f, a, b, y0, n0)
plt.figure()
plt.plot(xs_rk0, ys_rk0,        color="tab:gray",   linestyle="--", marker="x", label=f"{name_rk}, n0={n0}")
plt.plot(xs_rk_fin, ys_rk_fin,  color="tab:purple", marker="D",     label=f"{name_rk}, n_final={n_fin_rk}")
plt.title(f"{name_rk}: n0 vs n_final")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

plt.show()
