import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt

# === Общие параметры ===
n0       = 4                    # стартовое число шагов (степень двойки)
eps_list = [1e-2, 1e-3, 1e-4]   # список точностей

tol_y    = 1e-6  # допуск по изменению y в неявном методе
tol_f    = 1e-8  # допуск по невязке в неявном методе
max_iter = 20    # макс. число итераций в неявном методе

# === Выбор тестовой функции (интерфейс на русском) ===
print("Выберите задачу:")
print("  1) y' = x + y/2,        y(1.8)=2.6")
print("  2) y' = 1 + 0.2·y·sin(x) - y^2,   y(0)=0")
choice = input("Номер задачи [1-2]: ").strip()

if choice == "1":
    a, y0 = 1.8, 2.6
    b = float(input("Введите правую границу b (для первой задачи): "))
    def f(x, y): return x + 0.5*y
    def y_exact(x): return -2*x + 4.14701 * np.exp(x/2) - 4
    # аналитическая производная по y
    def df_dy(x, y): return 0.5
else:
    a, y0 = 0.0, 0.0
    b = float(input("Введите правую границу b (для второй задачи): "))
    def f(x, y): return 1 + 0.2*y*math.sin(x) - y*y
    y_exact = None  # аналитического решения нет
    # аналитическая производная по y: df/dy = 0.2*sin(x) - 2*y
    def df_dy(x, y): return 0.2*math.sin(x) - 2.0*y

# === Методы ===
def explicit_euler(f, a, b, y0, n):
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0
    for i in range(n):
        ys[i+1] = ys[i] + h * f(xs[i], ys[i])
    return xs, ys


def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x+0.5*h, y+0.5*h*k1)
    k3 = f(x+0.5*h, y+0.5*h*k2)
    k4 = f(x+h,       y+h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def runge_kutta4(f, a, b, y0, n):
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0
    for i in range(n):
        ys[i+1] = rk4_step(f, xs[i], ys[i], h)
    return xs, ys


def adams_bashforth3(f, a, b, y0, n):
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0
    _, y_init = runge_kutta4(f, a, a+2*h, y0, 2)
    ys[1], ys[2] = y_init[1], y_init[2]
    for i in range(3, n+1):
        ys[i] = ys[i-1] + (h/12)*(
            23*f(xs[i-1], ys[i-1])
          - 16*f(xs[i-2], ys[i-2])
          +  5*f(xs[i-3], ys[i-3])
        )
    return xs, ys


def implicit_euler(f, a, b, y0, n):
    """
    Неявный метод Эйлера с решением на каждом шаге методом Ньютона.
    """
    h = (b - a)/n
    xs = np.linspace(a, b, n+1)
    ys = np.empty(n+1); ys[0] = y0

    for i in range(n):
        x_next = xs[i+1]
        y_prev = ys[i]
        # начальное приближение
        y_i = y_prev
        # Ньютоновская итерация
        for _ in range(max_iter):
            F  = y_i - y_prev - h*f(x_next, y_i)
            dF = 1.0 - h*df_dy(x_next, y_i)
            y_new = y_i - F/dF
            # проверка сходимости
            if abs(y_new - y_i) <= tol_y and abs(F) <= tol_f:
                y_i = y_new
                break
            y_i = y_new
        ys[i+1] = y_i

    return xs, ys

# === Поиск минимального n по правилу Рунге ===
def find_min_n(method, order, eps):
    n = n0
    while True:
        _, y_n  = method(f, a, b, y0, n)
        _, y_2n = method(f, a, b, y0, 2*n)
        err = abs(y_n[-1] - y_2n[-1])/(2**order - 1)
        if err <= eps:
            return n
        n *= 2

methods = [
    ("Явный Эйлер (p=1)",    explicit_euler,    1),
    ("РК4 (p=4)",            runge_kutta4,      4),
    ("AB3 (p=3)",           adams_bashforth3,  3),
    ("Неявный Эйлер (p=1)", implicit_euler,    1),
]

# === 1) Таблица минимальных n ===
table = []
for eps in eps_list:
    row = [eps]
    for name, mth, order in methods:
        row.append(find_min_n(mth, order, eps))
    table.append(row)
print(tabulate(table, headers=["ε"]+[m[0] for m in methods], tablefmt="grid"))

# === 2) График: точное vs численное ===
eps_last = eps_list[-1]
name, method, order = methods[1]  # индекс метода
n_final = find_min_n(method, order, eps_last)
xs_num, ys_num = method(f, a, b, y0, n_final)

plt.figure()
if y_exact is not None:
    x_ex = np.linspace(a, b, 500)
    plt.plot(x_ex, y_exact(x_ex), label="Аналитич.", linewidth=2)
plt.plot(xs_num, ys_num, marker="o", label=f"{name}, n={n_final}")
plt.title(f"Задача {choice}: Аналитич. vs {name}, ε={eps_last}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

# === 3) График: сравнение сеток выбранного метода ===
plt.figure()
xs0, ys0 = method(f, a, b, y0, n0)
plt.plot(xs0, ys0, linestyle="--", marker="s", label=f"{name}, n0={n0}")
plt.plot(xs_num, ys_num, marker="o",       label=f"{name}, n_final={n_final}")
plt.title(f"Задача {choice}: Сравнение сеток {name}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

# === 4) График: Явный Эйлер ===
xs_eu, ys_eu = explicit_euler(f, a, b, y0, n_final)
plt.figure()
if y_exact is not None:
    x_ex = np.linspace(a, b, 500)
    plt.plot(x_ex, y_exact(x_ex), label="Аналитич.", linewidth=2)
plt.plot(xs_eu, ys_eu, marker="o", label=f"Явный Эйлер, n={n_final}")
plt.title(f"Задача {choice}: Явный Эйлер, ε={eps_last}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

# === 5) График: сравнение сеток Явного Эйлера ===
plt.figure()
xs_e0, ys_e0 = explicit_euler(f, a, b, y0, n0)
plt.plot(xs_e0, ys_e0, linestyle="--", marker="s", label=f"Явный Эйлер, n0={n0}")
plt.plot(xs_eu, ys_eu, marker="o",       label=f"Явный Эйлер, n_final={n_final}")
plt.title(f"Задача {choice}: Сравнение сеток Явного Эйлера")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()

plt.show()
