import numpy as np
import matplotlib.pyplot as plt

# Исходные функции
def build_table(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return x, y

def build_spline(x, y, cond_type, A=0, B=0):
    n = len(x) - 1
    h = np.diff(x)

    alpha = np.zeros(n + 1)
    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    if cond_type == 'clamped':
        # Если A и B являются функциями, вычисляем их в крайних точках
        A_val = A(x[0]) if callable(A) else A
        B_val = B(x[-1]) if callable(B) else B
        alpha[0] = 3 * ((y[1] - y[0]) / h[0] - A_val)
        alpha[n] = 3 * (B_val - (y[n] - y[n-1]) / h[n-1])

    for i in range(1, n):
        alpha[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    if cond_type == 'natural':
        l[0] = 1
        mu[0] = 0
        z[0] = 0
    elif cond_type == 'clamped':
        l[0] = 2 * h[0]
        mu[0] = 0.5
        z[0] = alpha[0] / l[0]

    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    if cond_type == 'natural':
        l[n] = 1
        z[n] = 0
    elif cond_type == 'clamped':
        l[n] = 1
        mu[n] = 0
        z[n] = alpha[n] / l[n]

    c = np.zeros(n + 1)
    b_coeff = np.zeros(n)
    d = np.zeros(n)

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b_coeff[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    return y, b_coeff, c, d

def evaluate_spline(x_val, x, y, b, c, d):
    i = np.searchsorted(x, x_val) - 1
    i = np.clip(i, 0, len(x)-2)
    dx = x_val - x[i]
    return y[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

def compute_interpolation_error(f, a, b, x, y, b_coeff, c_coeff, d_coeff, k=100):
    x_fine = np.linspace(a, b, len(x) * k)
    y_true = f(x_fine)
    y_spline = np.array([evaluate_spline(xx, x, y, b_coeff, c_coeff, d_coeff) for xx in x_fine])
    error = np.abs(y_true - y_spline)
    max_error = np.max(error)
    return max_error, x_fine, error

# Пользовательский интерфейс
if __name__ == "__main__":
    print("Выберите функцию для интерполяции:")
    print("1: sin(2x)")
    print("2: Многочлен 4-й степени (x^4 - 3x^3 + 2x^2 - x + 1)")
    print("3: |x| на симметричном промежутке")
    func_choice = input("Введите номер функции (1, 2 или 3): ")

    if func_choice == "1":
        f = lambda x: np.sin(2*x)
    elif func_choice == "2":
        f = lambda x: x**4 - 3*x**3 + 2*x**2 - x + 1
    elif func_choice == "3":
        f = lambda x: np.abs(x)
    else:
        print("Неверный выбор, используется sin(2x) по умолчанию.")
        f = lambda x: np.sin(2*x)

    print("Выберите тип краевых условий:")
    print("1: Natural")
    print("2: Clamped")
    cond_choice = input("Введите номер краевого условия (1 или 2): ")

    a, b = 0, np.pi  # интервал интерполяции
    if cond_choice == "1":
        cond_type = 'natural'
        A, B = 0, 0
    elif cond_choice == "2":
        cond_type = 'clamped'
        # Здесь сразу запрашиваем функции для производной
        A_str = input("Введите функцию для производной в левой точке: ")
        B_str = input("Введите функцию для производной в правой точке: ")
        # Преобразуем строку в функцию через eval
        A_func = lambda x: eval(A_str, {"x": x, "np": np})
        B_func = lambda x: eval(B_str, {"x": x, "np": np})
        A, B = A_func, B_func
    else:
        print("Неверный выбор, используются естественные условия по умолчанию.")
        cond_type = 'natural'
        A, B = 0, 0

    # Список количеств узлов для расчёта
    N_values = [10, 20, 40, 60, 100, 500]

    print("\nТаблица погрешностей интерполяции:")
    print("N\tMax Error")
    last_results = None
    for N in N_values:
        x_nodes, y_nodes = build_table(f, a, b, N)
        y_vals, b_coeff, c_coeff, d_coeff = build_spline(x_nodes, y_nodes, cond_type, A, B)
        max_error, x_fine, error = compute_interpolation_error(f, a, b, x_nodes, y_vals, b_coeff, c_coeff, d_coeff)
        print(f"{N}\t{max_error:.6e}")
        last_results = (N, x_nodes, y_nodes, y_vals, b_coeff, c_coeff, d_coeff, x_fine, error)

    # Построение графиков для последнего значения N
    if last_results is not None:
        N_last, x_nodes, y_nodes, y_vals, b_coeff, c_coeff, d_coeff, x_fine, error = last_results

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_fine, f(x_fine), label='f(x)', color='blue')
        plt.plot(x_fine, [evaluate_spline(xx, x_nodes, y_vals, b_coeff, c_coeff, d_coeff) for xx in x_fine],
                 '--', label='Spline S(x)', color='green')
        plt.scatter(x_nodes, y_nodes, color='red', label='Узлы')
        plt.title(f"Cubic Spline, N={N_last}, Condition: {cond_type}")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(x_fine, error, label='|f(x) - S(x)|', color='purple')
        plt.title("Ошибка интерполяции сплайна")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
