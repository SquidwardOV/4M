import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# =========================================
#            Часть 1: Подготовка
# =========================================

# Пример нескольких тестовых функций:
def f1(x):
    return np.sin(x)


def f2(x):
    return np.exp(x)


def f3(x):
    return 1 / (1 + x ** 2)


# Можно добавлять и другие функции при необходимости.

# Генерация узлов интерполяции:
def uniform_nodes(a, b, n):
    """ Равномерное разбиение """
    return np.linspace(a, b, n + 1)


def chebyshev_nodes(a, b, n):
    """ Чебышевское разбиение """
    # Формула: x[i] = (a+b)/2 - ((b-a)/2)*cos((2i+1)/(2n+2)*pi)
    i = np.arange(n + 1)
    return (a + b) / 2 - (b - a) / 2 * np.cos((2 * i + 1) / (2 * (n + 1)) * np.pi)


# Построение таблицы (x[i], y[i]):
def generate_table(f, a, b, n, method='uniform'):
    if method == 'uniform':
        x = uniform_nodes(a, b, n)
    elif method == 'chebyshev':
        x = chebyshev_nodes(a, b, n)
    else:
        raise ValueError("Unknown method for node generation.")

    y = f(x)
    return x, y


# =========================================
#    Часть 2.1: Построение интерп. полинома
# =========================================

# 2.2(A): Интерполяционный полином Лагранжа
def lagrange_poly(x_nodes, y_nodes, x):
    """
    Вычисляет значение интерполяционного полинома Лагранжа в точках x.
    x_nodes, y_nodes: массивы узлов интерполяции и значений.
    x: точка (или массив точек), для которых вычисляем Pn(x).
    """
    n = len(x_nodes)
    # Используем формулу Лагранжа:
    # P(x) = sum_{j=0}^{n-1} y_j * l_j(x),
    # где l_j(x) = product_{m=0,m!=j}^{n-1} (x - x_m)/(x_j - x_m)
    # Оптимизация: можно использовать векторизованный подход
    L = np.zeros_like(x, dtype=float)
    for j in range(n):
        # Вычисляем базисный полином l_j(x)
        lj = np.ones_like(x)
        for m in range(n):
            if m != j:
                lj *= (x - x_nodes[m]) / (x_nodes[j] - x_nodes[m])
        L += y_nodes[j] * lj
    return L


# Вспомогательная функция для вычисления таблицы разделенных разностей
def divided_diff(x_nodes, y_nodes):
    """
    Строит таблицу разделенных разностей для узлов x_nodes и значений y_nodes.
    Возвращает массив коэффициентов для Ньютона в форме f[x0], f[x0,x1], ...
    """
    n = len(x_nodes)
    F = np.zeros((n, n))
    F[:, 0] = y_nodes
    for i in range(1, n):
        for j in range(n - i):
            F[j, i] = (F[j + 1, i - 1] - F[j, i - 1]) / (x_nodes[j + i] - x_nodes[j])
    # Коэффициенты в первом ряду — это f[x0], f[x0,x1], f[x0,x1,x2], ...
    return F[0]


# 2.2(A) Интерполяционный полином Ньютона по таблице разделённых разностей
def newton_poly_div_diff(x_nodes, y_nodes, x):
    """
    Вычисляет значение интерполяционного полинома Ньютона,
    используя таблицу разделенных разностей.
    """
    coeff = divided_diff(x_nodes, y_nodes)
    n = len(x_nodes)
    # P(x) = coeff[0] + coeff[1]*(x - x0) + coeff[2]*(x - x0)(x - x1) + ...
    # Последовательно вычисляем значение
    val = coeff[0]
    for k in range(1, n):
        term = coeff[k]
        for j in range(k):
            term *= (x - x_nodes[j])
        val += term
    return val


# 2.2(A) Интерполяционный полином Ньютона через коэффициенты d[i]
# По сути, d[i] - это те же разделённые разности, но отдельно можно реализовать.
def newton_coefficients(x_nodes, y_nodes):
    """
    Вычисление коэффициентов d[i] полинома Ньютона.
    d[i] - последовательные коэффициенты, аналогичные разделенным разностям.
    """
    return divided_diff(x_nodes, y_nodes)


def newton_poly_from_coeff(x_nodes, d, x):
    """
    Вычисляет значение полинома Ньютона, если уже известны коэффициенты d[i].
    d - то, что вернул newton_coefficients
    """
    n = len(x_nodes)
    val = d[0]
    for k in range(1, n):
        term = d[k]
        for j in range(k):
            term *= (x - x_nodes[j])
        val += term
    return val


# =========================================
#   2.3: Вычисление максимальной ошибки
# =========================================

def interpolation_error(f, x_nodes, y_nodes, poly_func, a, b, num_points=1000):
    """
    Вычисляет максимальную по модулю погрешность интерполяции на отрезке [a,b].
    f - исходная функция
    (x_nodes, y_nodes) - таблица значений
    poly_func - функция, вычисляющая интерполяционный полином в заданных точках
    a,b - отрезок проверки
    num_points - количество точек для оценки ошибки
    """
    test_x = np.linspace(a, b, num_points)
    f_val = f(test_x)
    p_val = poly_func(x_nodes, y_nodes, test_x)
    err = np.abs(f_val - p_val)
    return np.max(err)


# =========================================
#        Часть 3: Вычислительный эксперимент
# =========================================

def experiment(f_list, a, b, degrees, k_points=50):
    """
    Исследовать зависимость ошибки интерполяции от степени полинома для
    равномерного и чебышевского разбиений.

    f_list: список функций для исследования
    a,b: интервал интерполяции
    degrees: список степеней полинома (n)
    k_points: количество точек для подсчёта ошибки (разбиение отрезка [a,b])

    Возвращает словарь результатов:
    results = {
       f_name: {
         'n': [n1, n2, ...],
         'uniform_error_lagrange': [...],
         'chebyshev_error_lagrange': [...],
         'uniform_error_newton_diff': [...],
         'chebyshev_error_newton_diff': [...],
         'uniform_error_newton_coeff': [...],
         'chebyshev_error_newton_coeff': [...]
       },
       ...
    }
    """

    results = {}
    for func in f_list:
        f_name = func.__name__
        results[f_name] = {
            'n': [],
            'uniform_error_lagrange': [],
            'chebyshev_error_lagrange': [],
            'uniform_error_newton_diff': [],
            'chebyshev_error_newton_diff': [],
            'uniform_error_newton_coeff': [],
            'chebyshev_error_newton_coeff': []
        }

        for n in degrees:
            # Равномерное разбиение
            x_u, y_u = generate_table(func, a, b, n, method='uniform')
            # Чебышевское разбиение
            x_c, y_c = generate_table(func, a, b, n, method='chebyshev')

            # Ошибка для Лагранжа
            ue_lag = interpolation_error(func, x_u, y_u, lagrange_poly, a, b, k_points)
            ce_lag = interpolation_error(func, x_c, y_c, lagrange_poly, a, b, k_points)

            # Ошибка для Ньютона (разделённые разности)
            ue_newt_diff = interpolation_error(func, x_u, y_u, newton_poly_div_diff, a, b, k_points)
            ce_newt_diff = interpolation_error(func, x_c, y_c, newton_poly_div_diff, a, b, k_points)

            # Ошибка для Ньютона (коэффициенты d[i])
            d_u = newton_coefficients(x_u, y_u)
            d_c = newton_coefficients(x_c, y_c)

            # Обертки для удобства
            def newton_from_d_u(xx_nodes, yy_nodes, XX): return newton_poly_from_coeff(x_u, d_u, XX)

            def newton_from_d_c(xx_nodes, yy_nodes, XX): return newton_poly_from_coeff(x_c, d_c, XX)

            ue_newt_coeff = interpolation_error(func, x_u, y_u, newton_from_d_u, a, b, k_points)
            ce_newt_coeff = interpolation_error(func, x_c, y_c, newton_from_d_c, a, b, k_points)

            results[f_name]['n'].append(n)
            results[f_name]['uniform_error_lagrange'].append(ue_lag)
            results[f_name]['chebyshev_error_lagrange'].append(ce_lag)
            results[f_name]['uniform_error_newton_diff'].append(ue_newt_diff)
            results[f_name]['chebyshev_error_newton_diff'].append(ce_newt_diff)
            results[f_name]['uniform_error_newton_coeff'].append(ue_newt_coeff)
            results[f_name]['chebyshev_error_newton_coeff'].append(ce_newt_coeff)
    return results


# =========================================
# Часть 3(B): Построение графиков
# =========================================

def plot_function_and_interpolation(f, x_nodes, y_nodes, poly_func, a, b, plot_name="interp_plot.png"):
    """
    Строит график исходной функции и интерполяционного полинома,
    а также отмечает значения в узлах интерполяции.
    """
    xx = np.linspace(a, b, 500)
    ff = f(xx)
    pp = poly_func(x_nodes, y_nodes, xx)

    plt.figure(figsize=(10, 6))
    plt.plot(xx, ff, label='f(x)', color='blue')
    plt.plot(xx, pp, label='P_n(x)', color='red', linestyle='--')
    plt.scatter(x_nodes, y_nodes, label='Узлы интерп.', color='black', zorder=5)
    plt.title(f"Функция и интерполяционный полином ({f.__name__})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)
    plt.show()


def plot_error_function(f, x_nodes, y_nodes, poly_func, a, b, plot_name="error_plot.png"):
    """
    Строит график ошибки e(x) = |f(x)-P_n(x)| на отрезке [a,b].
    """
    xx = np.linspace(a, b, 500)
    ff = f(xx)
    pp = poly_func(x_nodes, y_nodes, xx)
    ee = np.abs(ff - pp)

    plt.figure(figsize=(10, 6))
    plt.plot(xx, ee, label='|f(x)-P_n(x)|', color='green')
    plt.title(f"Ошибка интерполяции ({f.__name__})")
    plt.xlabel('x')
    plt.ylabel('Error')
    #plt.yscale('log')  # Для удобства можно взять логарифмический масштаб
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_name)
    plt.show()


def print_results(results):
    for fname, data in results.items():
        print(f"Результаты для функции {fname}:\n")

        # Подготовка данных для таблицы
        headers = [
            "n",
            "Uniform_Lagr",
            "Cheb_Lagr",
            "Uniform_NewtDiff",
            "Cheb_NewtDiff",
            "Uniform_NewtCoeff",
            "Cheb_NewtCoeff",
        ]

        rows = []
        for i in range(len(data['n'])):
            row = [
                data['n'][i],
                data['uniform_error_lagrange'][i],
                data['chebyshev_error_lagrange'][i],
                data['uniform_error_newton_diff'][i],
                data['chebyshev_error_newton_diff'][i],
                data['uniform_error_newton_coeff'][i],
                data['chebyshev_error_newton_coeff'][i],
            ]
            rows.append(row)

        # Используем tabulate для красивого форматирования
        table = tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".3e")
        print(table)
        print("\n")

# =========================================
#                 Пример запуска
# =========================================

if __name__ == "__main__":
    # Зададим параметры эксперимента
    a, b = -1, 1
    degrees = [2, 4, 6, 8, 10, 20, 50, 70, 100]  # Степени полинома
    f_list = [f1, f2, f3]  # Несколько функций для теста

    # Проведём эксперимент
    results = experiment(f_list, a, b, degrees, k_points=200)

    # Выводим результаты в консоль
    print_results(results)

    # Демонстрация построения графиков для одной функции
    # Возьмём функцию f1, степень n=10, равномерное разбиение и полином Лагранжа
    n =  10
    x_nodes, y_nodes = generate_table(f1, a, b, n, method='uniform')
    plot_function_and_interpolation(f1, x_nodes, y_nodes, lagrange_poly, a, b, plot_name="f1_lagrange_uniform.png")
    plot_error_function(f1, x_nodes, y_nodes, lagrange_poly, a, b, plot_name="f1_lagrange_uniform_error.png")

    # Аналогично можно построить графики для других методов/функций/степеней.
