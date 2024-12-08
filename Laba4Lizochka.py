import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


# =========================================
# Раздел 1: Инициализация
# =========================================

# Определение тестовых функций:
def func_sin(t):
    return np.sin(t)


def func_exp(t):
    return np.exp(t)


def func_rational(t):
    return 1 / (1 + t ** 2)


# Функции для генерации узлов:
def generate_uniform(a_start, a_end, num_points):
    """ Генерация равномерных узлов """
    return np.linspace(a_start, a_end, num_points + 1)


def generate_chebyshev(a_start, a_end, num_points):
    """ Генерация узлов Чебышева """
    indices = np.arange(num_points + 1)
    return (a_start + a_end) / 2 - (a_end - a_start) / 2 * np.cos((2 * indices + 1) / (2 * (num_points + 1)) * np.pi)


# Создание таблицы значений функции:
def create_table(func, a_start, a_end, num_points, node_type='uniform'):
    if node_type == 'uniform':
        x_vals = generate_uniform(a_start, a_end, num_points)
    elif node_type == 'chebyshev':
        x_vals = generate_chebyshev(a_start, a_end, num_points)
    else:
        raise ValueError("Неверный тип узлов. Используйте 'uniform' или 'chebyshev'.")
    y_vals = func(x_vals)
    return x_vals, y_vals


# =========================================
# Раздел 2: Интерполяция
# =========================================

# Полином Лагранжа:
def compute_lagrange(x_nodes, y_nodes, x_eval):
    total = np.zeros_like(x_eval, dtype=float)
    num_nodes = len(x_nodes)
    for i in range(num_nodes):
        term = y_nodes[i]
        for j in range(num_nodes):
            if j != i:
                term *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        total += term
    return total


# Таблица разделенных разностей:
def compute_divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    for i in range(1, n):
        for j in range(n - i):
            table[j][i] = (table[j + 1][i - 1] - table[j][i - 1]) / (x_nodes[j + i] - x_nodes[j])
    return table[0]


# Полином Ньютона через разделенные разности:
def compute_newton_divided(x_nodes, y_nodes, x_eval):
    coefficients = compute_divided_differences(x_nodes, y_nodes)
    n = len(coefficients)
    result = coefficients[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_eval - x_nodes[i - 1])
        result += coefficients[i] * product
    return result


# Получение коэффициентов для полинома Ньютона:
def get_newton_coefficients(x_nodes, y_nodes):
    return compute_divided_differences(x_nodes, y_nodes)


# Полином Ньютона по коэффициентам:
def evaluate_newton_from_coeff(x_nodes, coeffs, x_eval):
    n = len(coeffs)
    result = coeffs[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_eval - x_nodes[i]) + coeffs[i]
    return result


# =========================================
# Раздел 3: Анализ Ошибок
# =========================================

def calculate_max_error(original_func, x_nodes, y_nodes, interp_func, a_start, a_end, resolution=1000):
    test_points = np.linspace(a_start, a_end, resolution)
    true_values = original_func(test_points)
    approx_values = interp_func(x_nodes, y_nodes, test_points)
    errors = np.abs(true_values - approx_values)
    return np.max(errors)


# =========================================
# Раздел 4: Экспериментальные Исследования
# =========================================

def perform_experiment(functions, a_start, a_end, degrees, test_points=50):
    experiment_results = {}
    for func in functions:
        func_name = func.__name__
        experiment_results[func_name] = {
            'Degree': [],
            'Lagrange_Uniform_Error': [],
            'Lagrange_Chebyshev_Error': [],
            'NewtonDiv_Uniform_Error': [],
            'NewtonDiv_Chebyshev_Error': [],
            'NewtonCoeff_Uniform_Error': [],
            'NewtonCoeff_Chebyshev_Error': []
        }

        for deg in degrees:
            # Равномерные узлы
            x_uni, y_uni = create_table(func, a_start, a_end, deg, node_type='uniform')
            # Чебышевские узлы
            x_cheb, y_cheb = create_table(func, a_start, a_end, deg, node_type='chebyshev')

            # Лагранж
            error_lagrange_uni = calculate_max_error(func, x_uni, y_uni, compute_lagrange, a_start, a_end, test_points)
            error_lagrange_cheb = calculate_max_error(func, x_cheb, y_cheb, compute_lagrange, a_start, a_end,
                                                      test_points)

            # Ньютон через разделенные разности
            error_newton_div_uni = calculate_max_error(func, x_uni, y_uni, compute_newton_divided, a_start, a_end,
                                                       test_points)
            error_newton_div_cheb = calculate_max_error(func, x_cheb, y_cheb, compute_newton_divided, a_start, a_end,
                                                        test_points)

            # Ньютон через коэффициенты
            coeffs_uni = get_newton_coefficients(x_uni, y_uni)
            coeffs_cheb = get_newton_coefficients(x_cheb, y_cheb)

            def newton_uni(z_nodes, w_nodes, z_points):
                return evaluate_newton_from_coeff(x_uni, coeffs_uni, z_points)

            def newton_cheb(z_nodes, w_nodes, z_points):
                return evaluate_newton_from_coeff(x_cheb, coeffs_cheb, z_points)

            error_newton_coeff_uni = calculate_max_error(func, x_uni, y_uni, newton_uni, a_start, a_end, test_points)
            error_newton_coeff_cheb = calculate_max_error(func, x_cheb, y_cheb, newton_cheb, a_start, a_end,
                                                          test_points)

            # Сохранение результатов
            experiment_results[func_name]['Degree'].append(deg)
            experiment_results[func_name]['Lagrange_Uniform_Error'].append(error_lagrange_uni)
            experiment_results[func_name]['Lagrange_Chebyshev_Error'].append(error_lagrange_cheb)
            experiment_results[func_name]['NewtonDiv_Uniform_Error'].append(error_newton_div_uni)
            experiment_results[func_name]['NewtonDiv_Chebyshev_Error'].append(error_newton_div_cheb)
            experiment_results[func_name]['NewtonCoeff_Uniform_Error'].append(error_newton_coeff_uni)
            experiment_results[func_name]['NewtonCoeff_Chebyshev_Error'].append(error_newton_coeff_cheb)

    return experiment_results


# =========================================
# Раздел 5: Визуализация Результатов
# =========================================

def plot_functions(original_func, x_nodes, y_nodes, interp_func, a_start, a_end, filename="interpolation.png"):
    dense_x = np.linspace(a_start, a_end, 500)
    true_y = original_func(dense_x)
    interp_y = interp_func(x_nodes, y_nodes, dense_x)

    plt.figure(figsize=(10, 6))
    plt.plot(dense_x, true_y, label='Исходная функция', color='blue')
    plt.plot(dense_x, interp_y, label='Интерполянт', color='red', linestyle='--')
    plt.scatter(x_nodes, y_nodes, color='black', label='Узлы интерполяции')
    plt.title(f"Функция и Интерполяционный Полином ({original_func.__name__})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_error(original_func, x_nodes, y_nodes, interp_func, a_start, a_end, filename="error.png"):
    dense_x = np.linspace(a_start, a_end, 500)
    true_y = original_func(dense_x)
    interp_y = interp_func(x_nodes, y_nodes, dense_x)
    error = np.abs(true_y - interp_y)

    plt.figure(figsize=(10, 6))
    plt.plot(dense_x, error, label='|f(x) - P_n(x)|', color='green')
    plt.title(f"Ошибка Интерполяции ({original_func.__name__})")
    plt.xlabel('x')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def display_experiment_results(results):
    for func_name, metrics in results.items():
        print(f"Результаты для функции: {func_name}\n")
        headers = [
            "Степень n",
            "Лагранж (Равн.)",
            "Лагранж (Чеб.)",
            "Ньютон DivDiff (Равн.)",
            "Ньютон DivDiff (Чеб.)",
            "Ньютон Coeff (Равн.)",
            "Ньютон Coeff (Чеб.)"
        ]
        table = []
        for i in range(len(metrics['Degree'])):
            row = [
                metrics['Degree'][i],
                f"{metrics['Lagrange_Uniform_Error'][i]:.3e}",
                f"{metrics['Lagrange_Chebyshev_Error'][i]:.3e}",
                f"{metrics['NewtonDiv_Uniform_Error'][i]:.3e}",
                f"{metrics['NewtonDiv_Chebyshev_Error'][i]:.3e}",
                f"{metrics['NewtonCoeff_Uniform_Error'][i]:.3e}",
                f"{metrics['NewtonCoeff_Chebyshev_Error'][i]:.3e}"
            ]
            table.append(row)
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print("\n")


# =========================================
# Раздел 6: Основной Выполнение
# =========================================

if __name__ == "__main__":
    # Параметры интерполяции
    start, end = -1, 1
    polynomial_degrees = [2, 4, 6, 8, 10, 20, 50, 70, 100]
    test_functions = [func_sin, func_exp, func_rational]

    # Проведение эксперимента
    experiment_data = perform_experiment(test_functions, start, end, polynomial_degrees, test_points=200)

    # Вывод результатов
    display_experiment_results(experiment_data)

    # Пример построения графиков для одной функции и одного метода
    example_degree = 10
    x_ex, y_ex = create_table(func_rational, start, end, example_degree, node_type='uniform')
    plot_functions(func_rational, x_ex, y_ex, compute_lagrange, start, end, filename="func_rational_lagrange_uniform.png")
    plot_error(func_rational, x_ex, y_ex, compute_lagrange, start, end, filename="func_rational_lagrange_uniform_error.png")

    # Можно добавить дополнительные примеры построения графиков для других методов и функций по аналогии.
