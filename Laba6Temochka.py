import math
import numpy as np


# =======================
# Функции для задачи A (одномерный интеграл)
# =======================

def f_A(x):
    """Интегрируемая функция для задачи A: f(x) = exp(x) + 1/x."""
    return math.exp(x) + 1 / x


def midpoint_integral(f, a, b, n):
    """
    Формула средних прямоугольников:
      ∫[a..b] f(x) dx ≈ (b-a)/n * Σ{ i=1..n } f( (x_{i-1}+x_i)/2 ),
    где x_i = a + i*(b-a)/n.
    """
    dx = (b - a) / n
    total = 0.0
    for i in range(1, n + 1):
        x_left = a + (i - 1) * dx
        x_right = a + i * dx
        x_mid = (x_left + x_right) / 2
        total += f(x_mid)
    return dx * total


def trapezoidal_integral(f, a, b, n):
    """
    Композитная формула трапеций:
      ∫[a..b] f(x) dx ≈ (b-a)/n * ( (f(a)+f(b))/2 + Σ{ i=1,..,n-1 } f( a + i*(b-a)/n ) ).
    """
    dx = (b - a) / n
    y0 = f(a)
    y_n = f(b)
    s = 0.0
    for i in range(1, n):
        x_i = a + i * dx
        s += f(x_i)
    return dx * ((y0 + y_n) / 2 + s)


def simpson_integral(f, a, b, n):
    """
    Композитная формула Симпсона в виде:
      ∫[a..b] f(x) dx ≈ (b-a)/(6*n) * [ f(x_0) + f(x_{2n})
            + 4( f(x_1) + f(x_3) + ... + f(x_{2n-1}) )
            + 2( f(x_2) + f(x_4) + ... + f(x_{2n-2}) ) ],
    где x_k = a + k*(b-a)/(2n) и n — число "половинных" отрезков
         (общее число подотрезков = 2*n).
    """
    N = 2 * n  # общее число подотрезков
    h = (b - a) / N
    total = f(a) + f(b)

    sum_odd = 0.0  # для k = 1, 3, ..., N-1
    for k in range(1, N, 2):
        sum_odd += f(a + k * h)

    sum_even = 0.0  # для k = 2, 4, ..., N-2
    for k in range(2, N, 2):
        sum_even += f(a + k * h)

    total = total + 4 * sum_odd + 2 * sum_even
    return (b - a) / (6 * n) * total


def iterative_integration(method, f, a, b, eps, initial_n, k):
    """
    Итеративное уточнение вычисления интеграла по правилу Рунге.

    Пусть I_prev — значение интеграла при n разбиениях, а I_curr — при 2n разбиениях.
    Оценка погрешности по правилу Рунге:
         err_est = |I_curr - I_prev| / (2^k - 1)
    Если err_est < eps, то процесс останавливается, иначе число разбиений удваивается.

    Параметры:
      - method: функция вычисления интеграла (midpoint, trapezoidal, simpson)
      - k: порядок точности выбранного метода (2 или 4)
    """
    n = initial_n
    I_prev = method(f, a, b, n)
    while True:
        n2 = 2 * n
        I_curr = method(f, a, b, n2)
        err_est = abs(I_curr - I_prev) / (2 ** k - 1)
        # Вывод для отладки:
        # print(f"n = {n2}, I_curr = {I_curr:.6f}, I_prev = {I_prev:.6f}, err_est = {err_est:.6f}")
        if err_est < eps:
            return I_curr, n2
        I_prev = I_curr
        n = n2


# =======================
# Функции для задачи B (двойной интеграл)
# =======================

def f_B(x, y):
    """Интегрируемая функция для задачи B: f(x,y) = x * ln(x*y)."""
    return x * math.log(x * y)


def double_integral_cell_method(f, a, b, phi1, phi2, m):
    """
    Метод ячеек для двойного интеграла.

    Область интегрирования: x ∈ [a, b] и y ∈ [phi1(x), phi2(x)].
    Для упрощения используем ограничивающий прямоугольник:
         x ∈ [a, b], y ∈ [phi1(a), phi2(b)].
    Разбиваем его на m×m ячеек и суммируем вклад ячеек, центры которых удовлетворяют условию.
    """
    min_y = phi1(a)
    max_y = phi2(b)
    dx = (b - a) / m
    dy = (max_y - min_y) / m
    s = 0.0
    for i in range(m):
        x_center = a + (i + 0.5) * dx
        for j in range(m):
            y_center = min_y + (j + 0.5) * dy
            if phi1(x_center) <= y_center <= phi2(x_center):
                s += f(x_center, y_center)
    return s * dx * dy


def iterative_double_integral(f, a, b, phi1, phi2, eps, initial_m):
    """
    Итеративное уточнение двойного интеграла методом ячеек.
    Увеличиваем число делений m (по каждой оси) до тех пор, пока |I_curr - I_prev| не станет меньше eps.
    """
    m = initial_m
    I_prev = double_integral_cell_method(f, a, b, phi1, phi2, m)
    while True:
        m2 = 2 * m
        I_curr = double_integral_cell_method(f, a, b, phi1, phi2, m2)
        if abs(I_curr - I_prev) < eps:
            return I_curr, m2
        I_prev = I_curr
        m = m2


def phi1(x):
    """Нижняя граница по y: phi1(x) = x."""
    return x


def phi2(x):
    """Верхняя граница по y: phi2(x) = 2x."""
    return 2 * x


# =======================
# Основная программа
# =======================

def main():
    eps = 0.0001  # требуемая точность
    print("Выберите задачу для решения:")
    print("A - Численные методы вычисления определённого интеграла")
    print("B - Численные методы вычисления двойного интеграла")
    choice = input("Введите A или B: ").strip().upper()

    if choice == "A":
        # Задача A: вариант 6
        a, b = 1, 2
        print("\nЗадача A (вариант 6):")
        print("Параметры: a = 1, b = 2, f(x) = exp(x) + 1/x")
        print("Точный ответ: exp(e-1) + ln2")
        print("\nВыберите метод вычисления:")
        print("1 - Формула средних прямоугольников")
        print("2 - Формула трапеций")
        print("3 - Формула Симпсона")
        method_choice = input("Введите 1, 2 или 3: ").strip()

        if method_choice == "1":
            method_func = midpoint_integral
            method_name = "формула средних прямоугольников"
            initial_n = 2
            k = 2  # порядок точности для этого метода
        elif method_choice == "2":
            method_func = trapezoidal_integral
            method_name = "формула трапеций"
            initial_n = 2
            k = 2
        elif method_choice == "3":
            method_func = simpson_integral
            method_name = "формула Симпсона"
            initial_n = 4
            k = 4
        else:
            print("Неверный выбор метода.")
            return

        I, n_used = iterative_integration(method_func, f_A, a, b, eps, initial_n, k)
        exact_value = exact_value = math.exp(2) - math.exp(1) + math.log(2)  # точное значение интеграла
        error = abs(exact_value - I)

        print("\nРезультаты для задачи A ({}):".format(method_name))
        print("Приближённое значение интеграла: {:.6f}".format(I))
        print("Число разбиений: {}".format(n_used))
        print("Абсолютная погрешность: {:.6f}".format(error))
        print("Точное значение интеграла: {:.6f}".format(exact_value))

    elif choice == "B":
        # Задача B: вариант 9
        print("\nЗадача B (вариант 9):")
        print("Параметры: a = 1, b = 2,")
        print("phi1(x) = x,  phi2(x) = 2x,  f(x,y) = x * ln(x*y)")
        print("Точный ответ: 10*ln2 - 35/9")
        m_initial = 10
        I, m_used = iterative_double_integral(f_B, 1, 2, phi1, phi2, eps, m_initial)
        exact_value = 10 * math.log(2) - 35 / 9
        error = abs(exact_value - I)

        print("\nРезультаты для задачи B (метод ячеек):")
        print("Приближённое значение двойного интеграла: {:.6f}".format(I))
        print("Число ячеек по оси x (и y): {}".format(m_used))
        print("Абсолютная погрешность: {:.6f}".format(error))
        print("Точное значение двойного интеграла: {:.6f}".format(exact_value))
    else:
        print("Неверный выбор задачи.")


if __name__ == "__main__":
    main()
