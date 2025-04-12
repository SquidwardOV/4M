import math
import numpy as np


# =======================
#  Задача A (одномерный интеграл) – вариант 1 (остается без изменений)
# =======================

def f_variant1(x):
    """
    Интегрируемая функция для варианта 1:
    f(x) = exp(x) + 1/e.
    Здесь a = 0, b = 1.
    """
    return math.exp(x) + 1


def midpt_rule_1d(func, start, end, parts):
    h_val = (end - start) / parts
    total_sum = 0.0
    for i in range(1, parts + 1):
        left = start + (i - 1) * h_val
        right = start + i * h_val
        mid = (left + right) / 2.0
        total_sum += func(mid)
    return h_val * total_sum


def trap_rule_1d(func, start, end, parts):
    h_val = (end - start) / parts
    s_val = (func(start) + func(end)) / 2.0
    for i in range(1, parts):
        s_val += func(start + i * h_val)
    return h_val * s_val


def simp_rule_1d(func, start, end, parts):
    total_sub = 2 * parts
    h_val = (end - start) / total_sub
    summ = func(start) + func(end)
    for i in range(1, total_sub):
        x_val = start + i * h_val
        summ += 2 * func(x_val) if i % 2 == 0 else 4 * func(x_val)
    return (end - start) / (6 * parts) * summ


def refine_1d(method, func, a, b, tolerance, init_parts, order):
    n_val = init_parts
    I_prev = method(func, a, b, n_val)
    while True:
        n_val2 = 2 * n_val
        I_curr = method(func, a, b, n_val2)
        err_est = abs(I_curr - I_prev) / (2 ** order - 1)
        if err_est < tolerance:
            refined = (2 ** order * I_curr - I_prev) / (2 ** order - 1)
            return refined, n_val2
        I_prev = I_curr
        n_val = n_val2


# =======================
#  Задача B (двойной интеграл) – перевод из непрямоугольной области в прямоугольную
# =======================

def f_variant14(x, y):
    """
    Функция для варианта 14:
    f(x,y) = 1/(1+x^2).
    """
    return 1 / (1 + x ** 2)


def triangular_transform(u, v):
    """
    Прямое преобразование из прямоугольной области Q=[0,1]x[0,1] в область
    R = {(x,y): 0 <= x <= 1, x <= y <= 1}.

    """
    x = u
    y = u + v * (1 - u)
    J = 1 - u
    return x, y, J

def inverse_triangular_transform(x, y):
    """
    Обратное преобразование, переводящее точку (x,y) из области
    R = {(x,y): 0 <= x <= 1, x <= y <= 1}
    в точку (u,v) из прямоугольной области Q = [0,1]x[0,1].

    """
    u = x
    v = 0 if x == 1 else (y - x) / (1 - x)
    return u, v


def sequential_trap_2d_transformed(func, transform, Nu, Nv):
    """
    Вычисление двойного интеграла по прямоугольной области Q=[0,1]x[0,1] с использованием
    преобразования, которое переводит Q в исходную область R.

    Если мы задаём равномерную сетку в Q, то точки (u,v) преобразуются в (x,y) с якобианом J.
    Формула:
      I ≈ Δu Δv Σ_{i,j} q_{ij} * f( x(u_i,v_j), y(u_i,v_j) ) * J(u_i,v_j),
    где веса q_{ij} = 1/4 для угловых, 1/2 для граничных (не угловых) и 1 для внутренних узлов.

    transform – функция, принимающая (u,v) и возвращающая (x,y,J).
    """
    Δu = 1.0 / Nu
    Δv = 1.0 / Nv
    total = 0.0
    for i in range(Nu + 1):
        for j in range(Nv + 1):
            u = i * Δu
            v = j * Δv
            x, y, J = transform(u, v)
            if (i == 0 or i == Nu) and (j == 0 or j == Nv):
                weight = 1 / 4
            elif (i == 0 or i == Nu) or (j == 0 or j == Nv):
                weight = 1 / 2
            else:
                weight = 1
            total += weight * func(x, y) * J
    return Δu * Δv * total


def refine_2d_transformed(func, transform, tol, init_N):
    N = init_N
    I_prev = sequential_trap_2d_transformed(func, transform, N, N)
    while True:
        N_new = 2 * N
        I_curr = sequential_trap_2d_transformed(func, transform, N_new, N_new)
        err_est = abs(I_curr - I_prev) / 3.0
        if err_est < tol:
            refined = (4 * I_curr - I_prev) / 3.0
            return refined, N_new
        I_prev = I_curr
        N = N_new


# =======================
# Основной пользовательский интерфейс
# =======================


def main():
    tol = 1e-3
    print("Выберите тип задачи:")
    print("1 --- Одномерный интеграл (вариант 1)")
    print("2 --- Двойной интеграл по непрямоугольной области с приведением к прямоугольной")
    choice = input("Введите 1 или 2: ").strip()

    if choice == "1":
        a_val, b_val = 0.0, 1.0
        print("\nЗадача A (вариант 1):")
        print("Интеграл: I = ∫₀¹ [exp(x) + 1/e] dx")
        print("Точное значение: e")
        print("\nВыберите метод:")
        print("1 --- Формула средних прямоугольников")
        print("2 --- Формула трапеций")
        print("3 --- Формула Симпсона")
        meth_choice = input("Введите 1, 2 или 3: ").strip()
        if meth_choice == "1":
            integrator = midpt_rule_1d
            method_name = "формула средних прямоугольников"
            init_parts = 2
            order = 2
        elif meth_choice == "2":
            integrator = trap_rule_1d
            method_name = "формула трапеций"
            init_parts = 2
            order = 2
        elif meth_choice == "3":
            integrator = simp_rule_1d
            method_name = "формула Симпсона"
            init_parts = 4
            order = 4
        else:
            print("Неверный выбор метода!")
            return

        result, parts_used = refine_1d(integrator, f_variant1, a_val, b_val, tol, init_parts, order)
        exact = math.e
        abs_err = abs(exact - result)
        print("\nРезультаты для задачи A (", method_name, "):")
        print("Приближённое значение интеграла: {:.6f}".format(result))
        print("Число разбиений: {}".format(parts_used))
        print("Абсолютная погрешность: {:.16f}".format(abs_err))
        print("Точное значение интеграла: {:.6f}".format(exact))

    elif choice == "2":
        # Здесь исходная область R задана как: 0 ≤ x ≤ 1, x ≤ y ≤ 1.
        print("\nЗадача B: интегрирование по непрямоугольной области R")
        print("R = {(x,y) : 0 ≤ x ≤ 1, x ≤ y ≤ 1}")
        print("Используется функция f(x,y) = 1/(1+x²)")

        exact2d = math.pi / 4 - 0.5 * math.log(2)
        print("Точное значение интеграла: {:.6f}".format(exact2d))
        init_N = 4
        I2d, N_used = refine_2d_transformed(f_variant14, triangular_transform, tol, init_N)
        abs_err2d = abs(exact2d - I2d)
        print("Приближённое значение двойного интеграла: {:.6f}".format(I2d))
        print("Число разбиений (по каждой оси): {}".format(N_used))
        print("Абсолютная погрешность: {:.10f}".format(abs_err2d))

    else:
        print("Неверный выбор задачи.")


if __name__ == "__main__":
    main()
