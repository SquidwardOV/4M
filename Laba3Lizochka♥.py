import numpy as np
import sys

def generate_random_matrix(n, min_val=-10.0, max_val=10.0):
    """
    Генерирует случайную матрицу размера n x n с элементами в диапазоне [min_val, max_val].
    """
    return np.random.uniform(min_val, max_val, (n, n))

def generate_diagonally_dominant_matrix(n, min_val=-10.0, max_val=10.0):
    """
    Генерирует диагонально доминантную матрицу размера n x n.
    Каждый диагональный элемент строго больше суммы абсолютных значений остальных элементов в строке.
    """
    A = np.random.uniform(min_val, max_val, (n, n))
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        A[i, i] = row_sum + np.abs(np.random.uniform(min_val, max_val)) + 1.0
    return A

def generate_hilbert_matrix(n):
    """
    Генерирует матрицу Гильберта размера n x n.
    Элемент (i, j) = 1 / (i + j + 1)
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1.0 / (i + j + 1)
    return A

def gauss_full_pivoting(A, b):
    """
    Решает систему линейных уравнений Ax = b методом Гаусса с полной стратегией выбора ведущего элемента.
    Возвращает решение x и относительную погрешность.
    """
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = A.shape[0]
    col_swaps = np.arange(n)

    for k in range(n):
        # Поиск максимального элемента в подматрице A[k:, k:]
        sub_matrix = np.abs(A[k:, k:])
        max_idx = np.unravel_index(np.argmax(sub_matrix, axis=None), sub_matrix.shape)
        max_row, max_col = max_idx[0] + k, max_idx[1] + k
        max_val = A[max_row, max_col]

        if max_val == 0:
            raise ValueError("Матрица вырождена!")

        # Перестановка строк
        if max_row != k:
            A[[k, max_row], :] = A[[max_row, k], :]
            b[k], b[max_row] = b[max_row], b[k]

        # Перестановка столбцов
        if max_col != k:
            A[:, [k, max_col]] = A[:, [max_col, k]]
            col_swaps[[k, max_col]] = col_swaps[[max_col, k]]

        # Прямой ход
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if A[i, i] == 0:
            raise ValueError("Матрица вырождена!")
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    # Восстановление порядка переменных
    x_reordered = np.zeros(n)
    for i in range(n):
        x_reordered[col_swaps[i]] = x[i]
    return x_reordered

def computational_experiment(n, matrix_type):
    """
    Выполняет вычислительный эксперимент для матрицы размера n и заданного типа.
    Возвращает относительную погрешность решения.
    """
    if matrix_type == "random":
        A = generate_random_matrix(n)
    elif matrix_type == "diagonally_dominant":
        A = generate_diagonally_dominant_matrix(n)
    elif matrix_type == "hilbert":
        if n > 12:
            print(f"Размер матрицы Гильберта {n} слишком велик и может привести к высокой погрешности.")
        A = generate_hilbert_matrix(n)
    else:
        raise ValueError("Неизвестный тип матрицы!")

    # Генерация точного решения
    x_exact = np.random.uniform(-10.0, 10.0, n)
    # Вычисление вектора правой части
    b = A @ x_exact

    try:
        # Решение системы методом Гаусса с полной стратегией выбора ведущего элемента
        x_approx = gauss_full_pivoting(A, b)
        # Вычисление относительной погрешности
        error = np.linalg.norm(x_approx - x_exact, ord=np.inf) / np.linalg.norm(x_exact, ord=np.inf)
    except ValueError as e:
        print(f"Ошибка при решении системы: {e}")
        error = np.inf

    return error

def main():
    print("Выберите тип матрицы:")
    print("1 - Случайная матрица")
    print("2 - Диагонально доминантная матрица")
    print("3 - Матрица Гильберта")
    try:
        choice = int(input("Введите номер выбора: "))
    except ValueError:
        print("Неверный ввод! Пожалуйста, введите число 1, 2 или 3.")
        sys.exit(1)

    if choice == 1:
        matrix_type = "random"
    elif choice == 2:
        matrix_type = "diagonally_dominant"
    elif choice == 3:
        matrix_type = "hilbert"
    else:
        print("Неверный выбор!")
        sys.exit(1)

    output_filename = f"errors_{matrix_type}.txt"
    with open(output_filename, "w") as outfile:
        print(f"Тип матрицы: {matrix_type}")
        print(f"Результаты будут сохранены в файл: {output_filename}")
        sizes = []
        errors = []
        size = 2
        while size <= 1024:
            error = computational_experiment(size, matrix_type)
            print(f"Размер: {size}, Относительная погрешность: {error}")
            outfile.write(f"{size} {error}\n")
            sizes.append(size)
            errors.append(error)
            size *= 2
          
            

    print("Вычислительный эксперимент завершён.")

if __name__ == "__main__":
    main()
