import math
import random
import statistics
from tabulate import tabulate  # <-- не забудьте установить библиотеку

#---------------------------------------------------------------------------
class NonZeroRandomProvider:
    """
    Возвращает случайное число в диапазоне [min_val, max_val],
    избегая при этом попадания ровно в 0, если 0 лежит внутри заданного диапазона.
    """
    def __init__(self, non_zero_eps=1e-5):
        self.non_zero_eps = non_zero_eps

    def next_value(self, min_val: float, max_val: float) -> float:
        """
        Возвращает псевдослучайное число из [min_val, max_val],
        при необходимости избегая нуля (если 0 внутри диапазона).
        """
        while True:
            val = random.uniform(min_val, max_val)
            # Если случайно попали ровно в окрестность нуля — попробуем снова.
            if abs(val) < self.non_zero_eps and min_val < 0 < max_val:
                continue
            return val

# ------------------------------------------------------------------------------
# 2. Генерация матрицы (различные типы)
# ------------------------------------------------------------------------------

def generate_matrix(rng: NonZeroRandomProvider,
                    rows_count: int,
                    columns_count: int,
                    min_value: float,
                    max_value: float) -> list[list[float]]:
    if rng is None:
        raise ValueError("rng cannot be None")
    if rows_count < 0:
        raise ValueError("rows_count must not be negative.")
    if columns_count < 0:
        raise ValueError("columns_count must not be negative.")

    # Инициализируем матрицу нулями.
    matrix = [[0.0 for _ in range(columns_count)] for _ in range(rows_count)]


    for i in range(rows_count):
        # Диагональ
        matrix[i][i] = rng.next_value(min_value, max_value)

        # Наддиагональный элемент
        if i + 1 < columns_count:
            matrix[i][i + 1] = rng.next_value(min_value, max_value)

        # Поддиагональный элемент
        if i + 1 < rows_count:
            matrix[i + 1][i] = rng.next_value(min_value, max_value)

        # Столбцы 5 и 6 (индексация Python начинается с 0, поэтому это реальные индексы 5 и 6)
        if 5 < columns_count:
            matrix[i][5] = rng.next_value(min_value, max_value)
        if 6 < columns_count:
            matrix[i][6] = rng.next_value(min_value, max_value)

    return matrix


def generate_matrix_diagonal_dominance(rng: NonZeroRandomProvider,
                    rows_count: int,
                    columns_count: int,
                    min_value: float,
                    max_value: float) -> list[list[float]]:
    if rng is None:
        raise ValueError("rng cannot be None")
    if rows_count < 0:
        raise ValueError("rows_count must not be negative.")
    if columns_count < 0:
        raise ValueError("columns_count must not be negative.")

    # Инициализируем матрицу нулями.
    matrix = [[0.0 for _ in range(columns_count)] for _ in range(rows_count)]


    for i in range(rows_count):
        # Диагональ — искусственно увеличиваем, чтобы достичь диагонального преобладания
        matrix[i][i] = rng.next_value(min_value + 1000, max_value + 1000)

        # Наддиагональный элемент
        if i + 1 < columns_count:
            matrix[i][i + 1] = rng.next_value(min_value, max_value)

        # Поддиагональный элемент
        if i + 1 < rows_count:
            matrix[i + 1][i] = rng.next_value(min_value, max_value)

        # Столбцы 5 и 6
        if 5 < columns_count:
            matrix[i][5] = rng.next_value(min_value, max_value)
        if 6 < columns_count:
            matrix[i][6] = rng.next_value(min_value, max_value)

    return matrix


def generate_matrix_hilbert(rng: NonZeroRandomProvider,
                    rows_count: int,
                    columns_count: int,
                    min_value: float,
                    max_value: float) -> list[list[float]]:
    """
    Генерация матрицы Гильберта.
    Для демонстрации "специальности" используем формулу 1/(i+j+1)
    только в позициях диагоналей, над- и поддиагоналей, столбцах 5 и 6.
    """
    if rng is None:
        raise ValueError("rng cannot be None")
    if rows_count < 0:
        raise ValueError("rows_count must not be negative.")
    if columns_count < 0:
        raise ValueError("columns_count must not be negative.")

    matrix = [[0.0 for _ in range(columns_count)] for _ in range(rows_count)]

    for i in range(rows_count):
        # Диагональ
        if i < columns_count:
            matrix[i][i] = 1 / (i + i + 1)  # H[i][i] = 1 / (2i + 1)

        # Наддиагональный элемент
        if i + 1 < columns_count:
            matrix[i][i + 1] = 1 / (i + (i + 1) + 1)  # H[i][i+1] = 1 / (2i + 2)

        # Поддиагональный элемент
        if i + 1 < rows_count:
            matrix[i + 1][i] = 1 / ((i + 1) + i + 1)  # H[i+1][i] = 1 / (2i + 2)

        # Столбцы 5 и 6
        if 5 < columns_count:
            matrix[i][5] = 1 / (i + 5 + 1)  # H[i][5] = 1 / (i + 6)
        if 6 < columns_count:
            matrix[i][6] = 1 / (i + 6 + 1)  # H[i][6] = 1 / (i + 7)

    return matrix


def generate_matrix_corrupted_d_e(rng: NonZeroRandomProvider,
                    rows_count: int,
                    columns_count: int,
                    min_value: float,
                    max_value: float) -> list[list[float]]:
    """
    Генерация "испорченной" матрицы, где столбцы d и e (5 и 6)
    заполняются значениями с большим разбросом (min_val-1e6, max_val+1e6).
    """
    if rng is None:
        raise ValueError("rng cannot be None")
    if rows_count < 0:
        raise ValueError("rows_count must not be negative.")
    if columns_count < 0:
        raise ValueError("columns_count must not be negative.")

    matrix = [[0.0 for _ in range(columns_count)] for _ in range(rows_count)]

    for i in range(rows_count):
        # Диагональ
        matrix[i][i] = rng.next_value(min_value, max_value)

        # Наддиагональный элемент
        if i + 1 < columns_count:
            matrix[i][i + 1] = rng.next_value(min_value, max_value)

        # Поддиагональный элемент
        if i + 1 < rows_count:
            matrix[i + 1][i] = rng.next_value(min_value, max_value)

        # Столбцы 5 и 6 — сильно сдвигаем диапазоны
        if 5 < columns_count:
            matrix[i][5] = rng.next_value(min_value - 1e6, max_value + 1e6)
        if 6 < columns_count:
            matrix[i][6] = rng.next_value(min_value - 1e6, max_value + 1e6)

    return matrix


def generate_matrix_largest_side_diagonals(rng: NonZeroRandomProvider,
                    rows_count: int,
                    columns_count: int,
                    min_value: float,
                    max_value: float) -> list[list[float]]:
    """
    Генерация матрицы, где побочные диагонали "накачаны" большими значениями.
    """
    if rng is None:
        raise ValueError("rng cannot be None")
    if rows_count < 0:
        raise ValueError("rows_count must not be negative.")
    if columns_count < 0:
        raise ValueError("columns_count must not be negative.")

    matrix = [[0.0 for _ in range(columns_count)] for _ in range(rows_count)]

    for i in range(rows_count):
        # Диагональ
        matrix[i][i] = rng.next_value(min_value, max_value)

        # Наддиагональный элемент с большим сдвигом
        if i + 1 < columns_count:
            matrix[i][i + 1] = rng.next_value(min_value + 1e6, max_value + 1e6)

        # Поддиагональный элемент
        if i + 1 < rows_count:
            matrix[i + 1][i] = rng.next_value(min_value, max_value)

        # Столбцы 5 и 6
        if 5 < columns_count:
            matrix[i][5] = rng.next_value(min_value, max_value)
        if 6 < columns_count:
            matrix[i][6] = rng.next_value(min_value, max_value)

    return matrix

# ------------------------------------------------------------------------------
# 3. Функция для "продолжительного" случайного вектора
# ------------------------------------------------------------------------------
def generate_random_vector(rng: NonZeroRandomProvider,
                           length: int,
                           min_value: float,
                           max_value: float) -> list[float]:
    result = []
    for _ in range(length):
        result.append(rng.next_value(min_value, max_value))
    return result

# ------------------------------------------------------------------------------
# 4. Функция для умножения матрицы на вектор
# ------------------------------------------------------------------------------
def multiply_matrix_by_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    if len(vector) != cols:
        raise ValueError("Dimension mismatch in multiply_matrix_by_vector")
    result = [0.0] * rows
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += matrix[i][j] * vector[j]
        result[i] = s
    return result

# ------------------------------------------------------------------------------
# 5. "Вставка" столбца в матрицу
# ------------------------------------------------------------------------------
def insert_column(matrix: list[list[float]], column: list[float]) -> list[list[float]]:
    rows = len(matrix)
    if rows == 0:
        return []
    cols = len(matrix[0])
    if len(column) != rows:
        raise ValueError("Dimension mismatch in insert_column")

    new_matrix = []
    for i in range(rows):
        new_row = matrix[i] + [column[i]]
        new_matrix.append(new_row)
    return new_matrix

# ------------------------------------------------------------------------------
# 6. Метрики точности
# ------------------------------------------------------------------------------
def calculate_accuracy(expected: list[float], actual: list[float], non_zero_eps: float) -> float:
    if len(expected) != len(actual):
        raise ValueError("Vectors must be the same length to calculate accuracy.")

    acc_sum = 0.0
    for e, a in zip(expected, actual):
        denom = e if abs(e) > non_zero_eps else 1.0
        acc_sum += abs(e - a) / abs(denom)
    return acc_sum / len(expected)

# ------------------------------------------------------------------------------
# 7. Класс FirstTaskMatrix для работы с матрицей
# ------------------------------------------------------------------------------
class FirstTaskMatrix:
    def __init__(self, matrix: list[list[float]]):
        if matrix is None:
            raise ValueError("matrix cannot be None")

        self._solved = False
        self._rows_count = len(matrix)
        if self._rows_count == 0:
            raise ValueError("Matrix cannot have 0 rows.")

        self._columns_count = len(matrix[0])
        # Проверка на прямоугольность:
        for row in matrix:
            if len(row) != self._columns_count:
                raise ValueError("All rows in the matrix must have the same length.")

        # Ниже повторяется та же схема, что и в C#:
        self._a = [0.0] * (self._rows_count - 1)  # поддиагональ
        for i in range(self._rows_count - 1):
            if i < self._columns_count:
                self._a[i] = matrix[i + 1][i]

        self._b = [0.0] * self._rows_count        # диагональ
        for i in range(self._rows_count):
            if i < self._columns_count:
                self._b[i] = matrix[i][i]

        self._c = [0.0] * (self._rows_count - 1)  # наддиагональ
        for i in range(self._rows_count - 1):
            if (i + 1) < self._columns_count:
                self._c[i] = matrix[i][i + 1]

        self._d = [0.0] * self._rows_count        # столбец 5
        if self._columns_count > 5:
            for i in range(self._rows_count):
                self._d[i] = matrix[i][5]

        self._e = [0.0] * self._rows_count        # столбец 6
        if self._columns_count > 6:
            for i in range(self._rows_count):
                self._e[i] = matrix[i][6]

        # последний столбец - свободные члены
        self._f = [0.0] * self._rows_count
        for i in range(self._rows_count):
            self._f[i] = matrix[i][self._columns_count - 1]

        self._result = [0.0] * self._rows_count

    def _is_belongs_to_c(self, row_index: int) -> bool:
        return row_index < self._rows_count - 1

    def _is_belongs_to_a(self, row_index: int) -> bool:
        return row_index > 0

    def _devide_line(self, row_index: int, element: float):
        if element == 0.0:
            return
        if self._is_belongs_to_a(row_index):
            self._a[row_index - 1] /= element
        self._b[row_index] /= element
        self._d[row_index] /= element
        self._e[row_index] /= element
        self._f[row_index] /= element
        if self._is_belongs_to_c(row_index):
            self._c[row_index] /= element

    def _sub_current_from_next(self, row_index: int):
        if row_index < self._rows_count - 1:
            next_row = row_index + 1
            self._a[row_index]    -= self._b[row_index]
            self._b[next_row]     -= self._c[row_index]
            self._d[next_row]     -= self._d[row_index]
            self._e[next_row]     -= self._e[row_index]
            self._f[next_row]     -= self._f[row_index]

            # Специальные костыли из примера
            if next_row == 4 and len(self._c) > 4:
                self._c[4] = self._d[4]
            elif next_row == 5 and len(self._c) > 5:
                self._c[5] = self._e[5]
        else:
            raise IndexError("rowIndex out of range in _sub_current_from_next")

    def _sub_prev_from_current(self, row_index: int):
        if row_index > 0:
            prev_row = row_index - 1
            self._c[prev_row] -= self._b[row_index]
            self._b[prev_row] -= self._a[prev_row]
            self._d[prev_row] -= self._d[row_index]
            self._e[prev_row] -= self._e[row_index]
            self._f[prev_row] -= self._f[row_index]

            if prev_row == 7 and len(self._a) > 7:
                self._a[6] = self._e[7]
            elif prev_row == 6 and len(self._a) > 6:
                self._a[5] = self._d[6]
        else:
            raise IndexError("rowIndex out of range in _sub_prev_from_current")

    def _first_phase(self):
        for row_index in range(0, min(6, self._rows_count)):
            if self._b[row_index] == 0.0:
                continue
            self._devide_line(row_index, self._b[row_index])
            if (row_index + 1) < self._rows_count and row_index < len(self._a):
                if self._a[row_index] == 0.0:
                    continue
                self._devide_line(row_index + 1, self._a[row_index])
                self._sub_current_from_next(row_index)

    def _second_phase(self):
        for row_index in range(self._rows_count - 1, 6, -1):
            if self._b[row_index] == 0.0:
                continue
            self._devide_line(row_index, self._b[row_index])
            if (row_index - 1) < len(self._c) and self._c[row_index - 1] != 0.0:
                self._devide_line(row_index - 1, self._c[row_index - 1])
                self._sub_prev_from_current(row_index)
        if self._rows_count > 6 and len(self._e) > 6 and self._e[6] != 0.0:
            self._devide_line(6, self._e[6])

    def _third_phase(self):
        for row_index in range(self._rows_count):
            if row_index != 5 and self._d[row_index] != 0.0:
                self._devide_line(row_index, self._d[row_index])
        for row_index in range(self._rows_count):
            if row_index != 5 and self._d[row_index] != 0.0:
                self._d[row_index] -= self._d[5]
                self._e[row_index] -= self._e[5]
                self._f[row_index] -= self._f[5]
        if self._rows_count > 5 and len(self._a) > 5:
            self._a[5] = self._d[6] if len(self._d) > 6 else 0.0
        if self._rows_count > 6 and len(self._a) > 6:
            self._a[6] = self._e[7] if len(self._e) > 7 else 0.0
        if len(self._b) > 5:
            self._b[5] = self._d[5]
        if len(self._b) > 6:
            self._b[6] = self._e[6]
        if len(self._c) > 4:
            self._c[4] = self._d[4]
        if len(self._c) > 5:
            self._c[5] = self._e[5]

    def _fourth_phase(self):
        if self._rows_count > 6 and len(self._e) > 6 and self._e[6] != 0.0:
            self._devide_line(6, self._e[6])
        for row_index in range(self._rows_count):
            if row_index != 6 and self._e[row_index] != 0.0:
                self._devide_line(row_index, self._e[row_index])
        for row_index in range(self._rows_count):
            if row_index != 6 and self._e[row_index] != 0.0:
                self._d[row_index] -= self._d[6]
                self._e[row_index] -= self._e[6]
                self._f[row_index] -= self._f[6]
        if self._rows_count > 5 and len(self._a) > 5:
            self._a[5] = self._d[6] if len(self._d) > 6 else 0.0
        if self._rows_count > 6 and len(self._a) > 6:
            self._a[6] = self._e[7] if len(self._e) > 7 else 0.0
        if len(self._b) > 5:
            self._b[5] = self._d[5]
        if len(self._b) > 6:
            self._b[6] = self._e[6]
        if len(self._c) > 4:
            self._c[4] = self._d[4]
        if len(self._c) > 5:
            self._c[5] = self._e[5]

    def _fifth_phase(self):
        for row_index in range(self._rows_count):
            if self._b[row_index] != 0.0:
                self._devide_line(row_index, self._b[row_index])

    def _calculate_phase(self):
        for row_index in range(4, min(8, self._rows_count)):
            self._result[row_index] = self._f[row_index]
        for row_index in range(3, -1, -1):
            if row_index < self._rows_count:
                self._result[row_index] = self._f[row_index]
                if row_index < len(self._c):
                    self._result[row_index] -= self._c[row_index] * (
                        self._result[row_index + 1] if (row_index + 1) < self._rows_count else 0.0
                    )
        for row_index in range(8, self._rows_count):
            self._result[row_index] = self._f[row_index]
            if (row_index - 1) < len(self._a):
                self._result[row_index] -= self._a[row_index - 1] * (
                    self._result[row_index - 1] if (row_index - 1) >= 0 else 0.0
                )

    def solve(self) -> list[float]:
        if not self._solved:
            self._solved = True
            self._first_phase()
            self._second_phase()
            self._third_phase()
            self._fourth_phase()
            self._fifth_phase()
            self._calculate_phase()
        return self._result

    def to_string(self, digits_after_comma: int = 2, separator: str = '\t') -> str:
        leaky_matrix = {}
        for i in range(len(self._a)):
            leaky_matrix[(i + 1, i)] = self._a[i]
        for i in range(len(self._b)):
            leaky_matrix[(i, i)] = self._b[i]
        for i in range(len(self._c)):
            leaky_matrix[(i, i + 1)] = self._c[i]
        if self._columns_count > 5:
            for i in range(len(self._d)):
                leaky_matrix[(i, 5)] = self._d[i]
        if self._columns_count > 6:
            for i in range(len(self._e)):
                leaky_matrix[(i, 6)] = self._e[i]
        for i in range(len(self._f)):
            leaky_matrix[(i, self._columns_count - 1)] = self._f[i]

        lines = []
        for i in range(self._rows_count):
            row_str = []
            for j in range(self._columns_count):
                val = leaky_matrix.get((i, j), 0.0)
                row_str.append(str(round(val, digits_after_comma)))
            lines.append(separator.join(row_str))
        return "\n".join(lines)

    #def __str__(self):
    #return self.to_string(2)

    def to_table(self, digits_after_comma: int = 2, table_format: str = "grid") -> str:
        """
        Возвращает строку с таблицей матрицы, отформатированной с помощью tabulate.

        :param digits_after_comma: Количество знаков после запятой для округления.
        :param table_format: Формат таблицы для tabulate (например, "grid", "fancy_grid", "plain", и т.д.).
        :return: Строка с отформатированной таблицей.
        """
        leaky_matrix = {}
        for i in range(len(self._a)):
            leaky_matrix[(i + 1, i)] = self._a[i]
        for i in range(len(self._b)):
            leaky_matrix[(i, i)] = self._b[i]
        for i in range(len(self._c)):
            leaky_matrix[(i, i + 1)] = self._c[i]
        if self._columns_count > 5:
            for i in range(len(self._d)):
                leaky_matrix[(i, 5)] = self._d[i]
        if self._columns_count > 6:
            for i in range(len(self._e)):
                leaky_matrix[(i, 6)] = self._e[i]
        for i in range(len(self._f)):
            leaky_matrix[(i, self._columns_count - 1)] = self._f[i]

        # Создаём 2D список для передачи в tabulate
        table = []
        for i in range(self._rows_count):
            row = []
            for j in range(self._columns_count):
                val = leaky_matrix.get((i, j), 0.0)
                row.append(round(val, digits_after_comma))
            table.append(row)

        # Генерируем заголовки столбцов (опционально)
        headers = [f"Col {j}" for j in range(1, self._columns_count + 1)]

        # Возвращаем отформатированную таблицу
        return tabulate(table, headers=headers, tablefmt=table_format)

    def __str__(self):
        return self.to_table()
# ------------------------------------------------------------------------------
# 8. Поиск погрешности и основная функция
# ------------------------------------------------------------------------------

def find_accuracies(generate_matrix_func, count: int, min_val: float, max_val: float, non_zero_eps: float) -> tuple[float, float]:
    if count < 0:
        raise ValueError("The number of elements must not be negative.")

    rng = NonZeroRandomProvider(non_zero_eps)

    # Генерируем матрицу без правой части (N x N):
    matrix_without_right_side = generate_matrix_func(rng, count, count, min_val, max_val)

    # Ожидаемое решение-1: случайный вектор
    expect_random_solution = generate_random_vector(rng, count, min_val, max_val)
    # Ожидаемое решение-2: единичный вектор
    expect_unit_solution = [1.0] * count

    # Получаем правые части:
    random_right_side = multiply_matrix_by_vector(matrix_without_right_side, expect_random_solution)
    unit_right_side = multiply_matrix_by_vector(matrix_without_right_side, expect_unit_solution)

    # Собираем «полные» матрицы
    random_matrix_data = insert_column(matrix_without_right_side, random_right_side)
    unit_matrix_data = insert_column(matrix_without_right_side, unit_right_side)

    # Создаём объекты FirstTaskMatrix и решаем системы
    random_matrix = FirstTaskMatrix(random_matrix_data)
    actual_random_solution = random_matrix.solve()

    unit_matrix = FirstTaskMatrix(unit_matrix_data)
    actual_unit_solution = unit_matrix.solve()

    # Возвращаем пару погрешностей
    return (
        calculate_accuracy(expect_unit_solution, actual_unit_solution, non_zero_eps),
        calculate_accuracy(expect_random_solution, actual_random_solution, non_zero_eps)
    )

def show_matrix_steps(matrix_types, non_zero_eps):
    print("Выберите тип матрицы для отображения этапов решения:")
    for key, (description, _) in matrix_types.items():
        if key != '6':  # Исключаем пункт 6
            print(f"{key}. {description}")

    selected_type = None
    while selected_type not in matrix_types or selected_type == '6':
        selected_type = input("Введите номер выбранного типа матрицы: ").strip()
        if selected_type not in matrix_types or selected_type == '6':
            print("Неверный выбор. Пожалуйста, попробуйте снова.")

    description, generate_func = matrix_types[selected_type]
    print(f"\nВыбранный тип матрицы: {description}\n")

    # Запрашиваем параметры для матрицы
    try:
        count = int(input("Введите размер матрицы N (например, 10): ").strip())
        min_val = float(input("Введите минимальное значение элементов матрицы (например, -100): ").strip())
        max_val = float(input("Введите максимальное значение элементов матрицы (например, 100): ").strip())
    except ValueError:
        print("Неверный ввод. Пожалуйста, введите числовые значения.\n")
        return

    # Вызов функции для отображения этапов решения
    show_matrix_steps_internal(
        selected_generate_matrix=generate_func,
        count=count,
        min_val=min_val,
        max_val=max_val,
        non_zero_eps=non_zero_eps
    )

def show_matrix_steps_internal(selected_generate_matrix, count, min_val, max_val, non_zero_eps):
    rng = NonZeroRandomProvider(non_zero_eps)

    # Генерируем матрицу без правой части (N x N)
    matrix_without_right_side = selected_generate_matrix(rng, count, count, min_val, max_val)

    # Ожидаемое решение-1: единичный вектор (для простоты)
    expect_unit_solution = [1.0] * count

    # Получаем правую часть
    unit_right_side = multiply_matrix_by_vector(matrix_without_right_side, expect_unit_solution)

    # Собираем «полную» матрицу
    unit_matrix_data = insert_column(matrix_without_right_side, unit_right_side)

    # Создаём объект FirstTaskMatrix
    matrix_obj = FirstTaskMatrix(unit_matrix_data)

    print("Шаг 0. Исходная матрица:")
    print(matrix_obj.to_table())
    print("\n")

    # 1. Первая фаза
    matrix_obj._first_phase()
    print("Шаг 1. После _first_phase():")
    print(matrix_obj.to_table())
    print("\n")

    # 2. Вторая фаза
    matrix_obj._second_phase()
    print("Шаг 2. После _second_phase():")
    print(matrix_obj.to_table())
    print("\n")

    # 3. Третья фаза
    matrix_obj._third_phase()
    print("Шаг 3. После _third_phase():")
    print(matrix_obj.to_table())
    print("\n")

    # 4. Четвёртая фаза
    matrix_obj._fourth_phase()
    print("Шаг 4. После _fourth_phase():")
    print(matrix_obj.to_table())
    print("\n")

    # 5. Пятая фаза
    matrix_obj._fifth_phase()
    print("Шаг 5. После _fifth_phase():")
    print(matrix_obj.to_table())
    print("\n")

    # 6. Финальный расчёт (заполнение self._result)
    matrix_obj._calculate_phase()
    print("Шаг 6. После _calculate_phase():")
    print(matrix_obj.to_table())
    print("\n")

    # Выводим итоговый вектор решений
    print("Полученный вектор решений:", matrix_obj._result)

# ------------------------------------------------------------------------------
# 9. Запуск main
# ------------------------------------------------------------------------------
def main():
    NON_ZERO_EPS = 1e-5
    TEST_COUNT = 3

    matrix_types = {
        '1': ('Матрица с диагональным преобладанием', generate_matrix_diagonal_dominance),
        '2': ('Рандомная матрица', generate_matrix),
        '3': ('Матрица Гильберта', generate_matrix_hilbert),
        '4': ('Матрица (испорченные столбцы d и e)', generate_matrix_corrupted_d_e),
        '5': ('Матрица с наибольшими побочными диагоналями', generate_matrix_largest_side_diagonals),
        '6': ('Показать этапы решения матрицы', None)  # Добавляем новый пункт
    }

    while True:
        print("Выберите тип матрицы или выполните тест:")
        for key, (description, _) in matrix_types.items():
            print(f"{key}. {description}")
        print("0. Выход")

        selected_type = input("Введите номер выбранного типа матрицы: ").strip()

        if selected_type == '0':
            print("Завершение программы.")
            break
        elif selected_type in matrix_types:
            description, generate_func = matrix_types[selected_type]
            if selected_type != '6':
                print(f"\nВыбранный тип матрицы: {description}\n")

                # Набор тестов
                test_cases = [
                    (10, -100, 100),
                    (20, -100, 100),
                    (40, -100, 100),
                    (60, -100, 100),
                    (100, -100, 100),
                    (1000, -100, 100),
                    (2000, -100, 100),
                    # Можно добавить ещё
                ]

                # Собираем результаты в таблицу
                table_data = []
                for (count, min_val, max_val) in test_cases:
                    accuracies = []
                    for _ in range(TEST_COUNT):
                        unit_acc, rand_acc = find_accuracies(generate_func, count, min_val, max_val, NON_ZERO_EPS)
                        accuracies.append((unit_acc, rand_acc))

                    avg_unit_accuracy = statistics.mean([x[0] for x in accuracies])
                    avg_rand_accuracy = statistics.mean([x[1] for x in accuracies])

                    # Добавляем строку в табличные данные
                    table_data.append([
                        count,
                        min_val,
                        max_val,
                        avg_unit_accuracy,
                        avg_rand_accuracy
                    ])

                # Печатаем таблицу
                headers = ["N", "Min", "Max", "Unit solution error", "Random solution error"]
                print(tabulate(table_data, tablefmt="grid", headers=headers, floatfmt=".5g"))

            else:
                # Обработка нового тестового выбора
                show_matrix_steps(matrix_types, NON_ZERO_EPS)

            print("\n")
        else:
            print("Неверный выбор. Пожалуйста, попробуйте снова.\n")

    print("С наступающим, Олег Геннадьевич!")

# ------------------------------------------------------------------------------
# Запуск main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
