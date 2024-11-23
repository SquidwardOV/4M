#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <random>
#include <string>

// Класс Vector
class Vector {
public:
    std::vector<double> data;

    // Конструкторы
    Vector() {}
    Vector(int size) : data(size) {}

    // Сложение
    Vector operator+(const Vector& other) const {
        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    // Вычитание
    Vector operator-(const Vector& other) const {
        Vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    // Скалярное произведение
    double dot(const Vector& other) const {
        double result = 0;
        for (size_t i = 0; i < data.size(); ++i)
            result += data[i] * other.data[i];
        return result;
    }

    // Норма (максимальная по модулю компонента)
    double norm() const {
        double max_val = 0;
        for (double val : data)
            max_val = std::max(max_val, std::abs(val));
        return max_val;
    }

    // Считывание с экрана
    void readFromConsole() {
        for (double& val : data)
            std::cin >> val;
    }

    // Вывод на экран
    void printToConsole() const {
        for (const double& val : data)
            std::cout << val << " ";
        std::cout << std::endl;
    }

    // Считывание из файла
    void readFromFile(const std::string& filename) {
        std::ifstream infile(filename);
        for (double& val : data)
            infile >> val;
        infile.close();
    }

    // Запись в файл
    void writeToFile(const std::string& filename) const {
        std::ofstream outfile(filename);
        for (const double& val : data)
            outfile << val << " ";
        outfile << std::endl;
        outfile.close();
    }

    // Заполнение случайными числами из заданного диапазона
    void fillRandom(double min_val, double max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min_val, max_val);
        for (double& val : data)
            val = dis(gen);
    }
};

// Класс Matrix
class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    // Конструкторы
    Matrix() : rows(0), cols(0) {}
    Matrix(int n, int m) : rows(n), cols(m), data(n, std::vector<double>(m)) {}

    // Сложение
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    // Вычитание
    Matrix operator-(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    // Умножение на вектор
    Vector operator*(const Vector& vec) const {
        Vector result(rows);
        for (int i = 0; i < rows; ++i) {
            result.data[i] = 0;
            for (int j = 0; j < cols; ++j)
                result.data[i] += data[i][j] * vec.data[j];
        }
        return result;
    }

    // Считывание с экрана
    void readFromConsole() {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                std::cin >> data[i][j];
    }

    // Вывод на экран
    void printToConsole() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << data[i][j] << " ";
            std::cout << std::endl;
        }
    }

    // Считывание из файла
    void readFromFile(const std::string& filename) {
        std::ifstream infile(filename);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                infile >> data[i][j];
        infile.close();
    }

    // Запись в файл
    void writeToFile(const std::string& filename) const {
        std::ofstream outfile(filename);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                outfile << data[i][j] << " ";
            outfile << std::endl;
        }
        outfile.close();
    }
};

// Функции генерации матриц
void generateRandomMatrix(Matrix& A, double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.data[i][j] = dis(gen);
}

void generateDiagonallyDominantMatrix(Matrix& A, double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);

    for (int i = 0; i < A.rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < A.cols; ++j) {
            if (i != j) {
                A.data[i][j] = dis(gen);
                sum += std::abs(A.data[i][j]);
            }
        }
        // Обеспечиваем диагональное преобладание
        A.data[i][i] = sum + std::abs(dis(gen)) + 1.0;
    }
}

void generateHilbertMatrix(Matrix& A) {
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.data[i][j] = 1.0 / (i + j + 1);
}

// Метод Гаусса с полной стратегией выбора ведущего элемента
void gaussFullPivoting(Matrix& A, Vector& b, Vector& x) {
    int n = A.rows;
    std::vector<int> row_swaps(n), col_swaps(n);
    for (int i = 0; i < n; ++i) {
        row_swaps[i] = i;
        col_swaps[i] = i;
    }

    for (int k = 0; k < n; ++k) {
        // Поиск максимального элемента
        double max_val = 0.0;
        int max_row = k, max_col = k;
        for (int i = k; i < n; ++i) {
            for (int j = k; j < n; ++j) {
                if (std::abs(A.data[i][j]) > max_val) {
                    max_val = std::abs(A.data[i][j]);
                    max_row = i;
                    max_col = j;
                }
            }
        }

        // Проверка на вырожденность
        if (max_val == 0.0) {
            std::cerr << "Матрица вырождена!" << std::endl;
            return;
        }

        // Перестановка строк
        if (max_row != k) {
            std::swap(A.data[k], A.data[max_row]);
            std::swap(b.data[k], b.data[max_row]);
            std::swap(row_swaps[k], row_swaps[max_row]);
        }

        // Перестановка столбцов
        if (max_col != k) {
            for (int i = 0; i < n; ++i)
                std::swap(A.data[i][k], A.data[i][max_col]);
            std::swap(col_swaps[k], col_swaps[max_col]);
        }

        // Прямой ход
        for (int i = k + 1; i < n; ++i) {
            double factor = A.data[i][k] / A.data[k][k];
            A.data[i][k] = 0;
            for (int j = k + 1; j < n; ++j)
                A.data[i][j] -= factor * A.data[k][j];
            b.data[i] -= factor * b.data[k];
        }
    }

    // Обратный ход
    x.data.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        x.data[i] = b.data[i];
        for (int j = i + 1; j < n; ++j)
            x.data[i] -= A.data[i][j] * x.data[j];
        x.data[i] /= A.data[i][i];
    }

    // Восстановление порядка переменных
    Vector x_reordered(n);
    for (int i = 0; i < n; ++i)
        x_reordered.data[col_swaps[i]] = x.data[i];
    x = x_reordered;
}

// Вычислительный эксперимент
void computationalExperiment(int size, const std::string& matrixType, double& error) {
    // Инициализация матрицы A и точного решения x_exact
    Matrix A(size, size);
    Vector x_exact(size);
    x_exact.fillRandom(-10.0, 10.0); // Случайное точное решение

    // Генерация матрицы A на основе выбранного типа
    if (matrixType == "random") {
        generateRandomMatrix(A, -10.0, 10.0);
    }
    else if (matrixType == "diagonally_dominant") {
        generateDiagonallyDominantMatrix(A, -10.0, 10.0);
    }
    else if (matrixType == "hilbert") {
        generateHilbertMatrix(A);
    }
    else {
        std::cerr << "Неизвестный тип матрицы!" << std::endl;
        return;
    }

    // Вычисляем вектор правой части b
    Vector b = A * x_exact;

    // Решаем систему методом Гаусса с полной стратегией выбора ведущего элемента
    Vector x_approx;
    Matrix A_copy = A; // Копируем A, так как gaussFullPivoting модифицирует A
    Vector b_copy = b;
    gaussFullPivoting(A_copy, b_copy, x_approx);

    // Вычисляем погрешность
    Vector diff = x_approx - x_exact;
    error = diff.norm() / x_exact.norm();
 
}

int main() {
    setlocale(LC_ALL, "Russian");
    std::cout << "Выберите тип матрицы:\n";
    std::cout << "1 - Случайная матрица\n";
    std::cout << "2 - Диагонально доминантная матрица\n";
    std::cout << "3 - Матрица Гильберта\n";
    int choice;
    std::cin >> choice;

    std::string matrixType;
    if (choice == 1) {
        matrixType = "random";
    }
    else if (choice == 2) {
        matrixType = "diagonally_dominant";
    }
    else if (choice == 3) {
        matrixType = "hilbert";
    }
    else {
        std::cerr << "Неверный выбор!" << std::endl;
        return 1;
    }

    std::ofstream outfile("errors_" + matrixType + ".txt");
   // std::cout << "Тип матрицы: " << matrixType << std::endl;
    for (int size = 2; size <= 1000; size *= 2) { // Можно изменить размерность по необходимости
        double error;
        computationalExperiment(size, matrixType, error);
        std::cout << "Размер: " << size << ", Погрешность: " << error << std::endl;
        outfile << size << " " << error << std::endl;
    }
    outfile.close();
    return 0;
}
