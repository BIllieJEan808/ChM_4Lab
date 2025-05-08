#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <chrono>
#include <iomanip>
#include <vector>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

// Размерность матрицы
const int N = 20;
const int NUM_RUNS = 100;
const double SINGULAR_THRESHOLD = 1e-12;

// Создание матрицы A
MatrixXd createMatrixA() {
    MatrixXd A(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A(i, j) = 1.0 / (1 + 0.3 * i + 4 * j);
    return A;
}

// Функция вычисления относительной Эйлеровой погрешности
double computeRelativeError(const VectorXd& x_exact, const VectorXd& x_computed) {
    return (x_exact - x_computed).norm() / x_exact.norm();
}

// Расчет числа обусловленности без вывода сингулярных чисел
double computeConditionNumber(const VectorXd& s) {
    double max_s = 0, min_s = numeric_limits<double>::max();
    for (int i = 0; i < s.size(); ++i) {
        double val = (s(i) < SINGULAR_THRESHOLD) ? 0.0 : s(i);
        if (val > max_s) max_s = val;
        if (val > 0 && val < min_s) min_s = val;
    }
    return (min_s > 0) ? (max_s / min_s) : numeric_limits<double>::infinity();
}

int main() {
    setlocale(LC_ALL, "Russian");
    MatrixXd A = createMatrixA();
    VectorXd x_exact = VectorXd::Ones(N);
    VectorXd f = A * x_exact;

    double totalTimeLU = 0, totalTimeQR = 0, totalTimeSVD = 0;
    double errorLU = 0, errorQR = 0, errorSVD = 0;

    VectorXd x;

    // LU
    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = high_resolution_clock::now();
        x = A.lu().solve(f);
        auto end = high_resolution_clock::now();
        totalTimeLU += duration<double>(end - start).count();
        errorLU += computeRelativeError(x_exact, x);
    }

    // QR (Givens rotations)
    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = high_resolution_clock::now();
        x = A.householderQr().solve(f);
        auto end = high_resolution_clock::now();
        totalTimeQR += duration<double>(end - start).count();
        errorQR += computeRelativeError(x_exact, x);
    }

    // SVD
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    VectorXd singular_values = svd.singularValues();
    double condition_number = computeConditionNumber(singular_values);

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = high_resolution_clock::now();
        x = svd.solve(f);
        auto end = high_resolution_clock::now();
        totalTimeSVD += duration<double>(end - start).count();
        errorSVD += computeRelativeError(x_exact, x);
    }

    // Улучшенный вывод
    cout << "\nСравнение методов решения СЛАУ\n";
    cout << "-----------------------------------------\n";
    cout << "| Метод          | Время (сек)  | Ошибка     |\n";
    cout << "-----------------------------------------\n";
    cout << fixed << setprecision(6);
    cout << "| LU-разложение  | " << setw(12) << totalTimeLU/NUM_RUNS << " | " 
         << scientific << setprecision(3) << setw(9) << errorLU/NUM_RUNS << " |\n";
    cout << "| QR-разложение  | " << fixed << setprecision(6) << setw(12) << totalTimeQR/NUM_RUNS << " | " 
         << scientific << setprecision(3) << setw(9) << errorQR/NUM_RUNS << " |\n";
    cout << "| SVD-метод      | " << fixed << setprecision(6) << setw(12) << totalTimeSVD/NUM_RUNS << " | " 
         << scientific << setprecision(3) << setw(9) << errorSVD/NUM_RUNS << " |\n";
    cout << "-----------------------------------------\n\n";

    cout << "Число обусловленности (cond) для SVD: ";
    if (isinf(condition_number)) {
        cout << "∞ (матрица вырождена или очень плохо обусловлена)\n";
    }
    else {
        cout << fixed << setprecision(1) << condition_number << "\n";
    }

    return 0;
}
