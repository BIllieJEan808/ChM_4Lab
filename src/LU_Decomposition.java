
public class LU_Decomposition {
    public static void luDecomposition(double[][] A, int[] p) {
        int N = A.length;

        // Инициализация вектора перестановок
        for (int i = 0; i < N; ++i) {
            p[i] = i;
        }

        for (int k = 0; k < N; ++k) {
            // Частичный выбор ведущего элемента
            int max_row = k;
            for (int i = k + 1; i < N; ++i) {
                if (Math.abs(A[i][k]) > Math.abs(A[max_row][k])) {
                    max_row = i;
                }
            }

            // Перестановка строк
            if (max_row != k) {
                // Обмен строк в матрице
                double[] temp = A[k];
                A[k] = A[max_row];
                A[max_row] = temp;

                // Обмен индексов в векторе перестановок
                int tempIdx = p[k];
                p[k] = p[max_row];
                p[max_row] = tempIdx;
            }

            // LU-разложение
            for (int i = k + 1; i < N; ++i) {
                A[i][k] /= A[k][k];
                for (int j = k + 1; j < N; ++j) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
        }
    }

    public static double[] LU(double[][] LU, int[] p, double[] f) {
        int N = LU.length;
        double[] x = new double[N];
        double[] y = new double[N];
        double[] b = new double[N];

        for (int i = 0; i < N; ++i) {
            b[i] = f[p[i]];
        }

        // Прямая подстановка (Ly = b)
        for (int i = 0; i < N; ++i) {
            y[i] = b[i];
            for (int j = 0; j < i; ++j) {
                y[i] -= LU[i][j] * y[j];
            }
        }

        // Обратная подстановка (Ux = y)
        for (int i = N - 1; i >= 0; --i) {
            x[i] = y[i];
            for (int j = i + 1; j < N; ++j) {
                x[i] -= LU[i][j] * x[j];
            }
            x[i] /= LU[i][i];
        }

        return x;
    }
}

