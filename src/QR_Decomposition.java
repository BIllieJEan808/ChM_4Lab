import java.util.Arrays;

public class QR_Decomposition {
    public static void householderQR(double[][] A, double[][] Q, double[][] R) {
        int n = A.length;

        // Инициализация Q как единичной матрицы
        for (int i = 0; i < n; ++i) {
            Arrays.fill(Q[i], 0.0);
            Q[i][i] = 1.0;
        }

        // Копирование A в R
        for (int i = 0; i < n; ++i) {
            System.arraycopy(A[i], 0, R[i], 0, n);
        }

        for (int k = 0; k < n - 1; ++k) {
            // Вычисление вектора отражения
            double norm = 0.0;
            for (int i = k; i < n; ++i) {
                norm += R[i][k] * R[i][k];
            }
            norm = Math.sqrt(norm);

            if (Math.abs(norm) < 1e-12) continue;

            double alpha = -Math.copySign(norm, R[k][k]);
            double[] v = new double[n];
            for (int i = k; i < n; ++i) {
                if (i == k){
                    v[i] = R[i][k] - alpha;
                }
                else {
                    v[i] = R[i][k];
                }
            }

            // Вычисление beta
            double beta = 0.0;
            for (int i = k; i < n; ++i) {
                beta += v[i] * v[i];
            }
            beta = 2.0 / beta;

            // Обновление R
            for (int j = k; j < n; ++j) {
                double dot = 0.0;
                for (int i = k; i < n; ++i) {
                    dot += v[i] * R[i][j];
                }
                for (int i = k; i < n; ++i) {
                    R[i][j] -= beta * v[i] * dot;
                }
            }

            // Обновление Q
            for (int j = 0; j < n; ++j) {
                double dot = 0.0;
                for (int i = k; i < n; ++i) {
                    dot += Q[j][i] * v[i];
                }
                for (int i = k; i < n; ++i) {
                    Q[j][i] -= beta * v[i] * dot;
                }
            }
        }
    }

    public static double[] QR(double[][] Q, double[][] R, double[] f) {
        int N = Q.length;
        double[] y = new double[N];
        double[] x = new double[N];

        // Вычисление Q^T * f
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                y[i] += Q[j][i] * f[j];
            }
        }

        // Обратная подстановка для Rx = y
        for (int i = N - 1; i >= 0; --i) {
            x[i] = y[i];
            for (int j = i + 1; j < N; ++j) {
                x[i] -= R[i][j] * x[j];
            }
            x[i] /= R[i][i];
        }

        return x;
    }
}
