import java.util.Arrays;

public class Matrix {
    public static double[][] createMatrix(int N) {

        double[][] A = new double[N][N];
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                if (row == col) {
                    A[row][col] = 100.0;
                } else {
                    A[row][col] = 1.0 - 0.1 * (row + 1) + 0.5 * (col + 1);
                }
            }
        }
        return A;
    }

    //(f = A * x*)
    public static double[] generate_f(final double[][] A) {
        int N = A.length;
        double[] f = new double[N];

        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                f[row] += A[row][col];
            }
        }
        return f;
    }

    // Вычисление относительной погрешности
    public static double calculateError(double[] x, double[] correct_x) {

        double diff_norm = 0.0;
        double correct_norm = 0.0;

        for (int i = 0; i < x.length; ++i) {
            diff_norm += (x[i] - correct_x[i]) * (x[i] - correct_x[i]);
            correct_norm += correct_x[i] * correct_x[i];
        }

        return Math.sqrt(diff_norm) / Math.sqrt(correct_norm);
    }

    public static double[][] copyMatrix(double[][] original) {
        double[][] copy = new double[original.length][];
        for (int i = 0; i < original.length; i++) {
            copy[i] = Arrays.copyOf(original[i], original[i].length);
        }
        return copy;
    }

}
