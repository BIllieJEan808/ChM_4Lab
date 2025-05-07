
import java.util.Arrays;

public class Main {
    private static final int MEASUREMENT_ROUNDS = 7;

    public static void main(String[] args) {
        int[] sizes = {250, 500, 1000};

        System.out.printf("%-6s | %-8s | %-16s | %s%n",
                "Размер", "Метод", "Среднее время (с)", "Погрешность");
        System.out.println("-----------------------------------------------");

        for (int N : sizes) {
            // Генерируем данные
            double[][] A = Matrix.createMatrix(N);
            double[] f = Matrix.generate_f(A);
            double[] xExact = new double[N];
            Arrays.fill(xExact, 1.0);

            // Измерение LU
            double luTime = measureLUTime(A, N);
            double luError = measureLUError(A, f, xExact, N);

            // Измерение QR
            double qrTime = measureQRTime(A, f, N);
            double qrError = measureQRError(A, f, xExact, N);

            // Вывод результатов
            printResults(N, "LU", luTime, luError);
            printResults(N, "QR", qrTime, qrError);
            System.out.println("-----------------------------------------------");
        }
    }

    private static double measureLUTime(double[][] A, int N) {
        long[] timings = new long[MEASUREMENT_ROUNDS];

        for (int i = 0; i < MEASUREMENT_ROUNDS; i++) {
            double[][] LU = Matrix.copyMatrix(A);
            int[] pivot = new int[N];

            long start = System.nanoTime();
            LU_Decomposition.luDecomposition(LU, pivot);
            long end = System.nanoTime();

            timings[i] = end - start;
        }

        return nanosToSeconds(median(timings));
    }

    private static double measureLUError(double[][] A, double[] f, double[] xExact, int N) {
        double[][] LU = Matrix.copyMatrix(A);
        int[] pivot = new int[N];
        LU_Decomposition.luDecomposition(LU, pivot);
        double[] x = LU_Decomposition.LU(LU, pivot, f);
        return Matrix.calculateError(x, xExact);
    }

    private static double measureQRTime(double[][] A, double[] f, int N) {
        long[] timings = new long[MEASUREMENT_ROUNDS];

        for (int i = 0; i < MEASUREMENT_ROUNDS; i++) {
            double[][] Q = new double[N][N];
            double[][] R = new double[N][N];
            double[][] A_copy = Matrix.copyMatrix(A);

            long start = System.nanoTime();
            QR_Decomposition.householderQR(A_copy, Q, R);
            long end = System.nanoTime();

            timings[i] = end - start;
        }

        return nanosToSeconds(median(timings));
    }

    private static double measureQRError(double[][] A, double[] f, double[] xExact, int N) {
        double[][] Q = new double[N][N];
        double[][] R = new double[N][N];
        double[][] A_copy = Matrix.copyMatrix(A);
        QR_Decomposition.householderQR(A_copy, Q, R);
        double[] x = QR_Decomposition.QR(Q, R, f);
        return Matrix.calculateError(x, xExact);
    }

    private static long median(long[] numbers) {
        Arrays.sort(numbers);
        return numbers[numbers.length / 2];
    }

    private static double nanosToSeconds(long nanos) {
        return nanos / 1_000_000_000.0;
    }

    private static void printResults(int size, String method, double time, double error) {
        System.out.printf("%-6d | %-8s | %-16.6f | %.3e%n",
                size, method, time, error);
    }
}