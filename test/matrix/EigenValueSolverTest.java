package matrix;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static java.lang.Math.*;
import static matrix.Matrix.ofTable;
import static org.junit.Assert.assertArrayEquals;

public class EigenValueSolverTest {

    @Test
    public void eigenValues_shouldBeCorrectAndSorted_onExample1() {
        double[][] data = {
                { 12, -51, 4 },
                { 6, 167, -68 },
                { -4, 24, -41 }
        };
        double[] eigenvalues = new double[] { 156.136_7, -34.196_7, 16.060_0 };

        test(data, 50, 0.000_1, eigenvalues);
    }

    @Test
    public void eigenValues_shouldBeCorrectAndSorted_onExample2() {
        double[][] data = {
                { 17, 49, 25 },
                { 49, 3, -40 },
                { 25, -40, 0 }
        };
        double[] eigenvalues = new double[] { -70.485, 60.864, 29.621 };

        test(data, 200, 0.000_1, eigenvalues);
    }

    @Test
    public void eigenValues_shouldBeCorrectAndSorted_onExample5x5() {
        double[][] data = {
                { 12, -51, 4, 0, 0 },
                { -51, 167, -68, 1, -12 },
                { 4, -68, -41, 7, 4 },
                { 0, 1, 7, 4, 56 },
                { 0, -12, 4 , 56, 30}
        };
        double[] eigenvalues = new double[] { 201.562, 73.976, -64.376, -39.494, 0.331 };

        test(data, 300, 0.000_1, eigenvalues);
    }

    private void test(double[][] data, int iterationBound, double sensitivity, double[] expectations) {
        var generator = new Random(6466585);
        double[] eigenvalues = new double[expectations.length];
        var A = ofTable(data);
        var solver = new EigenValueSolver() {
            @Override
            public double sensitivity() {
                return sensitivity;
            }

            @Override
            public int iterationBound() {
                return iterationBound;
            }

            @Override
            public double shiftInContext(double[][] data) {
                var lambda = generator.nextDouble();
                var i = generator.nextInt(data.length);
                var w1 = data[i][i];
                return (1.1 - lambda / 5) * w1;
            }
        };
        var efficiency = solver.flushEigenvalues(A, eigenvalues);

        { // Pretty prints
            System.out.println("Initial matrix A:");
            System.out.println(Matrix.toString(A));
            System.out.println("-----------");
            System.out.println("Eigenvalues");
            System.out.println("-----------");
            System.out.println(Arrays.toString(eigenvalues));
            System.out.printf("Took %d steps to iterate the process%n", iterationBound - efficiency);
            {
                double sum = Arrays.stream(eigenvalues).map(Math::abs).sum();
                for(int i = 0; i < expectations.length; i++)
                    System.out.printf("\tStrength of space of dimension %d: %.2f%n",
                            i+1,
                            Arrays.stream(eigenvalues).map(Math::abs).limit(i+1).sum()/sum
                    );
            }
            System.out.println("/----------");

            assertArrayEquals(expectations, eigenvalues, 10*sensitivity);
        }
    }
    private static double wilkinsonShift(double[][] data, int s) {
        var n = data.length - 1 - s;
        assert n >= 0;
        var c = data[n][n];
        var a = data[n-1][n-1];
        var b = (data[n-1][n] + data[n][n-1]) / 2;
        var d = (a - c) / 2;
        return c - (d < 0 ? -1 : 1) * pow(b, 2) / (abs(d) + hypot(d, b));
    }

}
