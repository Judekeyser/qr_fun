package matrix;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

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

    private void test(double[][] data, int iterationBound, double sensitivity, double[] expectations) {
        double[] eigenvalues = new double[3];
        var A = Matrices.ofTable(data);
        var efficiency = EigenValueSolver.flushEigenvalues(A, iterationBound, sensitivity, eigenvalues);

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
                for(int i = 0; i < 3; i++)
                    System.out.printf("\tStrength of space of dimension %d: %.2f%n",
                            i+1,
                            Arrays.stream(eigenvalues).map(Math::abs).limit(i+1).sum()/sum
                    );
            }
            System.out.println("/----------");

            assertArrayEquals(expectations, eigenvalues, 10*sensitivity);
        }
    }

}
