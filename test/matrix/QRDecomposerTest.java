package matrix;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class QRDecomposerTest {

    @Test
    public void qrdecomposition_stepByStep() {
        double[][] data = {
                { 12, -51, 4 },
                { 6, 167, -68 },
                { -4, 24, -41 }
        };
        var A = Matrices.ofTable(data);
        var H1 = QRDecomposer.step(A, 0);

        { // check H1
            assertArrayEquals(new double[] { 6.0/7 , 3.0/7, -2.0/7 }, H1.getRow(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 3.0/7 , -2.0/7, 6.0/7 }, H1.getRow(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { -2.0/7 , 6.0/7, 3.0/7 }, H1.getRow(2).toArray(), 0.000_001);

            assertArrayEquals(new double[] { 6.0/7 , 3.0/7, -2.0/7 }, H1.getColumn(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 3.0/7 , -2.0/7, 6.0/7 }, H1.getColumn(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { -2.0/7 , 6.0/7, 3.0/7 }, H1.getColumn(2).toArray(), 0.000_001);
        }

        { // check product
            var prod = Matrices.mult(H1, A);
            assertArrayEquals(new double[]{14.000000, 21.000000, -14.000000}, prod.getRow(0).toArray(), 0.000_001);
            assertArrayEquals(new double[]{-0.000000, -49.000000, -14.000000}, prod.getRow(1).toArray(), 0.000_001);
            assertArrayEquals(new double[]{0.000000, 168.000000, -77.000000}, prod.getRow(2).toArray(), 0.000_001);

            assertArrayEquals(new double[]{14.000000, -0.000000, 0.000000}, prod.getColumn(0).toArray(), 0.000_001);
            assertArrayEquals(new double[]{21.000000, -49.000000, 168.000000}, prod.getColumn(1).toArray(), 0.000_001);
            assertArrayEquals(new double[]{-14.000000, -14.000000, -77.000000}, prod.getColumn(2).toArray(), 0.000_001);
        }

        var H2 = QRDecomposer.step(H1.composeLeft(A), 1);

        { // check H2
            assertArrayEquals(new double[] { 1, 0, 0 }, H2.getRow(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, 7.0/25, -24.0/25 }, H2.getRow(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, -24.0/25, -7.0/25 }, H2.getRow(2).toArray(), 0.000_001);

            assertArrayEquals(new double[] { 1, 0, 0 }, H2.getColumn(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, 7.0/25, -24.0/25 }, H2.getColumn(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, -24.0/25, -7.0/25 }, H2.getColumn(2).toArray(), 0.000_001);
        }

        var Q = H2.composeLeft(H1).transpose();

        { // check Q
            assertArrayEquals(new double[] { 6.0/7, 69.0/175, -58.0/175 }, Q.getRow(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 3.0/7, -158.0/175, 6.0/175 }, Q.getRow(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { -2.0/7, -6.0/35, -33.0/35 }, Q.getRow(2).toArray(), 0.000_001);

            assertArrayEquals(new double[] { 6.0/7, 3.0/7, -2.0/7 }, Q.getColumn(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 69.0/175, -158.0/175, -6.0/35 }, Q.getColumn(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { -58.0/175, 6.0/175, -33.0/35 }, Q.getColumn(2).toArray(), 0.000_001);
        }

        { // check R
            var R = Q.transpose().composeLeft(A);
            assertArrayEquals(new double[] { 14, 21, -14 }, R.getRow(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, -175, 70 }, R.getRow(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, 0, 35 }, R.getRow(2).toArray(), 0.000_001);
        }

        { // check A = QR
            var R = Q.transpose().composeLeft(A);
            var Ab = Q.composeLeft(R);

            for(int i = 0; i < 3; i++)
                assertArrayEquals("Comparing row %d".formatted(i),
                        A.getRow(i).toArray(), Ab.getRow(i).toArray(),
                        0.000_001
                );
            for(int i = 0; i < 3; i++)
                assertArrayEquals("Comparing column %d".formatted(i),
                        A.getColumn(i).toArray(), Ab.getColumn(i).toArray(),
                        0.000_001
                );
        }

        { // Pretty prints
            System.out.println("Initial matrix A:");
            System.out.println(Matrix.toString(A));
            System.out.println("-----------");
            System.out.println("Find Q and R such that A = QR, R upper triangular, Q orthogonal");
            System.out.println("-----------");
            System.out.println("Found Q:");
            System.out.println(Matrix.toString(Q));
            System.out.println("Found R:");
            System.out.println(Matrix.toString(Q.transpose().composeLeft(A)));
        }
    }

    @Test
    public void qrdecomposition_allSteps() {
        double[][] data = {
                { 12, -51, 4 },
                { 6, 167, -68 },
                { -4, 24, -41 }
        };
        var A = Matrices.ofTable(data);
        var Q = QRDecomposer.qOfQRDecomposition(A);

        { // check Q
            assertArrayEquals(new double[] { 6.0/7, 69.0/175, -58.0/175 }, Q.getRow(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 3.0/7, -158.0/175, 6.0/175 }, Q.getRow(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { -2.0/7, -6.0/35, -33.0/35 }, Q.getRow(2).toArray(), 0.000_001);

            assertArrayEquals(new double[] { 6.0/7, 3.0/7, -2.0/7 }, Q.getColumn(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 69.0/175, -158.0/175, -6.0/35 }, Q.getColumn(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { -58.0/175, 6.0/175, -33.0/35 }, Q.getColumn(2).toArray(), 0.000_001);
        }

        { // check R
            var R = Q.transpose().composeLeft(A);
            assertArrayEquals(new double[] { 14, 21, -14 }, R.getRow(0).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, -175, 70 }, R.getRow(1).toArray(), 0.000_001);
            assertArrayEquals(new double[] { 0, 0, 35 }, R.getRow(2).toArray(), 0.000_001);
        }

        { // check A = QR
            var R = Q.transpose().composeLeft(A);
            var Ab = Q.composeLeft(R);

            for(int i = 0; i < 3; i++)
                assertArrayEquals("Comparing row %d".formatted(i),
                        A.getRow(i).toArray(), Ab.getRow(i).toArray(),
                        0.000_001
                );
            for(int i = 0; i < 3; i++)
                assertArrayEquals("Comparing column %d".formatted(i),
                        A.getColumn(i).toArray(), Ab.getColumn(i).toArray(),
                        0.000_001
                );
        }

        { // Pretty prints
            System.out.println("Initial matrix A:");
            System.out.println(Matrix.toString(A));
            System.out.println("-----------");
            System.out.println("Find Q and R such that A = QR, R upper triangular, Q orthogonal");
            System.out.println("-----------");
            System.out.println("Found Q:");
            System.out.println(Matrix.toString(Q));
            System.out.println("Found R:");
            System.out.println(Matrix.toString(Q.transpose().composeLeft(A)));
        }
    }

}
