package matrix;

import java.util.Arrays;
import java.util.Comparator;

import static java.lang.Math.*;

class EigenValueSolver {
    /* Use the QR decomposer to infer eigen values */

    static int flushEigenvalues(Matrix M, int iterationBound, double sensitivity, double[] eigenValues) {
        /*
            This algorithm implements the iterative QR-eigenvalue iteration
                C = M
                while C is not upper triangular:
                    C = QR
                    C = RQ

                Every-reassignment to C performs a data copy,
                    because we cannot foresee the number of different chains the iteration will require
                    and we want data to be kept as grouped in memory as possible.

         */
        assert M.rowSize() == M.colSize();
        assert eigenValues.length == M.rowSize();

        double[][] data = matrixToData(M);
        var cursor = Matrices.ofTable(matrixToData(M));

        while(iterationBound-- > 0) {
            /*
                Given H0, H1, H2, ..., Hk we know that
                    Q = (Hk * ... * H2 * H1 * H0)^T
                and Q is orthogonal. In particular, we know that
                    R = Q^T * A
                      = Hk * ... * H2 * H1 * H0 * A
                and
                    RQ = Q' * A * Q'^T
                    Q' = Q^T = (Hk * ... * (H2 * (H1 * H0))...)
                In order to mitigate lazy computations effects, we reduce this computation
                factor-by-factor, and store the information in a data source matrix.
             */
            var householderList = QRDecomposer.qOfQRDecomposition(cursor);
            Matrix qBis; { // Compute (Hk * ... * (H2 * (H1 * H0))...)
                var it = householderList.iterator();
                double[][] qBisData = matrixToData(it.next());
                while(it.hasNext())
                    qBisData = matrixToData(it.next().composeLeft(Matrices.ofTable(qBisData)));
                qBis = Matrices.ofTable(qBisData);
            }
            data = matrixToData(qBis.composeLeft(cursor));
            data = matrixToData(Matrices.ofTable(data).composeLeft(qBis.transpose()));
            if(isUpperTriangular(data, sensitivity)) break;
            cursor = Matrices.ofTable(data);
        }

        System.arraycopy(
                Arrays.stream(
                        diagonal(data)).boxed()
                        .sorted(Comparator.<Double>comparingDouble(Math::abs).reversed())
                        .mapToDouble(Double::doubleValue)
                        .toArray(),
                0, eigenValues, 0, eigenValues.length
        );
        return iterationBound;
    }

    private static double[][] matrixToData(Matrix M) {
        assert M.rowSize() == M.colSize();
        int size = M.rowSize();
        double[][] data = new double[size][];

        for(int i = 0; i < size; i++)
            data[i] = M.getRow(i).toArray();

        return data;
    }

    private static boolean isUpperTriangular(double[][] data, double sensitivity) {
        boolean isDiagonal = true;
        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < i; j++) {
                isDiagonal &= abs(data[i][j]) < sensitivity;
            }
        }
        return isDiagonal;
    }

    private static double[] diagonal(double[][] data) {
        double[] diagonal = new double[data.length];
        for(int i = 0; i < diagonal.length; i++)
            diagonal[i] = data[i][i];
        return diagonal;
    }

}
