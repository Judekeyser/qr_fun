package matrix;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.ToDoubleFunction;

import static java.lang.Math.*;

class EigenValueSolver {
    /* Use the QR decomposer to infer eigen values */

    static int flushEigenvalues(Matrix M, ToDoubleFunction<double[][]> shift, int iterationBound, double sensitivity, double[] eigenValues) {
        /*
            This algorithm implements the iterative QR-eigenvalue iteration
                C = M
                while C is not upper triangular:
                    C = C - shift
                    C = QR
                    C = RQ + shift

                Every-reassignment to C performs a data copy,
                    because we cannot foresee the number of different chains the iteration will require
                    and we want data to be kept as grouped in memory as possible.

                The shift is currently chosen as being the Wilkinson's shift, perturbed by a random noise.
                    The shift is defined by the heuristic
                        s = C_{nn} - sign(d) b^2 / (|d| + sqrt(d^2 + b^2))
                    with
                        b = average(C_{(n-1)n} , C_{n(n-1)})
                    and
                        d = (C_{(n-1)(n-1) - C_{nn}) / 2
                    A random noise close to 1 comes as a perturbation on the shift. The shift may include randomized
                    elements and takes the data as input.

                    The shift implementation doesn't require any kind of specific matrix operation
                    to be defined. Because of the previous remark on the data model, we already
                    backed our computations with a double[][]. Accessing the diagonal elements is
                    straightforward.
         */
        assert M.rowSize() == M.colSize();
        assert eigenValues.length == M.rowSize();

        double[][] data = matrixToData(M);

        while(iterationBound-- > 0 && !isUpperTriangular(data, sensitivity)) {
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
            double s = shift.applyAsDouble(data);
            shifts(data, -s);
            var cursor = Matrices.ofTable(data);
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
            shifts(data, s);
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

    private static void shifts(double[][] data, double shift) {
        for(int i = 0; i < data.length; i++)
            data[i][i] += shift;
    }

}
