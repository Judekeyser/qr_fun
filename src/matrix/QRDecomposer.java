package matrix;

import static java.lang.Math.*;

class QRDecomposer {
    /* QR decomposition using Householder reflections */

    static Matrix step(Matrix M, int rank) {
        /*
            Compute a single step H of the QR-decomposition algorithm using Householder
            matrices.
         */
        assert M.rowSize() == M.colSize();
        int size = M.colSize() - rank;
        var x = M.getColumn(rank).subView(rank, size).toArray();
        mutateToCancellingVector(x);
        var householderMatrix = Matrices.householder(x);
        return Matrices.upperAugmentWithIdentity(householderMatrix, M.colSize());
    }

    static Matrix qOfQRDecomposition(Matrix M) {
        /*
            Iterates the full steps of the QR-decomposition algorithm using Householder
            matrices.

            The result is the orthonormal matrix Q, as the transpose of the product of H_i.
            The matrix R can be recovered directly from the equation
                M = QR
                    R = Q^T M
         */
        assert M.rowSize() == M.colSize();
        if (M.rowSize() == 1) return M;

        Matrix[] chain = new Matrix[M.rowSize() - 1];
        chain[0] = step(M, 0);
        Matrix cumul = chain[0];

        for(int i = 1; i < chain.length; i++) {
            chain[i] = step(cumul.composeLeft(M), i);
            cumul = chain[i].composeLeft(cumul);
        }

        cumul = chain[0];
        for(int i = 1; i < chain.length; i++)
            cumul = chain[i].composeLeft(cumul);

        return cumul.transpose();
    }

    private static void mutateToCancellingVector(double[] x) {
        /*
            This method mutates, because due to usage, there is no reason to duplicate the data.
         */
        double xNormTail = normOfTail(x);
        x[0] += hypot(x[0], xNormTail) * (x[0] < 0D ? 1D : -1D);
        double uNorm = hypot(x[0], xNormTail);
        for(int i = 0; i < x.length; i++)
            x[i] /= uNorm;
    }

    private static double normOfTail(double[] x) {
        double s = 0D;
        for(int i = 1; i < x.length; i++)
            s += pow(x[i], 2);
        return sqrt(s);
    }

}
