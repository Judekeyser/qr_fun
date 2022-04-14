package matrix;

import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.stream.IntStream;

import static java.lang.Math.*;

class QRDecomposer {
    /* QR decomposition using Householder reflections */

    static VectorView identitySlice(int length, int onePosition) {
        return () -> IntStream.iterate(0, i -> i + 1)
                .mapToDouble(i -> i == onePosition ? 1D : 0D)
                .limit(length)
                .iterator();
    }

    static Matrix upperAugmentWithIdentity(Matrix m, int dim) {
        assert dim >= m.rowSize();
        assert dim >= m.colSize();
        return new Matrix() {
            /*
                This implementation takes the matrix m and injects it
                as a lower sub-matrix of the identity (of dimension dim).

                The identity matrix here is not required: we only need to generate
                identity-slices and glue them with the slices of m
             */
            @Override
            public int rowSize() {
                return dim;
            }

            @Override
            public int colSize() {
                return dim;
            }

            @Override
            public VectorView getColumn(int index) {
                var shift = dim - m.rowSize();
                if(index < shift) {
                    return identitySlice(dim, index);
                } else {
                    var subCol = m.getColumn(index - shift);
                    var slice = identitySlice(dim - m.colSize(), index);
                    return slice.then(subCol);
                }
            }

            @Override
            public VectorView getRow(int index) {
                var shift = dim - m.colSize();
                if(index < shift) {
                    return identitySlice(dim, index);
                } else {
                    var subRow = m.getRow(index - shift);
                    var slice = identitySlice(dim - m.rowSize(), index);
                    return slice.then(subRow);
                }
            }
        };
    }

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
        return upperAugmentWithIdentity(householderMatrix, M.colSize());
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

        class WithLastColumnCache extends MatrixByDelegation {
            /*
                Those matrices are caching the last column that was required.

                This is done to speed up the computation of col(E * A, i),
                which requires row(E)-times the same column of A.

                Caching means storing temporarily the column of A as a double[], in a strong reference.

                If columns of A are pre-encoded in advance, the advantage is really low.
                However, in the QR-decomposition algorithm, it won't be the case very long.
                This allows to not re-process the entire chain of matrices to compute, for every row of E.
             */
            record LastColumnComputation(int rank, double[] data) {}

            WithLastColumnCache(Matrix wrapped) { super(wrapped); }
            LastColumnComputation lastComputedColumn = null;

            @Override
            public VectorView getColumn(int index) {
                if(lastComputedColumn == null || lastComputedColumn.rank != index)
                    lastComputedColumn = new LastColumnComputation(index, super.getColumn(index).toArray());
                assert lastComputedColumn.rank == index;
                var data = lastComputedColumn.data;
                return () -> Arrays.stream(data).iterator();
            }
        }

        Matrix[] chain = new Matrix[M.rowSize() - 1];
        chain[0] = step(M, 0);
        Matrix cumul = new WithLastColumnCache(chain[0]);

        for(int i = 1; i < chain.length; i++) {
            chain[i] = step(cumul.composeLeft(M), i);
            cumul = new WithLastColumnCache(chain[i].composeLeft(cumul));
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
