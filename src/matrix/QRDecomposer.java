package matrix;

import java.util.Arrays;
import java.util.List;

import static java.lang.Double.isFinite;
import static java.lang.Math.*;
import static java.util.Collections.singletonList;

interface QRDecomposer {

    default List<Matrix> householderSuccessiveReflections(Matrix M) {
        if (M.rowSize() == 1) return singletonList(M);

        record CachedProduct(Matrix left, Matrix right) implements ProductOfTwo {
            @Override
            public Matrix left() {
                return left;
            }

            @Override
            public Matrix right() {
                return right;
            }

            @Override
            public VectorView getColumn(int index) {
                var data = right.getColumn(index).toArray();
                return left.apply(() -> Arrays.stream(data).iterator());
            }
        }

        Matrix[] chain = new Matrix[M.rowSize() - 1];
        chain[0] = step(M, 0);
        Matrix cumul = new CachedProduct(chain[0], M);

        for(int i = 1; i < chain.length; i++) {
            chain[i] = step(cumul, i);
            cumul = new CachedProduct(chain[i], cumul);
        }
        return Arrays.asList(chain);
    }

    default Matrix step(Matrix M, int rank) {
        int size = M.colSize() - rank;
        var x = M.getColumn(rank).subView(rank, size).toArray();
        mutateToCancellingVector(x);
        var householderMatrix = Matrices.householder(x);
        return Matrices.upperAugmentWithIdentity(householderMatrix, M.colSize());
    }

    private static void mutateToCancellingVector(double[] x) {
        double xNormTail = normOfTail(x);
        x[0] += hypot(x[0], xNormTail) * (x[0] < 0D ? 1D : -1D);
        double invertUNorm = 1D / hypot(x[0], xNormTail);
        if (isFinite(invertUNorm))
            for(int i = 0; i < x.length; i++) x[i] *= invertUNorm;
    }

    private static double normOfTail(double[] x) {
        double s = 0D;
        for(int i = 1; i < x.length; i++) s += pow(x[i], 2);
        return sqrt(s);
    }

}
