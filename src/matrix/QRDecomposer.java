package matrix;

import java.util.Arrays;

import static java.lang.Math.*;

class QRDecomposer {
    /* QR decomposition using Householder reflections */

    static Matrix step(Matrix M, int rank) {
        /*
            Compute a single step H of the QR-decomposition algorithm using Householder
            matrices.

            The complexity of `x` computation is as big as the complexity required to compute (a subview of)
            one of the column of the matrix M.
                When the matrix M is backed by an array, this is cheap.
                When it's not the case (for example, M = mult(M1,M2)), we should provide a way to make this step
                as cheap as O(n), as it's morally n scalar products with the rank-th column of M2. Since evaluation
                are lazy, the naive multiplication operator may recompute that column over and over again, yielding
                to a blowing algorithm. Caching the rows of the left-factors is less sensitive, because we know
                they will be backed by an array somehow (it's always Householder matrices on the left).

            The complexity of computing the cancellation vector is O(n), as it only requires the computation
            of a euclidean norm.

            Householder matrix creation is cheap once the previous cancellation vector is known.
            Similarly, the augment within an identity matrix is cheap.
                Those two operations could be improved in the case where M is in Hessenberg form,
                because in that case, we know the resulting matrix acts very lightly by composeLeft
                (it only impacts at most two columns, and not the entire right-factor).
                Here again, the composeLeft operator should be overwritten to benefit from the theorem.
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

            As explained in step, we *must* rely on a multiplication principle that allow
            caching of the columns of the left member. This is because otherwise, we risk to create
            an algorithm that blows-up in complexity. We therefore override the ProductOfTwo trait
            locally to bring a cache feature on the right-hand side.
         */
        assert M.rowSize() == M.colSize();
        if (M.rowSize() == 1) return M;

        class CachedProduct implements ProductOfTwo {
            record Cache(int columnIndex, VectorView data) {}

            Cache cache = null;
            final Matrix left, right;
            CachedProduct(Matrix left, Matrix right) {
                this.left = left;
                this.right = right;
            }
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
                if(cache == null || cache.columnIndex != index) {
                    var data = right.getColumn(index).toArray();
                    cache = new Cache(index, () -> Arrays.stream(data).iterator());
                } assert cache.columnIndex == index;

                return left.apply(cache.data);
            }
        }

        Matrix[] chain = new Matrix[M.rowSize() - 1];
        chain[0] = step(M, 0);
        Matrix cumul = chain[0];

        for(int i = 1; i < chain.length; i++) {
            chain[i] = step(new CachedProduct(cumul, M), i);
            cumul = new CachedProduct(chain[i], cumul);
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
