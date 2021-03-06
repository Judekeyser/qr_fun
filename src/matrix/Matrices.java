package matrix;

import java.util.PrimitiveIterator;
import java.util.stream.IntStream;

class Matrices {

    static Matrix householder(double[] d) {
        /*
            Factory to compute the Householder matrix of shape
                H = Id - 2 v v^T

            H is symmetric.
            We implement H using a coordinates based approach
         */
        class Impl implements CoordinatesBased, SymmetricMatrix {
            @Override
            public int rowSize() {
                return d.length;
            }

            @Override
            public int colSize() {
                /* col(A) = row(A) if A is symmetric */
                return SymmetricMatrix.super.colSize();
            }

            @Override
            public double getEntry(int row, int col) {
                return (row == col ? 1 : 0) - 2 * d[row]*d[col];
            }

            @Override
            public VectorView getColumn(int index) {
                return CoordinatesBased.super.getColumn(index);
            }

            @Override
            public VectorView getRow(int index) {
                return SymmetricMatrix.super.getRow(index);
            }
        } return new Impl();
    }

    static Matrix upperAugmentWithIdentity(Matrix m, int dim) {
        record IdentitySlice(int length, int onePosition) implements VectorView {
            @Override
            public PrimitiveIterator.OfDouble iterator() {
                return IntStream.iterate(0, i -> i + 1)
                        .mapToDouble(i -> i == onePosition ? 1D : 0D)
                        .limit(length)
                        .iterator();
            }

            @Override
            public VectorView subView(int skip, int l) {
                return new IdentitySlice(l, IdentitySlice.this.onePosition - skip);
            }
        }

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
                    return new IdentitySlice(dim, index);
                } else {
                    var subCol = m.getColumn(index - shift);
                    var slice = new IdentitySlice(dim - m.colSize(), index);
                    return slice.then(subCol);
                }
            }

            @Override
            public VectorView getRow(int index) {
                var shift = dim - m.colSize();
                if(index < shift) {
                    return new IdentitySlice(dim, index);
                } else {
                    var subRow = m.getRow(index - shift);
                    var slice = new IdentitySlice(dim - m.rowSize(), index);
                    return slice.then(subRow);
                }
            }
        };
    }

}
