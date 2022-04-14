package matrix;

import java.util.Arrays;
import java.util.PrimitiveIterator;
import java.util.stream.IntStream;

class Matrices {

    static Matrix ofTable(double[][] data) {
        assert Arrays.stream(data).mapToInt(arr -> arr.length).allMatch(i -> i == data[0].length)
                : "The data table is not rectangular";

        class Impl implements CoordinatesBased {
            @Override
            public double getEntry(int rowIndex, int colIndex) {
                return data[rowIndex][colIndex];
            }

            @Override
            public int rowSize() {
                return data[0].length;
            }

            @Override
            public int colSize() {
                return data.length;
            }
        }
        return new Impl();
    }

    static Matrix mult(Matrix left, Matrix right) {
        /*
            dim(A * B) = ( row(B) , col(A) )
        */
        return new Matrix() {
            @Override
            public int rowSize() {
                return right.rowSize();
            }

            @Override
            public int colSize() {
                return left.colSize();
            }

            @Override
            public VectorView getColumn(int index) {
                /* col(A * B, i) = A * col(B, i) */
                return left.apply(right.getColumn(index));
            }

            @Override
            public VectorView getRow(int index) {
                /* row(A * B, i) = B^T * row(A, i) */
                return right.transpose().apply(left.getRow(index));
            }

            @Override
            public Matrix transpose() {
                /* (A * B)^T = B^T * A^T */
                return mult(right.transpose(), left.transpose());
            }
        };
    }

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
