package matrix;

import java.util.Arrays;

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

}
