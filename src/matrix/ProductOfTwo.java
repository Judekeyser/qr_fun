package matrix;

interface ProductOfTwo extends Matrix {
    /*
        The product of two matrices, in its simplest form
     */

    Matrix left();
    Matrix right();

    @Override
    default int rowSize() {
        return right().rowSize();
    }

    @Override
    default int colSize() {
        return left().colSize();
    }

    @Override
    default VectorView getColumn(int index) {
        /* col(A * B, i) = A * col(B, i) */
        return left().apply(right().getColumn(index));
    }

    @Override
    default VectorView getRow(int index) {
        /* row(A * B, i) = B^T * row(A, i) */
        return right().transpose().apply(left().getRow(index));
    }

    @Override
    default Matrix transpose() {
        /* (A * B)^T = B^T * A^T */
        return right().transpose().composeLeft(left().transpose());
    }

}
