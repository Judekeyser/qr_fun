package matrix;

interface SymmetricMatrix extends Matrix {
    /*
        For a symmetric matrix, we have the bi-directional relations
            col = row
        and
            col(i) = row(i)
     */
    @Override
    default int rowSize() {
        return colSize();
    }

    @Override
    default int colSize() {
        return rowSize();
    }

    @Override
    default VectorView getColumn(int index) {
        return getRow(index);
    }

    @Override
    default VectorView getRow(int index) {
        return getColumn(index);
    }

    /*
        A symmetric matrix always equals its transpose
     */

    @Override
    default Matrix transpose() {
        return this;
    }
}
