package matrix;

class MatrixByDelegation implements Matrix {
    private final Matrix matrix;
    MatrixByDelegation(Matrix matrix) {
        this.matrix = matrix;
    }

    @Override
    public int rowSize() {
        return matrix.rowSize();
    }

    @Override
    public int colSize() {
        return matrix.colSize();
    }

    @Override
    public VectorView getColumn(int index) {
        return matrix.getColumn(index);
    }

    @Override
    public VectorView getRow(int index) {
        return matrix.getRow(index);
    }

    @Override
    public Matrix transpose() {
        return matrix.transpose();
    }

    @Override
    public VectorView apply(VectorView vec) {
        return matrix.apply(vec);
    }

    @Override
    public Matrix composeLeft(Matrix rightFactor) {
        return matrix.composeLeft(rightFactor);
    }
}
