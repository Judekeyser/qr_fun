package matrix;

import java.util.PrimitiveIterator;
import java.util.StringJoiner;

public interface Matrix {

    int rowSize();
    int colSize();

    VectorView getColumn(int index);
    VectorView getRow(int index);

    default Matrix transpose() {
        class TransposeMatrix implements Matrix {
            /*
                For a matrix A,
                    row(A) = col(A^T)
                and
                    col(A) = row(A^T)

                Similarly,
                    row(A, i) = col(A^T, i)
                and
                    col(A, i) = row(A^T, i)

                Furthermore, the transpose operator is its own invert (lax)
                    A = (A^T)^T
             */

            @Override
            public int rowSize() {
                return Matrix.this.colSize();
            }

            @Override
            public int colSize() {
                return Matrix.this.rowSize();
            }

            @Override
            public VectorView getColumn(int index) {
                return Matrix.this.getRow(index);
            }

            @Override
            public VectorView getRow(int index) {
                return Matrix.this.getColumn(index);
            }

            @Override
            public Matrix transpose() {
                return Matrix.this;
            }
        } return new TransposeMatrix();
    }

    default VectorView apply(VectorView vec) {
        /*
            The computation of A * v for a given vector v,
            follows the rule
                (A * v)_i = A_{i*} * v
            that is : the ith coordinate is the product of the ith row of A against v
         */
        var length = colSize();
        return () -> new PrimitiveIterator.OfDouble() {
            int cursor = 0;

            @Override
            public double nextDouble() {
                var rowView = Matrix.this.getRow(cursor++).iterator();
                double stack = 0D;
                for(var it = vec.iterator(); it.hasNext();) {
                    assert rowView.hasNext();
                    stack += it.nextDouble() * rowView.nextDouble();
                }
                return stack;
            }

            @Override
            public boolean hasNext() {
                return cursor < length;
            }
        };
    }

    default Matrix composeLeft(Matrix rightFactor) {
        /*
            Composing to the left with respect to A, is the arrow
                B --> A * B

            This is a group action (but we do not encode the identity in a specific way).
         */
        record Prod(Matrix left, Matrix right) implements ProductOfTwo{}
        return new Prod(this, rightFactor);
    }

    static String toString(Matrix matrix) { // Java doesn't allow interfaces to extend methods of Object
        var big = new StringJoiner(System.lineSeparator());
        for (int i = 0; i < matrix.rowSize(); i++) {
            var joiner = new StringJoiner("; ");
            for (var it = matrix.getRow(i).iterator(); it.hasNext(); )
                joiner.add("%.4f".formatted(it.nextDouble()));
            big.add("[ %s ]".formatted(joiner.toString()));
        }
        return big.toString();
    }

    static Matrix ofTable(double[][] data) {
        return CoordinatesBased.ofTable(data);
    }

}
