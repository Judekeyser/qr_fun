package matrix;

import java.util.PrimitiveIterator;

interface CoordinatesBased extends Matrix {
    @Override
    int rowSize();

    @Override
    int colSize();

    double getEntry(int rowIndex, int colIndex);

    @Override
    default VectorView getColumn(int index) {
        return getColumnSlice(index, 0, colSize());
    }

    @Override
    default VectorView getRow(int index) {
        return getRowSlice(index, 0, rowSize());
    }

    private VectorView getRowSlice(int index, int skip, int length) {
        /*
            VectorViews of a coordinate based matrix
            can be encoded more directly using the coordinates.

            It also allows better definition of the subview process, as a O(1) operation.
         */
        assert length + skip <= rowSize();
        class RowSlice implements VectorView {
            @Override
            public PrimitiveIterator.OfDouble iterator() {
                class RowBasedVue implements PrimitiveIterator.OfDouble {
                    int cursor = skip;
                    @Override
                    public double nextDouble() {
                        return getEntry(index, cursor++);
                    }

                    @Override
                    public boolean hasNext() {
                        return cursor < length + skip;
                    }
                }
                return new RowBasedVue();
            }

            @Override
            public VectorView subView(int s, int l) {
                return getRowSlice(index, s+skip, l);
            }
        }
        return new RowSlice();
    }

    private VectorView getColumnSlice(int index, int skip, int length) {
        /*
            VectorViews of a coordinate based matrix
            can be encoded more directly using the coordinates.

            It also allows better definition of the subview process, as a O(1) operation.
         */
        assert skip + length <= colSize();
        class ColSlice implements VectorView {
            @Override
            public PrimitiveIterator.OfDouble iterator() {
                class ColBasedView implements PrimitiveIterator.OfDouble {
                    int cursor = skip;
                    @Override
                    public double nextDouble() {
                        return getEntry(cursor++, index);
                    }

                    @Override
                    public boolean hasNext() {
                        return cursor < skip + length;
                    }
                }
                return new ColBasedView();
            }

            @Override
            public VectorView subView(int s, int l) {
                return getColumnSlice(index, s+skip, l);
            }
        }
        return new ColSlice();
    }
}
