package matrix;

import org.junit.Test;

import static matrix.Matrices.mult;
import static matrix.Matrices.ofTable;
import static org.junit.Assert.*;

public class MatricesTest {

    @Test
    public void ofTable_createsCorrectly_givenDataTable() {
        double[][] data = {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        var matrix = ofTable(data);
        System.out.println(Matrix.toString(matrix));

        assertArrayEquals(new double[]{1, 2}, matrix.getRow(0).toArray(), 0.5);
        assertArrayEquals(new double[]{5, 6}, matrix.getRow(2).toArray(), 0.5);
        assertArrayEquals(new double[]{1, 3, 5}, matrix.getColumn(0).toArray(), 0.5);
    }

    @Test
    public void mult_multipliesCorrectly() {
        double[][] dataA = {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        double[][] dataB = {
                {0, 1, 0, 3,  1},
                {1, 2, 4, -5, 0}
        };
        Matrix A = ofTable(dataA);
        Matrix B = ofTable(dataB);
        Matrix AB = mult(A, B);

        assertEquals(3, A.colSize());
        assertEquals(2, A.rowSize());
        assertEquals(2, B.colSize());
        assertEquals(5, B.rowSize());
        assertEquals(5, AB.rowSize());
        assertEquals(3, AB.colSize());

        assertArrayEquals(new double[]{2, 5, 8, -7, 1}, AB.getRow(0).toArray(), 0.5);
        assertArrayEquals(new double[]{5, 11, 17}, AB.getColumn(1).toArray(), 0.5);
    }

    @Test
    public void transposeOfProduct_doesNotThrow_StackOverflow() {
        double[][] dataA = {
                {1, 2},
                {3, 4},
                {5, 6}
        };
        double[][] dataB = {
                {0, 1, 0, 3,  1},
                {1, 2, 4, -5, 0}
        };
        Matrix A = ofTable(dataA);
        Matrix B = ofTable(dataB);
        assertSame(A, A.transpose().transpose());

        Matrix AB = mult(A, B);
        assertNotSame(AB, AB.transpose().transpose());
        assertEquals(5, AB.rowSize());
        assertEquals(3, AB.colSize());
        assertEquals(3, AB.transpose().rowSize());
        assertEquals(5, AB.transpose().colSize());

        assertArrayEquals(new double[]{5, 11, 17}, AB.getColumn(1).toArray(), 0.5);

        for(int i = 0; i < (1 << 23); i++) {
            AB = AB.transpose().transpose();
            assertArrayEquals(new double[]{5, 11, 17}, AB.getColumn(1).toArray(), 0.5);
        }
    }

    @Test
    public void householder_mustHaveCorrectProperties() {
        double[] v = { 1, 2, -1 };
        /*
            v v^T = [ 1  2  -1 ]
                    [ 2  4  -2 ]
                    [ -1 -2  1 ]
         */
        var m = Matrices.householder(v);

        System.out.println(Matrix.toString(m));

        assertSame(m, m.transpose());
        assertArrayEquals(new double[] { -4, -7, 4 }, m.getRow(1).toArray(), 0.5);
    }

}
