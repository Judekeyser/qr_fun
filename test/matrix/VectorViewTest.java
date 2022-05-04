package matrix;

import org.junit.Test;

import java.util.stream.DoubleStream;

import static matrix.Matrix.ofTable;
import static org.junit.Assert.assertArrayEquals;

public class VectorViewTest {

    static VectorView ofData(double... data) {
        return () -> DoubleStream.of(data).iterator();
    }

    @Test
    public void subview_canCompute_givenDefaultImpl() {
        var view = ofData(1, 2, 3, 4, 5, 6, 7, 8, 9);

        assertArrayEquals(new double[] { 1, 2, 3 }, view.subView(0, 3).toArray(), 0.5);
        assertArrayEquals(new double[] {}, view.subView(1, 0).toArray(), 0.5);
        assertArrayEquals(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, view.subView(0, 8).toArray(), 0.5);

        assertArrayEquals(new double[] { 3, 4, 5, 6, 7, 8 }, view.subView(0, 8).subView(2, 6).toArray(), 0.5);
        assertArrayEquals(new double[] { 3, 4, 5, 6, 7 }, view.subView(0, 8).subView(2, 5).toArray(), 0.5);
    }

    @Test
    public void subview_canCompute_givenTableImpl() {
        var matrix = ofTable(new double[][] {{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }});
        var view = matrix.getRow(0);

        assertArrayEquals(new double[] { 1, 2, 3 }, view.subView(0, 3).toArray(), 0.5);
        assertArrayEquals(new double[] {}, view.subView(1, 0).toArray(), 0.5);
        assertArrayEquals(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, view.subView(0, 8).toArray(), 0.5);

        assertArrayEquals(new double[] { 3, 4, 5, 6, 7, 8 }, view.subView(0, 8).subView(2, 6).toArray(), 0.5);
        assertArrayEquals(new double[] { 3, 4, 5, 6, 7 }, view.subView(0, 8).subView(2, 5).toArray(), 0.5);
    }

    @Test
    public void then_shouldConcat_givenTwo() {
        var v1 = ofData(5,4,3);
        var v2 = ofData(2,1);
        var v3 = ofData();

        var v = v3.then(v1).then(v2).then(v3);

        assertArrayEquals(new double[] { 5, 4, 3, 2, 1 }, v.toArray(), 0.5);
    }

}
