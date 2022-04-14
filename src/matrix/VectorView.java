package matrix;

import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.stream.StreamSupport;

@FunctionalInterface
public interface VectorView  {

    PrimitiveIterator.OfDouble iterator();

    default double[] toArray() {
        Iterable<Double> self = this::iterator; // Java wants the cast
        return StreamSupport.stream(self.spliterator(), false)
                .mapToDouble(Double::valueOf)
                .toArray();
    }

    default VectorView then(VectorView next) {
        /*
            Aggregation of two vectors:
                a ° b is a followed by b

            This operator is useful for aggregating subcolumns during an algorithmic computation

            Since VectorView do not know about their length,
            it would be too hard to implement a custom sub-view operation for results of then.
            We thus do not override that operator.
         */
        return () -> {
            class Impl implements PrimitiveIterator.OfDouble {
                boolean switchOn = true;
                OfDouble itCursor = VectorView.this.iterator();
                @Override
                public double nextDouble() {
                    if(hasNext())
                        return itCursor.nextDouble();
                    else throw new NoSuchElementException();
                }

                /**
                 * Watch out, hasNext here has a side-effect
                 * on the object, which makes it tricky. The hasNext() method must be called
                 * somehow on the nextDouble() implementation, to guarantee coherence.
                 */
                @Override
                public boolean hasNext() {
                    if(switchOn) {
                        boolean hasNext = itCursor.hasNext();
                        if(! hasNext) {
                            switchOn = false;
                            itCursor = next.iterator();
                            return hasNext();
                        } return true;
                    } else return itCursor.hasNext();
                }
            } return new Impl();
        };
    }

    default VectorView subView(int skip, int length) {
        /*
            Sub-viewing a vector corresponds the "inverse" operation of ° (then).

            This operation is useful when extracting subcolumns or rows to create submatrices virtually.

            Watch out that subviewing in its default form, is a O(n) operation
            (we need to skip some elements non lazily). This is somehow similar to a head-tail list implementation.

         */
        return new VectorView() {
            @Override
            public PrimitiveIterator.OfDouble iterator() {
                class Impl implements PrimitiveIterator.OfDouble {
                    int cursor;
                    final PrimitiveIterator.OfDouble it = VectorView.this.iterator();
                    Impl() {
                        cursor = 0;
                        for(; it.hasNext(); cursor++)
                            if (cursor == skip) break;
                            else it.nextDouble();
                    }

                    @Override
                    public double nextDouble() {
                        assert it.hasNext();
                        cursor++;
                        return it.nextDouble();
                    }

                    @Override
                    public boolean hasNext() {
                        return (cursor < skip + length) && it.hasNext();
                    }
                }
                return new Impl();
            }

            @Override
            public VectorView subView(int s, int l) {
                /*
                    Sub-viewing a view can be achieved more directly by going up to the source
                    vector and constraining a bit more:

                        subview(subview(v, skip1, length1), skip2, length2)
                            = subview(v, skip1+skip2, length2)
                 */
                assert l <= length;
                int newSkip = skip + s; assert newSkip >= 0; assert newSkip <= l;
                return VectorView.this.subView(newSkip, l);
            }
        };
    }
}
