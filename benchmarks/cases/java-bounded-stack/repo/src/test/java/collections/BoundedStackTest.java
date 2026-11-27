package collections;

import org.junit.Test;
import static org.junit.Assert.*;

public class BoundedStackTest {
    @Test
    public void popReturnsLastPushed() {
        BoundedStack<String> s = new BoundedStack<>(3);
        s.push("a");
        s.push("b");
        assertEquals("b", s.pop());
        assertEquals(1, s.size());
    }

    @Test
    public void isFullAtCapacity() {
        BoundedStack<Integer> s = new BoundedStack<>(2);
        s.push(1);
        s.push(2);
        assertTrue(s.isFull());
    }

    @Test(expected = IllegalStateException.class)
    public void pushOnFullStackThrows() {
        BoundedStack<Integer> s = new BoundedStack<>(1);
        s.push(1);
        s.push(2);
    }
}
