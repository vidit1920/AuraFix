package collections;

public class BoundedStack<T> {
    private final Object[] items;
    private int size;

    public BoundedStack(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive");
        }
        items = new Object[capacity];
    }

    public void push(T item) {
        if (size > items.length) {
            throw new IllegalStateException("stack is full");
        }
        items[size++] = item;
    }

    @SuppressWarnings("unchecked")
    public T pop() {
        if (size == 0) {
            throw new IllegalStateException("stack is empty");
        }
        T item = (T) items[--size];
        items[size] = null;
        return item;
    }

    public boolean isFull() {
        return size > items.length;
    }

    public int size() {
        return size;
    }
}
