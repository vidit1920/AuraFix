package shop;

/**
 * Discount calculation for the shop.
 *
 * THE BUG: applyDiscount() subtracts the percent as a flat amount instead of
 * applying it as a percentage of the price. So "10% off" a $200 item wrongly
 * returns $190 (200 - 10) instead of $180. The code runs fine and returns a
 * plausible-looking number, which is exactly why it slips past casual testing
 * and needs a real fix to the formula, not a guard.
 */
public class Discount {

    /**
     * Returns the price after applying a {@code percent}% discount.
     * Example: applyDiscount(200, 10) should return 180.0 (10% off 200).
     */
    public static double applyDiscount(double price, double percent) {
        return price - percent; // BUG: should be price * (1 - percent / 100.0)
    }
}
