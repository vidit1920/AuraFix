package shop;

import org.junit.Test;
import static org.junit.Assert.assertEquals;

/**
 * Includes the tests that currently FAIL because of the discount bug.
 * After AuraFix applies the correct percentage formula, all should pass.
 */
public class DiscountTest {

    @Test
    public void noDiscountReturnsFullPrice() {
        // 0% off -> full price (passes even with the bug: 100 - 0 == 100).
        assertEquals(100.0, Discount.applyDiscount(100, 0), 0.001);
    }

    @Test
    public void tenPercentOff200() {
        // THIS TEST CURRENTLY FAILS: 10% off 200 should be 180, bug returns 190.
        assertEquals(180.0, Discount.applyDiscount(200, 10), 0.001);
    }

    @Test
    public void halfOff() {
        // THIS TEST CURRENTLY FAILS: 50% off 50 should be 25, bug returns 0.
        assertEquals(25.0, Discount.applyDiscount(50, 50), 0.001);
    }
}
