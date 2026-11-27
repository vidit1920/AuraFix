package text;

import org.junit.Test;
import static org.junit.Assert.*;

public class TextUtilsTest {
    @Test
    public void mixedCasePalindrome() {
        assertTrue(TextUtils.isPalindrome("Racecar"));
    }

    @Test
    public void sentencePalindrome() {
        assertTrue(TextUtils.isPalindrome("A man, a plan, a canal: Panama"));
    }

    @Test
    public void notAPalindrome() {
        assertFalse(TextUtils.isPalindrome("hello"));
    }

    @Test
    public void countsWords() {
        assertEquals(3, TextUtils.countWords("  one two  three "));
    }
}
