package text;

public final class TextUtils {
    private TextUtils() {}

    /** True if s reads the same backwards, ignoring case and anything that isn't a letter or digit. */
    public static boolean isPalindrome(String s) {
        String cleaned = s.replaceAll("[^A-Za-z0-9]", "").toLowerCase();
        int i = 0;
        int j = cleaned.length() - 1;
        while (i < j) {
            if (cleaned.charAt(i) != cleaned.charAt(j)) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }

    public static int countWords(String s) {
        String trimmed = s.trim();
        return trimmed.isEmpty() ? 0 : trimmed.split("\\s+").length;
    }
}
