#include "text.h"

int count_vowels(const char *s) {
    int n = 0;
    for (; *s; s++) {
        switch (*s) {
            case 'a': case 'e': case 'i': case 'o': case 'u':
                n++;
                break;
            default:
                break;
        }
    }
    return n;
}
