#include <stdio.h>
static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

        #include "text.h"

        int main(void) {
            CHECK(count_vowels("hello") == 2);
            CHECK(count_vowels("AEIOU") == 5);
            CHECK(count_vowels("Apple Orange") == 5);
            CHECK(count_vowels("rhythm") == 0);
            if (failures) { printf("%d check(s) failed\n", failures); return 1; }
            printf("all checks passed\n");
            return 0;
        }
