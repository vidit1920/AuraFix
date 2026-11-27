#include <stdio.h>
static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

        #include "stats.h"

        int main(void) {
            int mixed[] = {3, 9, 4};
            CHECK(max_value(mixed, 3) == 9);
            int negative[] = {-7, -2, -9};
            CHECK(max_value(negative, 3) == -2);
            CHECK(average(mixed, 3) > 5.33 && average(mixed, 3) < 5.34);
            if (failures) { printf("%d check(s) failed\n", failures); return 1; }
            printf("all checks passed\n");
            return 0;
        }
