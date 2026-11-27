#include <cmath>
#include <cstdio>
#include "stats.hpp"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)

int main() {
    CHECK(std::fabs(median({3, 1, 2}) - 2.0) < 1e-9);
    CHECK(std::fabs(median({4, 1, 3, 2}) - 2.5) < 1e-9);
    CHECK(std::fabs(mean({1, 2, 3, 4}) - 2.5) < 1e-9);
    if (failures) { std::printf("%d check(s) failed\n", failures); return 1; }
    std::printf("all checks passed\n");
    return 0;
}
