#include <cstdio>
#include <exception>
#include "matrix.hpp"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)

int main() {
    try {
        Matrix square = {{1, 2}, {3, 4}};
        CHECK((transpose(square) == Matrix{{1, 3}, {2, 4}}));
        Matrix wide = {{1, 2, 3}, {4, 5, 6}};
        CHECK((transpose(wide) == Matrix{{1, 4}, {2, 5}, {3, 6}}));
        CHECK(trace(square) == 5);
    } catch (const std::exception& e) {
        std::printf("FAIL: exception: %s\n", e.what());
        return 1;
    }
    if (failures) { std::printf("%d check(s) failed\n", failures); return 1; }
    std::printf("all checks passed\n");
    return 0;
}
