#include "stats.hpp"

#include <algorithm>
#include <stdexcept>

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median of empty list");
    }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
    return values[n / 2];
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("mean of empty list");
    }
    double sum = 0;
    for (double v : values) {
        sum += v;
    }
    return sum / values.size();
}
