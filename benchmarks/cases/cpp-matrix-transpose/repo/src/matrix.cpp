#include "matrix.hpp"

Matrix transpose(const Matrix& m) {
    if (m.empty()) {
        return {};
    }
    const std::size_t rows = m.size();
    const std::size_t cols = m[0].size();
    Matrix t(rows, std::vector<int>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            t.at(i).at(j) = m.at(j).at(i);
        }
    }
    return t;
}

int trace(const Matrix& m) {
    int sum = 0;
    for (std::size_t i = 0; i < m.size() && i < m[i].size(); ++i) {
        sum += m[i][i];
    }
    return sum;
}
