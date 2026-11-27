#pragma once
#include <vector>

using Matrix = std::vector<std::vector<int>>;

Matrix transpose(const Matrix& m);
int trace(const Matrix& m);
