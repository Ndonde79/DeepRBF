#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace deeprbf {

using Point3 = std::array<double, 3>;
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct RBFConfig {
    std::string kernel = "biharmonic";
    double epsilon = 1.0;
    double regularization = 1.0e-4;
};

} // namespace deeprbf