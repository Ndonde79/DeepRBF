#pragma once

#include <string>
#include <vector>

namespace deeprbf {

// Dense linear solve for row-major dense matrix A (n x n) and RHS b (n x 1).
// On success, b is overwritten with the solution and backend_used is populated.
bool solve_dense_linear_system(
    std::vector<double>& A,
    std::vector<double>& b,
    int n,
    std::string& backend_used
);

class MKLSolver {
public:
    MKLSolver() = default;
    ~MKLSolver() = default;

    bool solve(
        std::vector<double>& A,
        std::vector<double>& b,
        int n,
        std::string& backend_used
    ) const;
};

} // namespace deeprbf