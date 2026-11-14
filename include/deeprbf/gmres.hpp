#pragma once

#include <functional>
#include <string>
#include <vector>

namespace deeprbf {

using ApplyOperatorFn =
    std::function<void(const std::vector<double>&, std::vector<double>&)>;

using ApplyPreconditionerFn =
    std::function<void(const std::vector<double>&, std::vector<double>&)>;

struct GMRESConfig {
    int restart = 40;
    int max_iter = 200;
    double tol = 1.0e-6;
};

bool gmres_solve(
    int n,
    const ApplyOperatorFn& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const GMRESConfig& cfg,
    std::string& status,
    const ApplyPreconditionerFn& M = ApplyPreconditionerFn()
);

} // namespace deeprbf