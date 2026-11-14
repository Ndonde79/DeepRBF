// deeprbf/gc_change_of_support.hpp
#pragma once

#include "deeprbf/gc_block_kriging.hpp"  // for Block, Point3
#include <string>
#include <vector>

namespace deeprbf {

struct ChangeOfSupportResult {
    double block_mean = 0.0;
    double block_variance = 0.0;
    double variance_reduction_factor = 1.0;
};

double compute_variance_reduction_factor(
    const Block& block,
    double range,
    double sill,
    double nugget
);

ChangeOfSupportResult affine_correction(
    double point_mean,
    double point_variance,
    double vrf
);

ChangeOfSupportResult lognormal_correction(
    double point_mean,
    double point_variance,
    double vrf
);

// affine/lognormal: one point draw per row. average: one jointly simulated
// value per equal-volume midpoint node; no second variance reduction.
std::vector<double> apply_change_of_support_to_realizations(
    const std::vector<std::vector<double>>& point_realizations,
    double point_mean,
    double point_variance,
    const Block& block,
    double range,
    double sill,
    double nugget,
    const std::string& method = "affine"
);

} // namespace deeprbf
