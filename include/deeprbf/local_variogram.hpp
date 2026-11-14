#ifndef DEEPRBF_LOCAL_VARIOGRAM_HPP
#define DEEPRBF_LOCAL_VARIOGRAM_HPP

#include <vector>

#include "deeprbf/core.hpp"

namespace deeprbf {

struct LocalVariogramParams {
    double sill = 1.0;
    double range = 100.0;
    double nugget = 0.0;
    bool valid = false;
    double rss = 0.0;
    double r_squared = 0.0;
};

// Fit a spherical variogram to the samples lying inside `search_radius`
// around `target`. Returns valid=false if there are fewer than `min_points`
// samples or the fit fails.
LocalVariogramParams fit_local_variogram_spherical(
    const std::vector<Point3>& samples,
    const std::vector<double>& values,
    const Point3& target,
    double search_radius,
    int min_points = 10,
    int n_lags = 5,
    double sill_guess = -1.0,
    double range_guess = -1.0,
    double nugget_guess = -1.0
);

// Batch variant (OpenMP-parallel over targets).
std::vector<LocalVariogramParams> fit_local_variograms_batch(
    const std::vector<Point3>& samples,
    const std::vector<double>& values,
    const std::vector<Point3>& targets,
    double search_radius,
    int min_points = 10,
    int n_lags = 5
);

} // namespace deeprbf

#endif