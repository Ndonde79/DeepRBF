#pragma once

#include "deeprbf/gc_geostatistics.hpp"
#include <vector>
#include <string>

namespace deeprbf {

LinearModelOfCoregionalization fit_lmc_from_experimental_variograms(
    const std::vector<VariogramResult>& direct_variograms,
    const std::vector<VariogramResult>& cross_variograms,
    const std::vector<int>& categories,
    int n_structures = 2,
    const std::string& model_type = "spherical",
    double tolerance = 1e-6,
    int max_iter = 100
);

} // namespace deeprbf