#pragma once

#include <array>
#include <vector>

namespace deeprbf {

struct Subdomain {
    std::vector<int> indices;
    std::vector<int> overlap_indices;
};

struct DDMConfig {
    int nx = 2;
    int ny = 2;
    int nz = 2;
    double overlap = 5.0;
};

std::vector<Subdomain> build_overlapping_subdomains(
    const std::vector<std::array<double, 3>>& centers,
    const DDMConfig& cfg
);

} // namespace deeprbf