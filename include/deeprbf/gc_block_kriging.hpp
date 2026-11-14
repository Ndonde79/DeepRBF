// deeprbf/gc_block_kriging.hpp
#pragma once

#include "deeprbf/gc_geostatistics.hpp"
#include <array>
#include <vector>

namespace deeprbf {

/// A rectangular block in 3D space.
struct Block {
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
    std::array<int, 3> discretization{2, 2, 2};   // {nx, ny, nz}
};

struct BlockKrigingResult {
    std::vector<double> estimate;
    std::vector<double> variance;
    std::vector<int> n_samples;
};

BlockKrigingResult block_kriging(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Block>& blocks,
    int max_neighbors,
    int min_neighbors,
    double search_radius,
    double variogram_range,
    double sill,
    double nugget
);

} // namespace deeprbf