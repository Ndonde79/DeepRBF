#include "deeprbf/domain_decomposition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace deeprbf {

namespace {

void split_kd_tree(
    std::vector<int>& current_indices,
    const std::vector<std::array<double, 3>>& centers,
    std::vector<std::vector<int>>& leaves,
    std::size_t target_size
) {
    if (current_indices.size() <= target_size || current_indices.size() <= 1U) {
        leaves.push_back(current_indices);
        return;
    }

    double min_pt[3] = {
        centers[static_cast<std::size_t>(current_indices[0])][0],
        centers[static_cast<std::size_t>(current_indices[0])][1],
        centers[static_cast<std::size_t>(current_indices[0])][2]
    };
    double max_pt[3] = {min_pt[0], min_pt[1], min_pt[2]};

    for (int idx : current_indices) {
        const auto& c = centers[static_cast<std::size_t>(idx)];
        for (int d = 0; d < 3; ++d) {
            min_pt[d] = std::min(min_pt[d], c[d]);
            max_pt[d] = std::max(max_pt[d], c[d]);
        }
    }

    double max_len = -1.0;
    int best_axis = 0;
    for (int d = 0; d < 3; ++d) {
        const double len = max_pt[d] - min_pt[d];
        if (len > max_len) { max_len = len; best_axis = d; }
    }

    std::sort(current_indices.begin(), current_indices.end(),
              [&centers, best_axis](int a, int b) {
                  return centers[static_cast<std::size_t>(a)][best_axis] <
                         centers[static_cast<std::size_t>(b)][best_axis];
              });

    const std::size_t mid = current_indices.size() / 2U;
    std::vector<int> left(
        current_indices.begin(),
        current_indices.begin() + static_cast<std::ptrdiff_t>(mid));
    std::vector<int> right(
        current_indices.begin() + static_cast<std::ptrdiff_t>(mid),
        current_indices.end());

    split_kd_tree(left, centers, leaves, target_size);
    split_kd_tree(right, centers, leaves, target_size);
}

} // namespace

std::vector<Subdomain> build_overlapping_subdomains(
    const std::vector<std::array<double, 3>>& centers,
    const DDMConfig& cfg
) {
    std::vector<Subdomain> subs;
    if (centers.empty()) return subs;

    const int num_partitions = std::max(1, cfg.nx * cfg.ny * cfg.nz);
    const std::size_t target_size =
        std::max<std::size_t>(1, centers.size() / static_cast<std::size_t>(num_partitions));

    std::vector<int> initial_indices(centers.size());
    for (std::size_t i = 0; i < centers.size(); ++i)
        initial_indices[i] = static_cast<int>(i);

    std::vector<std::vector<int>> leaves;
    split_kd_tree(initial_indices, centers, leaves, target_size);

    for (const auto& leaf_indices : leaves) {
        if (leaf_indices.empty()) continue;

        double min_pt[3] = {
            centers[static_cast<std::size_t>(leaf_indices[0])][0],
            centers[static_cast<std::size_t>(leaf_indices[0])][1],
            centers[static_cast<std::size_t>(leaf_indices[0])][2]
        };
        double max_pt[3] = {min_pt[0], min_pt[1], min_pt[2]};

        for (int idx : leaf_indices) {
            const auto& c = centers[static_cast<std::size_t>(idx)];
            for (int d = 0; d < 3; ++d) {
                min_pt[d] = std::min(min_pt[d], c[d]);
                max_pt[d] = std::max(max_pt[d], c[d]);
            }
        }

        for (int d = 0; d < 3; ++d) {
            double extent = max_pt[d] - min_pt[d];
            if (extent <= 0.0) extent = std::max(1.0, cfg.overlap);
            min_pt[d] -= cfg.overlap * extent;
            max_pt[d] += cfg.overlap * extent;
        }

        Subdomain sd;
        sd.indices = leaf_indices;

        for (std::size_t i = 0; i < centers.size(); ++i) {
            const auto& p = centers[i];
            if (p[0] >= min_pt[0] && p[0] <= max_pt[0] &&
                p[1] >= min_pt[1] && p[1] <= max_pt[1] &&
                p[2] >= min_pt[2] && p[2] <= max_pt[2]) {
                sd.overlap_indices.push_back(static_cast<int>(i));
            }
        }

        subs.push_back(std::move(sd));
    }

    return subs;
}

} // namespace deeprbf