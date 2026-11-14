// DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
// Sparse direct path for compactly supported RBF kernels (3.3.3).
//
// For a kernel that is exactly zero beyond a support radius the RBF matrix K is sparse. The augmented system
//   [[K + reg*I, F], [F^T, 0]] [w; a] = [v; 0]
// is solved by a Schur complement on the m drift columns F (m = 0, 1 or 4): S = K + reg*I is SPD for a strictly
// positive definite kernel at distinct points, so it is factored once by a sparse LDL^T (AMD ordering), then
//   S Z = F,  S y = v,  (F^T Z) a = F^T y,  w = y - Z a.
// This avoids factoring the indefinite saddle matrix, whose dense drift rows/columns cause heavy fill-in in LU.
#pragma once

#include "deeprbf/kernels.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace deeprbf {

// Support radius of a compactly supported kernel (kernel == 0 for d >= radius), or 0 if the kernel is global.
// Add new compact kernels (e.g. Wendland) here and in kernels.hpp.
inline double compact_support_radius(KernelId id, double epsilon) {
    switch (id) {
    case KernelId::Spherical: return epsilon;
    default: return 0.0;
    }
}
inline bool kernel_is_compact(KernelId id) { return compact_support_radius(id, 1.0) > 0.0; }

// Uniform grid hash with cell size equal to the support radius: every neighbour within the radius of a point lies in
// the 3x3x3 block of cells around it. Points are stored sorted by cell (contiguous coordinates per cell).
class CompactNeighbourGrid {
public:
    CompactNeighbourGrid(const std::vector<std::array<double, 3>>& points, double cell_size);

    // Calls f(j, distance) for every stored point j with distance < radius (radius <= cell size).
    template <class F>
    void for_each_within(const std::array<double, 3>& q, double radius, F&& f) const {
        const double r2 = radius * radius;
        const std::int64_t cx = cell_coord(q[0], 0), cy = cell_coord(q[1], 1), cz = cell_coord(q[2], 2);
        for (std::int64_t dx = -1; dx <= 1; ++dx)
            for (std::int64_t dy = -1; dy <= 1; ++dy)
                for (std::int64_t dz = -1; dz <= 1; ++dz) {
                    const auto it = cells_.find(key(cx + dx, cy + dy, cz + dz));
                    if (it == cells_.end()) continue;
                    for (int s = it->second.first; s < it->second.second; ++s) {
                        const auto& p = sorted_[static_cast<std::size_t>(s)];
                        const double ex = q[0] - p[0], ey = q[1] - p[1], ez = q[2] - p[2];
                        const double d2 = ex * ex + ey * ey + ez * ez;
                        if (d2 < r2) f(order_[static_cast<std::size_t>(s)], std::sqrt(d2));
                    }
                }
    }
    double cell_size() const { return h_; }

private:
    struct KeyHash {
        std::size_t operator()(const std::array<std::int64_t, 3>& c) const noexcept {
            std::uint64_t h = static_cast<std::uint64_t>(c[0]) * 0x9E3779B97F4A7C15ULL;
            h ^= static_cast<std::uint64_t>(c[1]) * 0xC2B2AE3D27D4EB4FULL + (h << 6) + (h >> 2);
            h ^= static_cast<std::uint64_t>(c[2]) * 0x165667B19E3779F9ULL + (h << 6) + (h >> 2);
            return static_cast<std::size_t>(h);
        }
    };
    static std::array<std::int64_t, 3> key(std::int64_t x, std::int64_t y, std::int64_t z) { return {x, y, z}; }
    std::int64_t cell_coord(double v, int axis) const;

    double h_;
    std::array<double, 3> origin_{};
    std::vector<std::array<double, 3>> sorted_;
    std::vector<int> order_;
    std::unordered_map<std::array<std::int64_t, 3>, std::pair<int, int>, KeyHash> cells_;
};

struct SparseCompactSolveInfo {
    long long nnz_lower = 0;       // stored entries of the lower triangle of S
    double assembly_seconds = 0.0;
    double factor_seconds = 0.0;
    double solve_seconds = 0.0;
    double residual = 0.0;         // ||[S w + F a - v; F^T w]|| / ||v||
};

// Solves the augmented system for a compact kernel. Returns weights of size n + 4: w, then drift coefficients for
// [1, x, y, z] (the last 4 - m are zero). drift_columns must be 0, 1 or 4. Throws std::runtime_error on failure.
std::vector<double> solve_compact_sparse_schur(const std::vector<std::array<double, 3>>& centers,
                                               const std::vector<double>& values,
                                               KernelId kernel, double epsilon, double regularization,
                                               int drift_columns, const CompactNeighbourGrid& grid,
                                               SparseCompactSolveInfo* info = nullptr);

// Evaluates sum_j w_j k(|q - c_j|) + drift over the centers within the support radius of each query (OpenMP).
std::vector<double> predict_compact_sparse(const std::vector<std::array<double, 3>>& queries,
                                           const std::vector<double>& weights, KernelId kernel, double epsilon,
                                           const CompactNeighbourGrid& grid);

} // namespace deeprbf
