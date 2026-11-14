// deeprbf/gc_block_kriging.cpp
#include "deeprbf/gc_block_kriging.hpp"
#include "deeprbf/kernels.hpp"
#include "deeprbf/spatial_hash.hpp"
#include "deeprbf/block_support.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace deeprbf {

namespace {
    constexpr double REGULARIZATION = 1e-10;
    constexpr int MAX_LOCAL_NEIGHBORS = 64;
}

/// Generate discretization points inside a block (regular grid).
static std::vector<Point3> discretize_block(const Block& blk) {
    return block_midpoints(blk);
}

/// Average covariance between a block and a sample point.
static double block_sample_covariance(
    const Block& blk,
    const Point3& sample,
    double range, double sill, double nugget
) {
    auto disc_pts = discretize_block(blk);
    if (disc_pts.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& p : disc_pts) {
        double h = euclidean_dist(p, sample) / range;
        sum += spherical_covariance(h, sill, nugget);
    }
    return sum / static_cast<double>(disc_pts.size());
}

/// Within-block variance.
static double block_variance(
    const Block& blk,
    double range, double sill, double nugget
) {
    auto disc_pts = discretize_block(blk);
    const size_t m = disc_pts.size();
    if (m == 0) return sill;

    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double h = euclidean_dist(disc_pts[i], disc_pts[j]) / range;
            sum += spherical_covariance(h, sill, nugget);
        }
    }
    return sum / static_cast<double>(m * m);
}

/// Local ordinary kriging for a block.
static bool local_block_ordinary_kriging(
    const Block& target_block,
    const std::vector<Point3>& cond_pts,
    const std::vector<double>& cond_vals,
    double range, double sill, double nugget,
    double& estimate, double& variance
) {
    const int m = static_cast<int>(cond_pts.size());
    if (m < 1 || m > MAX_LOCAL_NEIGHBORS) return false;

    Eigen::MatrixXd A(m+1, m+1);
    Eigen::VectorXd b(m+1);
    A.setZero(); b.setZero();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double h = euclidean_dist(cond_pts[i], cond_pts[j]) / range;
            A(i,j) = spherical_covariance(h, sill, nugget);
        }
        A(i,m) = 1.0;
        A(m,i) = 1.0;

        double h0 = block_sample_covariance(target_block, cond_pts[i], range, sill, nugget);
        b(i) = h0;
    }
    A(m,m) = 0.0;
    b(m) = 1.0;

    A.topLeftCorner(m,m).diagonal().array() += REGULARIZATION;

    Eigen::FullPivLU<Eigen::MatrixXd> factor(A);
    if(factor.rank()!=m+1)return false;
    Eigen::VectorXd lambda = factor.solve(b);
    if (!lambda.allFinite()||(A*lambda-b).norm()>1e-9*std::max(1.,b.norm())) return false;

    estimate = 0.0;
    for (int i = 0; i < m; ++i) estimate += lambda(i) * cond_vals[i];

    double block_var = block_variance(target_block, range, sill, nugget);
    variance = block_var - lambda.head(m).dot(b.head(m)) - lambda(m);
    if(!std::isfinite(variance)||variance < -1e-9*std::max(1.,sill))return false;
    variance = std::max(variance, 0.);
    return true;
}

// =============================================================================
// Public API
// =============================================================================
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
) {
    if(!std::isfinite(variogram_range)||variogram_range<=0)throw std::invalid_argument("Positive finite variogram range required");
    for(const auto& b:blocks){
        block_midpoints(b); // Reject invalid support before the OpenMP loop.
        for(double v:{b.xmin,b.xmax,b.ymin,b.ymax,b.zmin,b.zmax})if(!std::isfinite(v))throw std::invalid_argument("Nonfinite block extent");
        if(b.xmax<b.xmin||b.ymax<b.ymin||b.zmax<b.zmin)throw std::invalid_argument("Inverted block extent");
        size_t count=1;for(int n:b.discretization){if(n<1||n>32)throw std::invalid_argument("Block discretization outside 1..32");count*=n;}
        if(count>4096)throw std::invalid_argument("Block quadrature exceeds 4096 nodes");
    }
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    BlockKrigingResult result;
    const size_t n_blocks = blocks.size();
    result.estimate.assign(n_blocks, std::numeric_limits<double>::quiet_NaN());
    result.variance.assign(n_blocks, std::numeric_limits<double>::quiet_NaN());
    result.n_samples.assign(n_blocks, 0);

    if (sample_xyz.empty() || blocks.empty()) return result;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);
    double range = std::max(variogram_range, 1e-9);

    SpatialHash hash(std::max(search_radius / 4.0, 1.0));
    hash.build(sample_xyz, sample_values);
    for(const auto& b:blocks)hash.validate_target({b.xmin+(b.xmax-b.xmin)*.5,b.ymin+(b.ymax-b.ymin)*.5,b.zmin+(b.zmax-b.zmin)*.5});

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n_blocks); ++i) {
        const Block& blk = blocks[static_cast<size_t>(i)];
        Point3 centre = { (blk.xmin + blk.xmax) * 0.5,
                          (blk.ymin + blk.ymax) * 0.5,
                          (blk.zmin + blk.zmax) * 0.5 };

        auto neighbours = hash.query(centre, search_radius, max_neighbors);
        if (neighbours.size() < static_cast<size_t>(min_neighbors)) continue;

        std::vector<Point3> cond_pts;
        std::vector<double> cond_vals;
        cond_pts.reserve(neighbours.size());
        cond_vals.reserve(neighbours.size());
        for (const auto& nb : neighbours) {
            cond_pts.push_back(hash.points[static_cast<size_t>(nb.index)]);
            cond_vals.push_back(hash.values[static_cast<size_t>(nb.index)]);
        }

        double est, var;
        if (local_block_ordinary_kriging(blk, cond_pts, cond_vals,
                                         range, sill, nugget, est, var)) {
            result.estimate[static_cast<size_t>(i)] = est;
            result.variance[static_cast<size_t>(i)] = var;
            result.n_samples[static_cast<size_t>(i)] = static_cast<int>(neighbours.size());
        }
    }
    return result;
}

} // namespace deeprbf
