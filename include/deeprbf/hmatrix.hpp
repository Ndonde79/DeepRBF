#pragma once
// 3.5.0: hierarchical (H-matrix) approximation of the RBF kernel matrix K_ij = phi(|x_i - x_j|).
//
// The points are ordered by a k-d cluster tree (bisection of the longest box side at the median, leaves of at most
// `leaf_size` points). A pair of clusters (t, s) whose boxes satisfy  min(diam t, diam s) <= eta * dist(t, s)  is
// "admissible": the kernel is smooth between them and the block is compressed to a low-rank product U V^T by adaptive
// cross approximation (ACA, partial pivoting) to a relative tolerance `tol`, then recompressed by QR + SVD. Blocks
// that are not admissible down to the leaves are the near field: they are evaluated exactly, either stored or
// recomputed on every product with the tabulated kernel when they do not fit in `near_memory_mb`.
// The product K x then costs about O(n log n) instead of O(n^2), with a relative error of order `tol`.

#include "deeprbf/core.hpp"
#include "deeprbf/kernel_table.hpp"

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace deeprbf {

struct HMatrixOptions {
    double tol = 1.0e-9;          // ACA / recompression tolerance, relative to each block's Frobenius norm
    double eta = 1.5;             // admissibility: min(diam) <= eta * dist
    int leaf_size = 64;           // maximum points per leaf cluster
    double near_memory_mb = 2048; // store the exact near field if it fits, else recompute it on every product
};

struct HMatrixStats {
    int n = 0;
    int n_lowrank = 0, n_near = 0;
    int max_rank = 0;
    double mean_rank = 0.0;
    double lowrank_mb = 0.0, near_mb = 0.0;
    bool near_stored = false;
    double build_seconds = 0.0;
    double compression = 0.0;     // stored doubles / n^2
};

class HMatrix {
public:
    HMatrix(const std::vector<Point3>& points, const std::string& kernel, double epsilon,
            const HMatrixOptions& options = {});

    // y = K x (no regularisation on the diagonal; the caller adds it).
    void apply(const std::vector<double>& x, std::vector<double>& y) const;

    int size() const { return static_cast<int>(points_.size()); }
    const HMatrixStats& stats() const { return stats_; }
    // Debug: the block holding entry (i, j) in the original ordering: {is_far, rows, cols, rank, count}.
    std::vector<double> locate(int i, int j) const;

private:
    struct Node { int begin = 0, end = 0, left = -1, right = -1; double lo[3], hi[3]; };
    struct LowRank { int t, s; Eigen::MatrixXd U, V; };
    struct Near { int t, s; std::vector<double> D; };   // D row-major |t| x |s| when stored

    int build_tree(int begin, int end);
    void build_blocks(int t, int s, std::vector<std::pair<int,int>>& far, std::vector<std::pair<int,int>>& near) const;
    bool admissible(int t, int s) const;
    void aca(LowRank& block) const;
    double k(int i, int j) const { return kernel_(fast_distance(pts_[i], pts_[j])); }

    std::vector<Point3> points_;   // original order
    std::vector<Point3> pts_;      // tree order
    std::vector<int> perm_;        // pts_[i] = points_[perm_[i]]
    FastKernel kernel_;
    HMatrixOptions opt_;
    std::vector<Node> nodes_;
    std::vector<LowRank> far_;
    std::vector<Near> near_;
    HMatrixStats stats_;
};

}  // namespace deeprbf
