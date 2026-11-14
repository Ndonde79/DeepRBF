// DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
#include "deeprbf/sparse_compact.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace deeprbf {

namespace {
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
constexpr std::int64_t CELL_COORD_LIMIT = std::int64_t(1) << 52;
} // namespace

std::int64_t CompactNeighbourGrid::cell_coord(double v, int axis) const {
    const double c = std::floor((v - origin_[static_cast<std::size_t>(axis)]) / h_);
    if (!std::isfinite(c)) throw std::invalid_argument("CompactNeighbourGrid: non-finite coordinate");
    // Far-away queries are clamped; they find no occupied cells (all centers lie well inside the limit).
    if (c > static_cast<double>(CELL_COORD_LIMIT)) return CELL_COORD_LIMIT;
    if (c < -static_cast<double>(CELL_COORD_LIMIT)) return -CELL_COORD_LIMIT;
    return static_cast<std::int64_t>(c);
}

CompactNeighbourGrid::CompactNeighbourGrid(const std::vector<std::array<double, 3>>& points, double cell_size)
    : h_(cell_size) {
    if (!std::isfinite(cell_size) || !(cell_size > 0.0))
        throw std::invalid_argument("CompactNeighbourGrid: cell size must be finite and positive");
    if (points.empty()) return;
    origin_ = points[0];
    for (const auto& p : points)
        for (int k = 0; k < 3; ++k) origin_[static_cast<std::size_t>(k)] = std::min(origin_[static_cast<std::size_t>(k)], p[static_cast<std::size_t>(k)]);
    for (const auto& p : points)
        for (int k = 0; k < 3; ++k)
            if ((p[static_cast<std::size_t>(k)] - origin_[static_cast<std::size_t>(k)]) / h_ > 0.5 * static_cast<double>(CELL_COORD_LIMIT))
                throw std::invalid_argument("CompactNeighbourGrid: extent too large for the support radius");

    const std::size_t n = points.size();
    std::vector<std::array<std::int64_t, 3>> keys(n);
    for (std::size_t i = 0; i < n; ++i)
        keys[i] = key(cell_coord(points[i][0], 0), cell_coord(points[i][1], 1), cell_coord(points[i][2], 2));
    order_.resize(n);
    std::iota(order_.begin(), order_.end(), 0);
    std::stable_sort(order_.begin(), order_.end(), [&keys](int a, int b) {
        return keys[static_cast<std::size_t>(a)] < keys[static_cast<std::size_t>(b)];
    });
    sorted_.resize(n);
    for (std::size_t s = 0; s < n; ++s) sorted_[s] = points[static_cast<std::size_t>(order_[s])];
    cells_.reserve(n);
    std::size_t s = 0;
    while (s < n) {
        const auto& k = keys[static_cast<std::size_t>(order_[s])];
        std::size_t e = s + 1;
        while (e < n && keys[static_cast<std::size_t>(order_[e])] == k) ++e;
        cells_.emplace(k, std::make_pair(static_cast<int>(s), static_cast<int>(e)));
        s = e;
    }
}

std::vector<double> solve_compact_sparse_schur(const std::vector<std::array<double, 3>>& centers,
                                               const std::vector<double>& values,
                                               KernelId kernel, double epsilon, double regularization,
                                               int drift_columns, const CompactNeighbourGrid& grid,
                                               SparseCompactSolveInfo* info) {
    const double radius = compact_support_radius(kernel, epsilon);
    if (!(radius > 0.0)) throw std::invalid_argument("solve_compact_sparse_schur: kernel is not compactly supported");
    if (radius > grid.cell_size() * (1.0 + 1e-12))
        throw std::invalid_argument("solve_compact_sparse_schur: grid cell size smaller than the support radius");
    if (drift_columns != 0 && drift_columns != 1 && drift_columns != 4)
        throw std::invalid_argument("solve_compact_sparse_schur: drift_columns must be 0, 1 or 4");
    const int n = static_cast<int>(centers.size());
    const int m = drift_columns;
    if (n == 0 || values.size() != centers.size()) throw std::invalid_argument("solve_compact_sparse_schur: bad input sizes");
    if (n <= m) throw std::runtime_error("solve_compact_sparse_schur: need more centers than drift terms");

    SparseCompactSolveInfo local;
    // ---- Assembly: lower triangle of S = K + reg*I, column i holds rows j >= i with |c_i - c_j| < radius.
    auto t0 = Clock::now();
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int i = 0; i < n; ++i) {
        auto& col = cols[static_cast<std::size_t>(i)];
        grid.for_each_within(centers[static_cast<std::size_t>(i)], radius, [&](int j, double d) {
            if (j < i) return;
            double v = evaluate_kernel_id(kernel, d, epsilon);
            if (j == i) v = evaluate_kernel_id(kernel, 0.0, epsilon) + regularization;
            col.emplace_back(j, v);
        });
        std::sort(col.begin(), col.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    long long nnz = 0;
    for (const auto& c : cols) nnz += static_cast<long long>(c.size());
    if (nnz > static_cast<long long>(std::numeric_limits<int>::max()))
        throw std::length_error("solve_compact_sparse_schur: too many nonzeros for 32-bit sparse indices");
    SpMat S(n, n);
    S.resizeNonZeros(static_cast<Eigen::Index>(nnz));
    {
        int* outer = S.outerIndexPtr();
        int* inner = S.innerIndexPtr();
        double* val = S.valuePtr();
        outer[0] = 0;
        for (int i = 0; i < n; ++i) outer[i + 1] = outer[i] + static_cast<int>(cols[static_cast<std::size_t>(i)].size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            int p = outer[i];
            for (const auto& e : cols[static_cast<std::size_t>(i)]) { inner[p] = e.first; val[p] = e.second; ++p; }
        }
    }
    std::vector<std::vector<std::pair<int, double>>>().swap(cols);
    local.nnz_lower = nnz;
    local.assembly_seconds = seconds_since(t0);

    // ---- Factor S once (sparse LDL^T, AMD fill-reducing ordering).
    t0 = Clock::now();
    Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>> ldlt;
    ldlt.compute(S);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error("solve_compact_sparse_schur: sparse LDL^T factorisation failed");
    {
        const Eigen::VectorXd D = ldlt.vectorD();
        const double dmax = D.cwiseAbs().maxCoeff();
        if (!D.allFinite() || D.minCoeff() <= 1e-14 * std::max(1.0, dmax))
            throw std::runtime_error("solve_compact_sparse_schur: K + reg*I is not numerically positive definite "
                                     "(duplicate or nearly coincident centers? use regularization > 0)");
    }
    local.factor_seconds = seconds_since(t0);

    // ---- Drift basis, centred and scaled for conditioning: [1, (x-cx)/s, (y-cy)/s, (z-cz)/s][:, :m].
    t0 = Clock::now();
    std::array<double, 3> cen{0.0, 0.0, 0.0};
    double scale = 1.0;
    if (m == 4) {
        for (const auto& c : centers) for (int k = 0; k < 3; ++k) cen[static_cast<std::size_t>(k)] += c[static_cast<std::size_t>(k)];
        for (int k = 0; k < 3; ++k) cen[static_cast<std::size_t>(k)] /= n;
        double ext = 0.0;
        for (const auto& c : centers)
            for (int k = 0; k < 3; ++k) ext = std::max(ext, std::abs(c[static_cast<std::size_t>(k)] - cen[static_cast<std::size_t>(k)]));
        scale = ext > 0.0 ? ext : 1.0;
    }
    Eigen::MatrixXd F(n, m);
    for (int i = 0; i < n; ++i) {
        if (m >= 1) F(i, 0) = 1.0;
        if (m == 4)
            for (int k = 0; k < 3; ++k)
                F(i, k + 1) = (centers[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] - cen[static_cast<std::size_t>(k)]) / scale;
    }
    const Eigen::VectorXd v = Eigen::Map<const Eigen::VectorXd>(values.data(), n);

    Eigen::MatrixXd Z;
    Eigen::MatrixXd G;
    Eigen::FullPivLU<Eigen::MatrixXd> Glu;
    if (m > 0) {
        Z = ldlt.solve(F);
        if (ldlt.info() != Eigen::Success || !Z.allFinite())
            throw std::runtime_error("solve_compact_sparse_schur: solve for drift columns failed");
        G = F.transpose() * Z;                       // m x m, SPD when F has full column rank
        Glu.compute(G);
        if (Glu.rank() < m)
            throw std::runtime_error("solve_compact_sparse_schur: drift columns are rank deficient (coplanar or collinear centers?)");
    }
    // Saddle solve with the cached factors: S w + F a = rt, F^T w = rb.
    auto solve_aug = [&](const Eigen::VectorXd& rt, const Eigen::VectorXd& rb, Eigen::VectorXd& w, Eigen::VectorXd& a) {
        Eigen::VectorXd y = ldlt.solve(rt);
        if (m > 0) {
            a = Glu.solve(F.transpose() * y - rb);
            w = y - Z * a;
        } else {
            a.resize(0);
            w = std::move(y);
        }
    };
    Eigen::VectorXd w, a;
    solve_aug(v, Eigen::VectorXd::Zero(m), w, a);
    const double vnorm = std::max(v.norm(), std::numeric_limits<double>::min());
    auto residual = [&](Eigen::VectorXd& rt, Eigen::VectorXd& rb) {
        rt = v - S.selfadjointView<Eigen::Lower>() * w;
        if (m > 0) { rt -= F * a; rb = -(F.transpose() * w); } else rb.resize(0);
        return std::sqrt(rt.squaredNorm() + rb.squaredNorm()) / vnorm;
    };
    Eigen::VectorXd rt, rb;
    double res = residual(rt, rb);
    for (int it = 0; it < 3 && res > 1e-13; ++it) {   // iterative refinement with the existing factorisation
        Eigen::VectorXd dw, da;
        solve_aug(rt, rb, dw, da);
        w += dw;
        if (m > 0) a += da;
        const double r_new = residual(rt, rb);
        if (!(r_new < res)) { w -= dw; if (m > 0) a -= da; residual(rt, rb); break; }
        res = r_new;
    }
    if (!w.allFinite() || (m > 0 && !a.allFinite()) || !(res < 1e-6))
        throw std::runtime_error("solve_compact_sparse_schur: solution failed the residual check (relative residual " +
                                 std::to_string(res) + ")");
    local.residual = res;
    local.solve_seconds = seconds_since(t0);

    std::vector<double> out(static_cast<std::size_t>(n) + 4U, 0.0);
    for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = w[i];
    if (m >= 1) out[static_cast<std::size_t>(n)] = a[0];
    if (m == 4) {
        // Back to the unscaled basis [1, x, y, z].
        double c0 = a[0];
        for (int k = 0; k < 3; ++k) {
            const double ak = a[k + 1] / scale;
            out[static_cast<std::size_t>(n + 1 + k)] = ak;
            c0 -= ak * cen[static_cast<std::size_t>(k)];
        }
        out[static_cast<std::size_t>(n)] = c0;
    }
    if (info) *info = local;
    return out;
}

std::vector<double> predict_compact_sparse(const std::vector<std::array<double, 3>>& queries,
                                           const std::vector<double>& weights, KernelId kernel, double epsilon,
                                           const CompactNeighbourGrid& grid) {
    const double radius = compact_support_radius(kernel, epsilon);
    if (!(radius > 0.0)) throw std::invalid_argument("predict_compact_sparse: kernel is not compactly supported");
    if (weights.size() < 4U) throw std::invalid_argument("predict_compact_sparse: invalid weights");
    const std::size_t n = weights.size() - 4U;
    std::vector<double> out(queries.size(), 0.0);
    const long long nq = static_cast<long long>(queries.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 512)
#endif
    for (long long i = 0; i < nq; ++i) {
        const auto& q = queries[static_cast<std::size_t>(i)];
        double s = 0.0;
        grid.for_each_within(q, radius, [&](int j, double d) {
            s += weights[static_cast<std::size_t>(j)] * evaluate_kernel_id(kernel, d, epsilon);
        });
        s += weights[n] + weights[n + 1] * q[0] + weights[n + 2] * q[1] + weights[n + 3] * q[2];
        out[static_cast<std::size_t>(i)] = s;
    }
    return out;
}

} // namespace deeprbf
