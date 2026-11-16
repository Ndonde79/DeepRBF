#include "deeprbf/hmatrix.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace deeprbf {

HMatrix::HMatrix(const std::vector<Point3>& points, const std::string& kernel, double epsilon,
                 const HMatrixOptions& options)
    : points_(points), opt_(options) {
    const auto t0 = std::chrono::steady_clock::now();
    const int n = static_cast<int>(points_.size());
    if (n <= 0) throw std::invalid_argument("HMatrix: no points");
    if (!(opt_.tol > 0.0 && opt_.tol < 1.0)) throw std::invalid_argument("HMatrix: tol must be in (0, 1)");
    if (!(opt_.eta > 0.0)) throw std::invalid_argument("HMatrix: eta must be > 0");
    if (opt_.leaf_size < 4) throw std::invalid_argument("HMatrix: leaf_size must be >= 4");
    for (const auto& p : points_) for (double v : p)
        if (!std::isfinite(v)) throw std::invalid_argument("HMatrix: non-finite coordinate");
    kernel_ = FastKernel(kernel, epsilon, bbox_diagonal(points_));

    perm_.resize(static_cast<std::size_t>(n));
    std::iota(perm_.begin(), perm_.end(), 0);
    nodes_.reserve(static_cast<std::size_t>(4 * n / opt_.leaf_size + 8));
    build_tree(0, n);
    pts_.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) pts_[static_cast<std::size_t>(i)] = points_[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])];

    std::vector<std::pair<int,int>> far, near;
    build_blocks(0, 0, far, near);

    // Far field: ACA per block, in parallel.
    far_.resize(far.size());
    for (std::size_t b = 0; b < far.size(); ++b) { far_[b].t = far[b].first; far_[b].s = far[b].second; }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int b = 0; b < static_cast<int>(far_.size()); ++b) aca(far_[static_cast<std::size_t>(b)]);

    // Near field: stored if it fits in the budget.
    double near_entries = 0.0;
    for (const auto& p : near) {
        const Node& t = nodes_[static_cast<std::size_t>(p.first)];
        const Node& s = nodes_[static_cast<std::size_t>(p.second)];
        near_entries += static_cast<double>(t.end - t.begin) * static_cast<double>(s.end - s.begin);
    }
    stats_.near_stored = near_entries * 8.0 <= opt_.near_memory_mb * 1024.0 * 1024.0;
    near_.resize(near.size());
    for (std::size_t b = 0; b < near.size(); ++b) { near_[b].t = near[b].first; near_[b].s = near[b].second; }
    if (stats_.near_stored) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int b = 0; b < static_cast<int>(near_.size()); ++b) {
            Near& blk = near_[static_cast<std::size_t>(b)];
            const Node& t = nodes_[static_cast<std::size_t>(blk.t)];
            const Node& s = nodes_[static_cast<std::size_t>(blk.s)];
            const int m = t.end - t.begin, c = s.end - s.begin;
            blk.D.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(c));
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < c; ++j)
                    blk.D[static_cast<std::size_t>(i) * static_cast<std::size_t>(c) + static_cast<std::size_t>(j)] = k(t.begin + i, s.begin + j);
        }
    }

    // Statistics.
    stats_.n = n;
    stats_.n_lowrank = static_cast<int>(far_.size());
    stats_.n_near = static_cast<int>(near_.size());
    double lr_doubles = 0.0, rank_sum = 0.0;
    for (const auto& b : far_) {
        stats_.max_rank = std::max(stats_.max_rank, static_cast<int>(b.U.cols()));
        rank_sum += static_cast<double>(b.U.cols());
        lr_doubles += static_cast<double>(b.U.size() + b.V.size());
    }
    stats_.mean_rank = far_.empty() ? 0.0 : rank_sum / static_cast<double>(far_.size());
    stats_.lowrank_mb = lr_doubles * 8.0 / 1048576.0;
    stats_.near_mb = stats_.near_stored ? near_entries * 8.0 / 1048576.0 : 0.0;
    stats_.compression = (lr_doubles + near_entries) / (static_cast<double>(n) * static_cast<double>(n));
    stats_.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int HMatrix::build_tree(int begin, int end) {
    const int id = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    Node nd;
    nd.begin = begin; nd.end = end;
    for (int a = 0; a < 3; ++a) { nd.lo[a] = std::numeric_limits<double>::infinity(); nd.hi[a] = -nd.lo[a]; }
    for (int i = begin; i < end; ++i) {
        const Point3& p = points_[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])];
        for (int a = 0; a < 3; ++a) { nd.lo[a] = std::min(nd.lo[a], p[a]); nd.hi[a] = std::max(nd.hi[a], p[a]); }
    }
    if (end - begin > opt_.leaf_size) {
        int axis = 0;
        for (int a = 1; a < 3; ++a) if (nd.hi[a] - nd.lo[a] > nd.hi[axis] - nd.lo[axis]) axis = a;
        if (nd.hi[axis] > nd.lo[axis]) {   // identical points cannot be split; keep them in one (large) leaf
            const int mid = begin + (end - begin) / 2;
            std::nth_element(perm_.begin() + begin, perm_.begin() + mid, perm_.begin() + end, [&](int a, int b) {
                return points_[static_cast<std::size_t>(a)][axis] < points_[static_cast<std::size_t>(b)][axis];
            });
            nd.left = build_tree(begin, mid);
            nd.right = build_tree(mid, end);
        }
    }
    nodes_[static_cast<std::size_t>(id)] = nd;
    return id;
}

bool HMatrix::admissible(int ti, int si) const {
    const Node& t = nodes_[static_cast<std::size_t>(ti)];
    const Node& s = nodes_[static_cast<std::size_t>(si)];
    double dt = 0.0, ds = 0.0, dist = 0.0;
    for (int a = 0; a < 3; ++a) {
        dt += (t.hi[a] - t.lo[a]) * (t.hi[a] - t.lo[a]);
        ds += (s.hi[a] - s.lo[a]) * (s.hi[a] - s.lo[a]);
        const double gap = std::max({0.0, s.lo[a] - t.hi[a], t.lo[a] - s.hi[a]});
        dist += gap * gap;
    }
    return dist > 0.0 && std::sqrt(std::min(dt, ds)) <= opt_.eta * std::sqrt(dist);
}

void HMatrix::build_blocks(int t, int s, std::vector<std::pair<int,int>>& far,
                           std::vector<std::pair<int,int>>& near) const {
    if (admissible(t, s)) { far.emplace_back(t, s); return; }
    const Node& nt = nodes_[static_cast<std::size_t>(t)];
    const Node& ns = nodes_[static_cast<std::size_t>(s)];
    const bool tl = nt.left < 0, sl = ns.left < 0;
    if (tl && sl) { near.emplace_back(t, s); return; }
    if (tl) { build_blocks(t, ns.left, far, near); build_blocks(t, ns.right, far, near); return; }
    if (sl) { build_blocks(nt.left, s, far, near); build_blocks(nt.right, s, far, near); return; }
    build_blocks(nt.left, ns.left, far, near);
    build_blocks(nt.left, ns.right, far, near);
    build_blocks(nt.right, ns.left, far, near);
    build_blocks(nt.right, ns.right, far, near);
}

static constexpr int PROBES = 16;   // rows and columns checked before accepting convergence (half farthest-point, half random)

// Adaptive cross approximation with partial pivoting, then QR + SVD recompression.
// Plain partial-pivot ACA can stop while part of a block is still unapproximated, typically a separate piece of
// drillhole (or a single isolated sample) inside a cluster. Before accepting convergence the rows and columns
// farthest from every pivot so far (farthest-point order) are checked, then a few random ones; any with a residual
// above the expected error restarts the crosses there.
void HMatrix::aca(LowRank& blk) const {
    const Node& t = nodes_[static_cast<std::size_t>(blk.t)];
    const Node& s = nodes_[static_cast<std::size_t>(blk.s)];
    const int m = t.end - t.begin, c = s.end - s.begin;
    const int kmax = std::min(m, c);
    std::vector<Eigen::VectorXd> us, vs;
    std::vector<char> row_used(static_cast<std::size_t>(m), 0), col_used(static_cast<std::size_t>(c), 0);
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> row_gap(static_cast<std::size_t>(m), inf), col_gap(static_cast<std::size_t>(c), inf);
    auto mark_row = [&](int r) {
        row_used[static_cast<std::size_t>(r)] = 1;
        for (int a = 0; a < m; ++a)
            row_gap[static_cast<std::size_t>(a)] = std::min(row_gap[static_cast<std::size_t>(a)], fast_distance(pts_[t.begin + a], pts_[t.begin + r]));
    };
    auto mark_col = [&](int q) {
        col_used[static_cast<std::size_t>(q)] = 1;
        for (int a = 0; a < c; ++a)
            col_gap[static_cast<std::size_t>(a)] = std::min(col_gap[static_cast<std::size_t>(a)], fast_distance(pts_[s.begin + a], pts_[s.begin + q]));
    };
    auto farthest = [](const std::vector<char>& used, const std::vector<double>& gap) {
        int best = -1;
        for (std::size_t a = 0; a < used.size(); ++a)
            if (!used[a] && (best < 0 || gap[a] > gap[static_cast<std::size_t>(best)])) best = static_cast<int>(a);
        return best;
    };
    std::uint64_t rng = 0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(blk.t) * 1000003ULL + static_cast<std::uint64_t>(blk.s));
    auto rand_below = [&rng](int bound) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return static_cast<int>(rng % static_cast<std::uint64_t>(bound));
    };
    auto residual_row = [&](int r, Eigen::VectorXd& out) {
        for (int j = 0; j < c; ++j) out(j) = k(t.begin + r, s.begin + j);
        for (std::size_t l = 0; l < us.size(); ++l) out -= us[l](r) * vs[l];
    };
    auto residual_col = [&](int q, Eigen::VectorXd& out) {
        for (int r = 0; r < m; ++r) out(r) = k(t.begin + r, s.begin + q);
        for (std::size_t l = 0; l < us.size(); ++l) out -= vs[l](q) * us[l];
    };
    double norm2 = 0.0;
    int i = 0, zero_rows = 0;
    Eigen::VectorXd row(c), col(m), probe_r(c), probe_c(m);
    while (static_cast<int>(us.size()) < kmax) {
        residual_row(i, row);
        mark_row(i);
        Eigen::Index jp = 0;
        const double piv = row.cwiseAbs().maxCoeff(&jp);
        const double rms = std::sqrt(std::max(norm2, 0.0) / (static_cast<double>(m) * c));   // typical approximated entry
        const bool accept = piv > 1.0e-300 && !(piv < 1.0e-3 * opt_.tol * rms);
        if (accept) {
            const Eigen::VectorXd v = row / row(jp);
            residual_col(static_cast<int>(jp), col);
            mark_col(static_cast<int>(jp));
            const double nu = col.norm(), nv = v.norm();
            double cross = 0.0;
            for (std::size_t l = 0; l < us.size(); ++l) cross += us[l].dot(col) * vs[l].dot(v);
            norm2 += 2.0 * cross + nu * nu * nv * nv;
            us.push_back(col);
            vs.push_back(v);
            if (!(nu * nv <= opt_.tol * std::sqrt(std::max(norm2, 0.0)))) {
                int next = -1;
                double best = -1.0;
                for (int r = 0; r < m; ++r)
                    if (!row_used[static_cast<std::size_t>(r)] && std::fabs(col(r)) > best) { best = std::fabs(col(r)); next = r; }
                if (next < 0) break;
                i = next;
                continue;
            }
        } else if (++zero_rows > 64) {
            break;
        }
        // Converged (or a zero row): look for a part of the block the crosses have missed.
        const double row_lim = 10.0 * opt_.tol * std::sqrt(std::max(norm2, 0.0) / m);
        const double col_lim = 10.0 * opt_.tol * std::sqrt(std::max(norm2, 0.0) / c);
        int next = -1;
        for (int trial = 0; trial < PROBES && next < 0; ++trial) {
            const int r = (trial < PROBES / 2) ? farthest(row_used, row_gap) : rand_below(m);
            if (r < 0 || row_used[static_cast<std::size_t>(r)]) continue;
            residual_row(r, probe_r);
            if (probe_r.norm() > row_lim && probe_r.cwiseAbs().maxCoeff() > 1.0e-300) next = r;
            else mark_row(r);   // checked and fine
        }
        for (int trial = 0; trial < PROBES && next < 0; ++trial) {
            const int q = (trial < PROBES / 2) ? farthest(col_used, col_gap) : rand_below(c);
            if (q < 0 || col_used[static_cast<std::size_t>(q)]) continue;
            mark_col(q);
            residual_col(q, probe_c);
            if (probe_c.norm() > col_lim && probe_c.cwiseAbs().maxCoeff() > 1.0e-300) {
                double best = -1.0;
                for (int r = 0; r < m; ++r)
                    if (!row_used[static_cast<std::size_t>(r)] && std::fabs(probe_c(r)) > best) { best = std::fabs(probe_c(r)); next = r; }
            }
        }
        if (next < 0) break;
        i = next;
    }
    const int kr = static_cast<int>(us.size());
    if (kr == 0) { blk.U.resize(m, 0); blk.V.resize(c, 0); return; }
    Eigen::MatrixXd U(m, kr), V(c, kr);
    for (int l = 0; l < kr; ++l) { U.col(l) = us[static_cast<std::size_t>(l)]; V.col(l) = vs[static_cast<std::size_t>(l)]; }
    if (kr == 1) { blk.U = std::move(U); blk.V = std::move(V); return; }

    // Recompression: U V^T = Qu (Ru Rv^T) Qv^T, truncated SVD of the small core.
    Eigen::HouseholderQR<Eigen::MatrixXd> qu(U), qv(V);
    const Eigen::MatrixXd Qu = qu.householderQ() * Eigen::MatrixXd::Identity(m, kr);
    const Eigen::MatrixXd Qv = qv.householderQ() * Eigen::MatrixXd::Identity(c, kr);
    const Eigen::MatrixXd Ru = qu.matrixQR().topRows(kr).triangularView<Eigen::Upper>();
    const Eigen::MatrixXd Rv = qv.matrixQR().topRows(kr).triangularView<Eigen::Upper>();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Ru * Rv.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& sv = svd.singularValues();
    const double total = sv.squaredNorm();
    int r = kr;
    double tail = 0.0;
    while (r > 1 && tail + sv(r - 1) * sv(r - 1) <= opt_.tol * opt_.tol * total) { tail += sv(r - 1) * sv(r - 1); --r; }
    blk.U = Qu * (svd.matrixU().leftCols(r) * sv.head(r).asDiagonal());
    blk.V = Qv * svd.matrixV().leftCols(r);
}

std::vector<double> HMatrix::locate(int i, int j) const {
    const int n = size();
    std::vector<int> inv(static_cast<std::size_t>(n));
    for (int a = 0; a < n; ++a) inv[static_cast<std::size_t>(perm_[static_cast<std::size_t>(a)])] = a;
    const int pi = inv[static_cast<std::size_t>(i)], pj = inv[static_cast<std::size_t>(j)];
    std::vector<double> out; int count = 0;
    auto in = [&](const Node& nd, int p) { return p >= nd.begin && p < nd.end; };
    for (const auto& b : far_) {
        const Node& t = nodes_[static_cast<std::size_t>(b.t)]; const Node& s = nodes_[static_cast<std::size_t>(b.s)];
        if (in(t, pi) && in(s, pj)) { ++count; out = {1.0, double(t.end - t.begin), double(s.end - s.begin), double(b.U.cols())};
            for (int a = t.begin; a < t.end; ++a) out.push_back(perm_[static_cast<std::size_t>(a)]);
            for (int a = s.begin; a < s.end; ++a) out.push_back(perm_[static_cast<std::size_t>(a)]); }
    }
    for (const auto& b : near_) {
        const Node& t = nodes_[static_cast<std::size_t>(b.t)]; const Node& s = nodes_[static_cast<std::size_t>(b.s)];
        if (in(t, pi) && in(s, pj)) { ++count; out = {0.0, double(t.end - t.begin), double(s.end - s.begin), 0.0}; }
    }
    out.push_back(count);
    return out;
}

void HMatrix::apply(const std::vector<double>& x, std::vector<double>& y) const {
    const int n = size();
    if (static_cast<int>(x.size()) != n) throw std::invalid_argument("HMatrix::apply: size mismatch");
    Eigen::VectorXd xp(n), yp = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) xp(i) = x[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])];
    const int nfar = static_cast<int>(far_.size()), nnear = static_cast<int>(near_.size());

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        Eigen::VectorXd yl = Eigen::VectorXd::Zero(n);
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 8) nowait
#endif
        for (int b = 0; b < nfar; ++b) {
            const LowRank& blk = far_[static_cast<std::size_t>(b)];
            if (blk.U.cols() == 0) continue;
            const Node& t = nodes_[static_cast<std::size_t>(blk.t)];
            const Node& s = nodes_[static_cast<std::size_t>(blk.s)];
            const Eigen::VectorXd tmp = blk.V.transpose() * xp.segment(s.begin, s.end - s.begin);
            yl.segment(t.begin, t.end - t.begin).noalias() += blk.U * tmp;
        }
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 8) nowait
#endif
        for (int b = 0; b < nnear; ++b) {
            const Near& blk = near_[static_cast<std::size_t>(b)];
            const Node& t = nodes_[static_cast<std::size_t>(blk.t)];
            const Node& s = nodes_[static_cast<std::size_t>(blk.s)];
            const int m = t.end - t.begin, c = s.end - s.begin;
            if (stats_.near_stored) {
                using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                Eigen::Map<const RowMat> D(blk.D.data(), m, c);
                yl.segment(t.begin, m).noalias() += D * xp.segment(s.begin, c);
            } else {
                for (int i = 0; i < m; ++i) {
                    double acc = 0.0;
                    for (int j = 0; j < c; ++j) acc += k(t.begin + i, s.begin + j) * xp(s.begin + j);
                    yl(t.begin + i) += acc;
                }
            }
        }
#if defined(_OPENMP)
#pragma omp critical(deeprbf_hmatrix_reduce)
#endif
        yp += yl;
    }
    y.assign(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) y[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])] = yp(i);
}

}  // namespace deeprbf
