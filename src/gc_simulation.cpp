// DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
// 3.3.4: GSLIB-convention kriging, SGS and SIS (see gc_simulation.hpp).
#include "deeprbf/gc_simulation.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace deeprbf {
namespace {

constexpr double DEG = 3.14159265358979323846 / 180.0;
constexpr int MAX_LOCAL = 128;

struct Rot {
    std::array<double, 9> m{};
    double h2(const Point3& a, const Point3& b) const {
        const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        const double u = m[0] * dx + m[1] * dy + m[2] * dz;
        const double v = m[3] * dx + m[4] * dy + m[5] * dz;
        const double w = m[6] * dx + m[7] * dy + m[8] * dz;
        return u * u + v * v + w * w;
    }
    Point3 apply(const Point3& p) const {
        return {m[0] * p[0] + m[1] * p[1] + m[2] * p[2], m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
                m[6] * p[0] + m[7] * p[1] + m[8] * p[2]};
    }
};

struct Model {
    double nugget = 0.0, sill = 0.0;
    struct S { int type; double cc; Rot rot; };
    std::vector<S> st;
    explicit Model(const GSLIBVariogram& v) : nugget(v.nugget), sill(v.sill()) {
        if (!(v.nugget >= 0.0) || !std::isfinite(v.nugget)) throw std::invalid_argument("nugget must be finite and >= 0");
        if (v.structures.empty() && v.nugget <= 0.0) throw std::invalid_argument("variogram needs a nugget or a structure");
        for (const auto& s : v.structures) {
            int t;
            if (s.type == "spherical") t = 0;
            else if (s.type == "exponential") t = 1;
            else if (s.type == "gaussian") t = 2;
            else throw std::invalid_argument("variogram structure type must be spherical, exponential or gaussian");
            if (!(s.cc >= 0.0) || !(s.range_major > 0) || !(s.range_minor > 0) || !(s.range_vertical > 0))
                throw std::invalid_argument("structure contribution must be >= 0 and ranges > 0");
            st.push_back({t, s.cc, Rot{gslib_rotation(s.azimuth, s.dip, s.rake, s.range_major, s.range_minor, s.range_vertical)}});
        }
    }
    // covariance between two points (nugget included only at zero lag)
    double cov(const Point3& a, const Point3& b) const {
        double c = 0.0;
        bool zero = (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]);
        if (zero) return sill;
        for (const auto& s : st) {
            const double h = std::sqrt(s.rot.h2(a, b));
            double r;
            if (s.type == 0) r = h < 1.0 ? 1.0 - h * (1.5 - 0.5 * h * h) : 0.0;
            else if (s.type == 1) r = std::exp(-3.0 * h);
            else r = std::exp(-3.0 * h * h);
            c += s.cc * r;
        }
        return c;
    }
};

// Hash in the search frame, where the search ellipsoid is the unit sphere.
struct Hash {
    Rot rot;
    double cell = 0.5;
    std::unordered_map<std::int64_t, std::vector<int>> cells;
    std::vector<Point3> pts;   // search-frame coordinates
    explicit Hash(const SearchEllipsoid& s) {
        if (!(s.radius_major > 0) || !(s.radius_minor > 0) || !(s.radius_vertical > 0))
            throw std::invalid_argument("search radii must be > 0");
        rot.m = gslib_rotation(s.azimuth, s.dip, s.rake, s.radius_major, s.radius_minor, s.radius_vertical);
    }
    static std::int64_t key(std::int64_t i, std::int64_t j, std::int64_t k) {
        return (i & 0x1FFFFF) | ((j & 0x1FFFFF) << 21) | ((k & 0x1FFFFF) << 42);
    }
    int add(const Point3& p) {
        const Point3 q = rot.apply(p);
        const int id = static_cast<int>(pts.size());
        pts.push_back(q);
        cells[key(std::llround(std::floor(q[0] / cell)), std::llround(std::floor(q[1] / cell)), std::llround(std::floor(q[2] / cell)))].push_back(id);
        return id;
    }
    // ids of the nearest `k` points within the unit sphere (search-frame distance), nearest first
    void query(const Point3& p, int k, std::vector<std::pair<double, int>>& out) const {
        out.clear();
        if (k <= 0) return;
        const Point3 q = rot.apply(p);
        const std::int64_t ci = std::llround(std::floor(q[0] / cell)), cj = std::llround(std::floor(q[1] / cell)),
                           ck = std::llround(std::floor(q[2] / cell));
        const int r = static_cast<int>(std::ceil(1.0 / cell));
        for (int i = -r; i <= r; ++i)
            for (int j = -r; j <= r; ++j)
                for (int l = -r; l <= r; ++l) {
                    auto it = cells.find(key(ci + i, cj + j, ck + l));
                    if (it == cells.end()) continue;
                    for (int id : it->second) {
                        const auto& o = pts[static_cast<size_t>(id)];
                        const double d2 = (o[0] - q[0]) * (o[0] - q[0]) + (o[1] - q[1]) * (o[1] - q[1]) + (o[2] - q[2]) * (o[2] - q[2]);
                        if (d2 <= 1.0) out.emplace_back(d2, id);
                    }
                }
        if (static_cast<int>(out.size()) > k) {
            std::nth_element(out.begin(), out.begin() + k, out.end());
            out.resize(static_cast<size_t>(k));
        }
        std::sort(out.begin(), out.end());
    }
};

// Kriging weights at target from neighbour points. Returns false if the system is singular.
bool krige(const Model& m, const Point3& target, const std::vector<Point3>& P, bool ordinary, Eigen::VectorXd& w,
           double& mu, double& var) {
    const int n = static_cast<int>(P.size());
    const int dim = n + (ordinary ? 1 : 0);
    Eigen::MatrixXd A(dim, dim);
    Eigen::VectorXd b(dim);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) A(i, j) = A(j, i) = m.cov(P[static_cast<size_t>(i)], P[static_cast<size_t>(j)]);
        b(i) = m.cov(P[static_cast<size_t>(i)], target);
    }
    if (ordinary) {
        for (int i = 0; i < n; ++i) A(i, n) = A(n, i) = 1.0;
        A(n, n) = 0.0;
        b(n) = 1.0;
    }
    Eigen::VectorXd x;
    if (ordinary) {
        Eigen::PartialPivLU<Eigen::MatrixXd> f(A);
        x = f.solve(b);
    } else {
        Eigen::LLT<Eigen::MatrixXd> f(A);
        if (f.info() != Eigen::Success) {
            A.diagonal().array() += 1e-10 * std::max(1.0, m.sill);
            f.compute(A);
            if (f.info() != Eigen::Success) return false;
        }
        x = f.solve(b);
    }
    if (!x.allFinite() || (A * x - b).norm() > 1e-8 * std::max(1.0, b.norm())) return false;
    w = x.head(n);
    mu = ordinary ? x(n) : 0.0;
    var = m.sill - w.dot(b.head(n)) - mu;
    if (var < 0.0) var = 0.0;
    return true;
}

// Multiple-grid order: coarse lattice nodes first (GSLIB nmult), each level visited in random order.
std::vector<int> multigrid_path(const std::vector<Point3>& t, int levels, std::mt19937_64& rng, std::vector<int>* counts) {
    const size_t n = t.size();
    std::vector<int> path(n);
    std::iota(path.begin(), path.end(), 0);
    if (levels <= 0 || n < 64) {
        std::shuffle(path.begin(), path.end(), rng);
        if (counts) *counts = {static_cast<int>(n)};
        return path;
    }
    // base spacing per axis = smallest positive step between sorted distinct coordinates (regular-grid targets)
    std::array<double, 3> lo{}, step{};
    for (int a = 0; a < 3; ++a) {
        std::vector<double> c(n);
        for (size_t i = 0; i < n; ++i) c[i] = t[i][static_cast<size_t>(a)];
        std::sort(c.begin(), c.end());
        lo[static_cast<size_t>(a)] = c.front();
        double s = std::numeric_limits<double>::infinity();
        for (size_t i = 1; i < n; ++i) { const double d = c[i] - c[i - 1]; if (d > 1e-9 * std::max(1.0, std::abs(c[i])) && d < s) s = d; }
        step[static_cast<size_t>(a)] = std::isfinite(s) ? s : 0.0;
    }
    std::vector<int> level(n, 0);
    for (size_t i = 0; i < n; ++i) {
        int L = levels;
        for (int a = 0; a < 3; ++a) {
            if (step[static_cast<size_t>(a)] <= 0) continue;
            const long long ix = std::llround((t[i][static_cast<size_t>(a)] - lo[static_cast<size_t>(a)]) / step[static_cast<size_t>(a)]);
            int l = 0;
            while (l < levels && (ix % (1LL << (l + 1))) == 0) ++l;
            L = std::min(L, l);
        }
        level[i] = L;
    }
    std::vector<std::vector<int>> by(static_cast<size_t>(levels) + 1);
    for (size_t i = 0; i < n; ++i) by[static_cast<size_t>(level[i])].push_back(static_cast<int>(i));
    path.clear();
    if (counts) counts->clear();
    for (int L = levels; L >= 0; --L) {
        auto& v = by[static_cast<size_t>(L)];
        std::shuffle(v.begin(), v.end(), rng);
        path.insert(path.end(), v.begin(), v.end());
        if (counts) counts->push_back(static_cast<int>(v.size()));
    }
    return path;
}

double norm_inv(double p) {   // Acklam's rational approximation, refined by one Halley step
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
    p = std::clamp(p, 1e-12, 1.0 - 1e-12);
    double x;
    if (p < 0.02425) { const double q = std::sqrt(-2 * std::log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    else if (p > 1 - 0.02425) { const double q = std::sqrt(-2 * std::log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    else { const double q = p - 0.5, r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    const double u = e * std::sqrt(2 * 3.14159265358979323846) * std::exp(x * x / 2);
    return x - u / (1 + x * u / 2);
}
double norm_cdf(double y) { return 0.5 * std::erfc(-y / std::sqrt(2.0)); }

void check_points(const std::vector<Point3>& p, const char* what) {
    for (const auto& q : p) for (double v : q) if (!std::isfinite(v)) throw std::invalid_argument(std::string("non-finite coordinate in ") + what);
}

std::uint64_t realisation_seed(std::uint64_t seed, int r) {   // splitmix64 of (seed, r): independent streams
    std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(r + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

double GSLIBVariogram::sill() const {
    double s = nugget;
    for (const auto& st : structures) s += st.cc;
    return s;
}

std::array<double, 9> gslib_rotation(double azimuth, double dip, double rake, double r_major, double r_minor, double r_vertical) {
    // GSLIB setrot with ang1 = azimuth, ang2 = -dip (GSLIB dips are negative downward), ang3 = rake,
    // anis1 = minor / major, anis2 = vertical / major; result additionally divided by the major range.
    const double alpha = ((azimuth >= 0.0 && azimuth < 270.0) ? (90.0 - azimuth) : (450.0 - azimuth)) * DEG;
    const double beta = dip * DEG;
    const double theta = rake * DEG;
    const double sa = std::sin(alpha), ca = std::cos(alpha), sb = std::sin(beta), cb = std::cos(beta), st = std::sin(theta), ct = std::cos(theta);
    const double f1 = r_major / r_minor, f2 = r_major / r_vertical, g = 1.0 / r_major;
    return {g * cb * ca, g * cb * sa, -g * sb,
            g * f1 * (-ct * sa + st * sb * ca), g * f1 * (ct * ca + st * sb * sa), g * f1 * (st * cb),
            g * f2 * (st * sa + ct * sb * ca), g * f2 * (-st * ca + ct * sb * sa), g * f2 * (ct * cb)};
}

double gslib_covariance(const GSLIBVariogram& v, const Point3& a, const Point3& b) { return Model(v).cov(a, b); }

KrigingGSLIBResult kriging_gslib(const std::vector<Point3>& X, const std::vector<double>& v, const std::vector<Point3>& T,
                                 const GSLIBVariogram& vario, const SearchEllipsoid& search, int max_data, int min_data,
                                 const std::string& kriging_type, double mean) {
    check_points(X, "samples"); check_points(T, "targets");
    if (X.size() != v.size()) throw std::invalid_argument("sample/value size mismatch");
    for (double x : v) if (!std::isfinite(x)) throw std::invalid_argument("non-finite sample value");
    if (kriging_type != "ordinary" && kriging_type != "simple") throw std::invalid_argument("kriging_type must be ordinary or simple");
    if (max_data < 1 || min_data < 1 || min_data > max_data) throw std::invalid_argument("invalid max_data/min_data");
    max_data = std::min(max_data, MAX_LOCAL);
    const bool ok = kriging_type == "ordinary";
    const Model m(vario);
    Hash h(search);
    for (const auto& p : X) h.add(p);
    KrigingGSLIBResult out;
    out.estimate.assign(T.size(), std::numeric_limits<double>::quiet_NaN());
    out.variance.assign(T.size(), std::numeric_limits<double>::quiet_NaN());
    out.n_samples.assign(T.size(), 0);
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<std::pair<double, int>> nb;
        std::vector<Point3> P;
        Eigen::VectorXd w;
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 64)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(T.size()); ++i) {
            h.query(T[static_cast<size_t>(i)], max_data, nb);
            if (static_cast<int>(nb.size()) < min_data) continue;
            P.clear();
            for (const auto& e : nb) P.push_back(X[static_cast<size_t>(e.second)]);
            double mu, var;
            if (!krige(m, T[static_cast<size_t>(i)], P, ok, w, mu, var)) continue;
            double est = ok ? 0.0 : mean * (1.0 - w.sum());
            for (size_t k = 0; k < nb.size(); ++k) est += w(static_cast<Eigen::Index>(k)) * v[static_cast<size_t>(nb[k].second)];
            out.estimate[static_cast<size_t>(i)] = est;
            out.variance[static_cast<size_t>(i)] = var;
            out.n_samples[static_cast<size_t>(i)] = static_cast<int>(nb.size());
        }
    }
    return out;
}

NormalScoreTable normal_score_table(const std::vector<double>& values, double zmin, double zmax) {
    if (values.size() < 2) throw std::invalid_argument("normal score needs at least two values");
    NormalScoreTable t;
    t.z = values;
    std::sort(t.z.begin(), t.z.end());
    const size_t n = t.z.size();
    t.y.resize(n);
    // tied values share the normal score of their mean rank, so equal grades map to one score
    for (size_t i = 0; i < n;) {
        size_t j = i;
        while (j + 1 < n && t.z[j + 1] == t.z[i]) ++j;
        const double p = ((static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 0.5) / static_cast<double>(n);
        for (size_t k = i; k <= j; ++k) t.y[k] = norm_inv(p);
        i = j + 1;
    }
    t.zmin = std::isfinite(zmin) ? std::min(zmin, t.z.front()) : t.z.front();
    t.zmax = std::isfinite(zmax) ? std::max(zmax, t.z.back()) : t.z.back();
    return t;
}

double ns_forward(const NormalScoreTable& t, double z) {
    auto it = std::lower_bound(t.z.begin(), t.z.end(), z);
    if (it == t.z.end()) return t.y.back();
    const size_t i = static_cast<size_t>(it - t.z.begin());
    if (*it == z || i == 0) return t.y[i];
    const double f = (z - t.z[i - 1]) / (t.z[i] - t.z[i - 1]);
    return t.y[i - 1] + f * (t.y[i] - t.y[i - 1]);
}

double ns_back(const NormalScoreTable& t, double y) {
    // linear interpolation in the (cdf, z) table, linear tails to zmin at cdf 0 and zmax at cdf 1
    const double p = norm_cdf(y);
    const size_t n = t.z.size();
    const double p0 = norm_cdf(t.y.front()), p1 = norm_cdf(t.y.back());
    if (p <= p0) return t.zmin + (t.z.front() - t.zmin) * (p / p0);
    if (p >= p1) return t.z.back() + (t.zmax - t.z.back()) * ((p - p1) / (1.0 - p1));
    auto it = std::upper_bound(t.y.begin(), t.y.end(), y);
    const size_t i = static_cast<size_t>(it - t.y.begin());
    const size_t a = std::min(i - 1, n - 2), b2 = a + 1;
    const double pa = norm_cdf(t.y[a]), pb = norm_cdf(t.y[b2]);
    if (pb <= pa) return t.z[a];
    return t.z[a] + (t.z[b2] - t.z[a]) * (p - pa) / (pb - pa);
}

SimulationGSLIBResult sgs_gslib(const std::vector<Point3>& X, const std::vector<double>& v, const std::vector<Point3>& T,
                                const GSLIBVariogram& vario, const SearchEllipsoid& search, int n_real, int max_data,
                                int max_nodes, int min_data, const std::string& kriging_type, bool normal_score,
                                double zmin, double zmax, int levels, std::uint64_t seed) {
    check_points(X, "samples"); check_points(T, "targets");
    if (X.size() != v.size()) throw std::invalid_argument("sample/value size mismatch");
    for (double x : v) if (!std::isfinite(x)) throw std::invalid_argument("non-finite sample value");
    if (kriging_type != "ordinary" && kriging_type != "simple") throw std::invalid_argument("kriging_type must be ordinary or simple");
    if (n_real < 1 || max_data < 0 || max_nodes < 0 || max_data + max_nodes < 1 || min_data < 0)
        throw std::invalid_argument("invalid realisation count or search limits");
    if (static_cast<double>(T.size()) * static_cast<double>(n_real) > 2.5e9) throw std::invalid_argument("output too large");
    max_data = std::min(max_data, MAX_LOCAL); max_nodes = std::min(max_nodes, MAX_LOCAL);
    const bool ordk = kriging_type == "ordinary";
    const Model m(vario);
    const double sd = std::sqrt(m.sill);
    NormalScoreTable tab;
    std::vector<double> y(v.size());
    if (normal_score) { tab = normal_score_table(v, zmin, zmax); for (size_t i = 0; i < v.size(); ++i) y[i] = ns_forward(tab, v[i]); }
    else y = v;
    // targets that coincide with a datum take the datum (GSLIB assigns data to nodes)
    Hash dh(search);
    for (const auto& p : X) dh.add(p);

    SimulationGSLIBResult out;
    out.n_targets = T.size();
    out.n_realizations = static_cast<size_t>(n_real);
    out.values.assign(T.size() * static_cast<size_t>(n_real), std::numeric_limits<double>::quiet_NaN());
    std::vector<int> level_counts;
    {
        std::mt19937_64 r0(realisation_seed(seed, -1));
        multigrid_path(T, levels, r0, &level_counts);
    }
    out.multigrid_level_counts = level_counts;
    std::vector<int> hard(T.size(), -1);
    {
        std::vector<std::pair<double, int>> nb;
        for (size_t i = 0; i < T.size(); ++i) {
            dh.query(T[i], 1, nb);
            if (!nb.empty()) { const auto& p = X[static_cast<size_t>(nb[0].second)]; if (p == T[i]) hard[i] = nb[0].second; }
        }
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < n_real; ++r) {
        std::mt19937_64 rng(realisation_seed(seed, r));
        std::normal_distribution<double> N01(0.0, 1.0);
        const auto path = multigrid_path(T, levels, rng, nullptr);
        Hash nh(search);
        std::vector<double> nodev;   // simulated normal scores in insertion order
        std::vector<Point3> nodep;
        std::vector<std::pair<double, int>> nbd, nbn;
        std::vector<Point3> P;
        std::vector<double> V;
        Eigen::VectorXd w;
        double* outr = out.values.data() + static_cast<size_t>(r) * T.size();
        for (int idx : path) {
            const Point3& t = T[static_cast<size_t>(idx)];
            double ys;
            if (hard[static_cast<size_t>(idx)] >= 0) {
                ys = y[static_cast<size_t>(hard[static_cast<size_t>(idx)])];
                outr[idx] = v[static_cast<size_t>(hard[static_cast<size_t>(idx)])];
                (void)ys;   // already in the data search; not added again as a simulated node
                continue;
            }
            dh.query(t, max_data, nbd);
            nh.query(t, max_nodes, nbn);
            P.clear(); V.clear();
            for (const auto& e : nbd) { P.push_back(X[static_cast<size_t>(e.second)]); V.push_back(y[static_cast<size_t>(e.second)]); }
            for (const auto& e : nbn) { P.push_back(nodep[static_cast<size_t>(e.second)]); V.push_back(nodev[static_cast<size_t>(e.second)]); }
            double mean = 0.0, var = m.sill, mu;
            if (static_cast<int>(nbd.size()) >= min_data && !P.empty() && krige(m, t, P, ordk, w, mu, var)) {
                mean = 0.0;
                for (size_t k = 0; k < P.size(); ++k) mean += w(static_cast<Eigen::Index>(k)) * V[k];
            } else {
                var = m.sill;
            }
            ys = mean + std::sqrt(std::max(var, 0.0)) * N01(rng);
            outr[idx] = normal_score ? ns_back(tab, ys / sd) : ys;
            nh.add(t); nodep.push_back(t); nodev.push_back(ys);
        }
    }
    return out;
}

SimulationGSLIBResult sis_gslib(const std::vector<Point3>& X, const std::vector<int>& c, const std::vector<Point3>& T,
                                const std::vector<int>& cats, const std::vector<GSLIBVariogram>& varios,
                                const SearchEllipsoid& search, int n_real, int max_data, int max_nodes, int levels,
                                std::uint64_t seed) {
    check_points(X, "samples"); check_points(T, "targets");
    if (X.size() != c.size()) throw std::invalid_argument("sample/category size mismatch");
    if (cats.size() < 2) throw std::invalid_argument("SIS needs at least two categories");
    if (varios.size() != cats.size()) throw std::invalid_argument("one variogram per category is required");
    if (n_real < 1 || max_data < 0 || max_nodes < 0 || max_data + max_nodes < 1) throw std::invalid_argument("invalid search limits");
    if (static_cast<double>(T.size()) * static_cast<double>(n_real) > 2.5e9) throw std::invalid_argument("output too large");
    max_data = std::min(max_data, MAX_LOCAL); max_nodes = std::min(max_nodes, MAX_LOCAL);
    const size_t K = cats.size();
    std::vector<double> prop(K, 0.0);
    std::vector<int> ci(X.size(), -1);
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t k = 0; k < K; ++k) if (c[i] == cats[k]) ci[i] = static_cast<int>(k);
        if (ci[i] < 0) throw std::invalid_argument("sample category not in the category list");
        prop[static_cast<size_t>(ci[i])] += 1.0;
    }
    for (auto& p : prop) p /= static_cast<double>(X.size());
    std::vector<Model> models;
    for (size_t k = 0; k < K; ++k) {
        GSLIBVariogram s = varios[k];
        const double f = prop[k] * (1.0 - prop[k]) / std::max(s.sill(), 1e-300);
        s.nugget *= f;
        for (auto& st : s.structures) st.cc *= f;
        models.emplace_back(s);
    }
    Hash dh(search);
    for (const auto& p : X) dh.add(p);
    std::vector<int> hard(T.size(), -1);
    {
        std::vector<std::pair<double, int>> nb;
        for (size_t i = 0; i < T.size(); ++i) {
            dh.query(T[i], 1, nb);
            if (!nb.empty() && X[static_cast<size_t>(nb[0].second)] == T[i]) hard[i] = ci[static_cast<size_t>(nb[0].second)];
        }
    }
    SimulationGSLIBResult out;
    out.n_targets = T.size();
    out.n_realizations = static_cast<size_t>(n_real);
    out.categories.assign(T.size() * static_cast<size_t>(n_real), 0);
    {
        std::mt19937_64 r0(realisation_seed(seed, -1));
        multigrid_path(T, levels, r0, &out.multigrid_level_counts);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int r = 0; r < n_real; ++r) {
        std::mt19937_64 rng(realisation_seed(seed, r));
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const auto path = multigrid_path(T, levels, rng, nullptr);
        Hash nh(search);
        std::vector<Point3> nodep;
        std::vector<int> nodec;
        std::vector<std::pair<double, int>> nbd, nbn;
        std::vector<Point3> P;
        std::vector<int> C;
        std::vector<double> pr(K);
        Eigen::VectorXd w;
        int* outr = out.categories.data() + static_cast<size_t>(r) * T.size();
        for (int idx : path) {
            const Point3& t = T[static_cast<size_t>(idx)];
            int drawn;
            if (hard[static_cast<size_t>(idx)] >= 0) {
                drawn = hard[static_cast<size_t>(idx)];
            } else {
                dh.query(t, max_data, nbd);
                nh.query(t, max_nodes, nbn);
                P.clear(); C.clear();
                for (const auto& e : nbd) { P.push_back(X[static_cast<size_t>(e.second)]); C.push_back(ci[static_cast<size_t>(e.second)]); }
                for (const auto& e : nbn) { P.push_back(nodep[static_cast<size_t>(e.second)]); C.push_back(nodec[static_cast<size_t>(e.second)]); }
                double tot = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    double p = prop[k], mu, var;
                    if (!P.empty() && krige(models[k], t, P, false, w, mu, var)) {
                        for (size_t j = 0; j < P.size(); ++j) p += w(static_cast<Eigen::Index>(j)) * ((C[j] == static_cast<int>(k) ? 1.0 : 0.0) - prop[k]);
                    }
                    pr[k] = std::clamp(p, 0.0, 1.0);   // order-relation correction: clip, then rescale
                    tot += pr[k];
                }
                if (tot <= 1e-12) pr = prop; else for (auto& p : pr) p /= tot;
                const double u = U(rng);
                double acc = 0.0;
                drawn = static_cast<int>(K) - 1;
                for (size_t k = 0; k < K; ++k) { acc += pr[k]; if (u < acc) { drawn = static_cast<int>(k); break; } }
            }
            outr[idx] = cats[static_cast<size_t>(drawn)];
            if (hard[static_cast<size_t>(idx)] < 0) { nh.add(t); nodep.push_back(t); nodec.push_back(drawn); }
        }
    }
    return out;
}

}  // namespace deeprbf
