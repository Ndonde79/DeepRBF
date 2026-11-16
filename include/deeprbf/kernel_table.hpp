#pragma once
// 3.5.0: fast evaluation of radial kernels for the O(n^2) loops (matrix-free products, block assembly, prediction).
//
// The kernel is tabulated once as a function of x = d / s (s = epsilon, or the gen_cauchy scale b) on geometric
// bands [2^e, 2^(e+1)), e = emin..emax-1, each split into M equal cells carrying a degree-D polynomial interpolant at
// Chebyshev nodes. Geometric bands keep the interpolation error uniform near d = 0, where gen_cauchy (r^p, p < 2),
// the exponential (|r|) and the thin plate (r^2 log r) are not smooth. With M = 32, D = 6 the interpolant agrees with
// the closed form to a few units in the last place; the table is checked against the closed form when it is built
// and, if any check fails, the closed form is used throughout (FastKernel::tabulated() reports which).
// Distances outside the tabulated bands (d = 0, very small or very large d) always use the closed form.

#include "deeprbf/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace deeprbf {

class FastKernel {
public:
    static constexpr int CELLS = 32;     // cells per band
    static constexpr int DEGREE = 6;     // polynomial degree per cell
    static constexpr int EMIN = -40;     // smallest band: d / s >= 2^-40
    static constexpr double CHECK_TOL = 1.0e-13;   // accepted |table - exact| / max(1, |exact|)

    FastKernel() = default;

    // d_max: largest distance that will be evaluated (e.g. the bounding-box diagonal). Larger distances still work,
    // through the closed form.
    FastKernel(const std::string& kernel, double epsilon, double d_max) : exact_(resolve_kernel(kernel, epsilon)) {
        build(d_max);
    }

    double operator()(double d) const {
        const double x = d * inv_s_;
        if (!(x >= xlo_) || x >= xhi_) return exact_(d);   // range test on x itself, so the band index is in range
        int e;
        const double mant = std::frexp(x, &e);   // x = d / s = mant * 2^e, mant in [0.5, 1)
        const double u = (mant * 2.0 - 1.0) * CELLS;      // position within band e-1, in cells
        int m = static_cast<int>(u);
        if (m >= CELLS) m = CELLS - 1;
        const double t = 2.0 * (u - m) - 1.0;
        const double* c = &coef_[(static_cast<std::size_t>(e - 1 - EMIN) * CELLS + static_cast<std::size_t>(m)) * (DEGREE + 1)];
        double y = c[DEGREE];
        for (int k = DEGREE - 1; k >= 0; --k) y = y * t + c[k];
        return y;
    }

    double exact(double d) const { return exact_(d); }
    bool tabulated() const { return xhi_ > xlo_; }
    const ResolvedKernel& resolved() const { return exact_; }

private:
    ResolvedKernel exact_{};
    double inv_s_ = 1.0, xlo_ = 0.0, xhi_ = 0.0;   // xhi_ <= xlo_: no table, closed form only
    std::vector<double> coef_;

    static bool worth_tabulating(KernelId id) {
        // Linear and cubic are cheaper in closed form; spherical goes to the sparse solver.
        return id == KernelId::GenCauchy || id == KernelId::Gaussian || id == KernelId::Exponential ||
               id == KernelId::InverseMultiquadric || id == KernelId::Multiquadric || id == KernelId::ThinPlate;
    }

    void build(double d_max) {
        if (!worth_tabulating(exact_.id) || !std::isfinite(d_max) || !(d_max > 0.0)) return;
        const double s = (exact_.id == KernelId::GenCauchy) ? 1.0 / exact_.gc_inv_b : exact_.epsilon;
        if (!std::isfinite(s) || !(s > 0.0)) return;
        int emax = static_cast<int>(std::ceil(std::log2(d_max * 1.01 / s))) + 1;
        emax = std::max(emax, EMIN + 1);
        if (emax > 64) return;
        const int bands = emax - EMIN;
        coef_.assign(static_cast<std::size_t>(bands) * CELLS * (DEGREE + 1), 0.0);

        double nodes[DEGREE + 1];
        for (int i = 0; i <= DEGREE; ++i) nodes[i] = std::cos(3.14159265358979323846 * (i + 0.5) / (DEGREE + 1));
        for (int b = 0; b < bands; ++b) {
            const double band_lo = std::ldexp(1.0, b + EMIN), w = band_lo / CELLS;
            for (int m = 0; m < CELLS; ++m) {
                // Monomial coefficients in t of the interpolant at the Chebyshev nodes (Gauss-Jordan, partial pivot).
                double A[DEGREE + 1][DEGREE + 2];
                for (int i = 0; i <= DEGREE; ++i) {
                    const double t = nodes[i];
                    double pw = 1.0;
                    for (int k = 0; k <= DEGREE; ++k) { A[i][k] = pw; pw *= t; }
                    A[i][DEGREE + 1] = exact_((band_lo + m * w + (t + 1.0) * 0.5 * w) * s);
                }
                for (int c = 0; c <= DEGREE; ++c) {
                    int pv = c;
                    for (int r = c + 1; r <= DEGREE; ++r) if (std::fabs(A[r][c]) > std::fabs(A[pv][c])) pv = r;
                    if (pv != c) for (int k = 0; k <= DEGREE + 1; ++k) std::swap(A[c][k], A[pv][k]);
                    for (int r = 0; r <= DEGREE; ++r) {
                        if (r == c) continue;
                        const double q = A[r][c] / A[c][c];
                        for (int k = c; k <= DEGREE + 1; ++k) A[r][k] -= q * A[c][k];
                    }
                }
                double* o = &coef_[(static_cast<std::size_t>(b) * CELLS + m) * (DEGREE + 1)];
                for (int k = 0; k <= DEGREE; ++k) o[k] = A[k][DEGREE + 1] / A[k][k];
            }
        }
        inv_s_ = 1.0 / s;
        xlo_ = std::ldexp(1.0, EMIN);
        xhi_ = std::ldexp(1.0, emax);

        // Check every cell at off-node points; fall back to the closed form if the table is not accurate.
        for (int b = 0; b < bands && tabulated(); ++b) {
            const double band_lo = std::ldexp(1.0, b + EMIN), w = band_lo / CELLS;
            for (int m = 0; m < CELLS; ++m) {
                for (double f : {0.0, 0.137, 0.5, 0.861, 0.9999}) {
                    const double d = (band_lo + (m + f) * w) * s;
                    const double ex = exact_(d), tb = (*this)(d);
                    if (!std::isfinite(tb) || std::fabs(tb - ex) > CHECK_TOL * std::max(1.0, std::fabs(ex))) {
                        xlo_ = xhi_ = 0.0;
                        coef_.clear();
                        return;
                    }
                }
            }
        }
    }
};

// Distance without std::hypot (the 3-argument hypot is several times slower and the coordinates here are bounded).
inline double fast_distance(const Point3& a, const Point3& b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Bounding-box diagonal of one or two point sets.
inline double bbox_diagonal(const std::vector<Point3>& a, const std::vector<Point3>* b = nullptr) {
    const double inf = std::numeric_limits<double>::infinity();
    double lo[3] = {inf, inf, inf}, hi[3] = {-inf, -inf, -inf};
    auto add = [&](const std::vector<Point3>& v) {
        for (const auto& p : v) for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], p[k]); hi[k] = std::max(hi[k], p[k]); }
    };
    add(a);
    if (b) add(*b);
    if (!(hi[0] >= lo[0])) return 0.0;
    return std::sqrt((hi[0]-lo[0])*(hi[0]-lo[0]) + (hi[1]-lo[1])*(hi[1]-lo[1]) + (hi[2]-lo[2])*(hi[2]-lo[2]));
}

}  // namespace deeprbf
