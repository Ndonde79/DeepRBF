#include "deeprbf/local_variogram.hpp"
#include "deeprbf/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace deeprbf {

namespace {

// Minimal internal spherical fit (grid search + local refinement).
// Kept here to avoid a dependency cycle with the geostatistics translation unit.
struct SphericalFit {
    double sill = 1.0;
    double range = 100.0;
    double nugget = 0.0;
    double rss = 0.0;
    double r_squared = 0.0;
    bool ok = false;
};

inline double sph_corr(double h, double range) {
    if (range <= 0.0 || h >= range) return 0.0;
    const double r = h / range;
    return 1.0 - (1.5 * r - 0.5 * r * r * r);
}

SphericalFit fit_spherical_simple(
    const std::vector<double>& lags,
    const std::vector<double>& gamma,
    double sill_guess,
    double range_guess,
    double nugget_guess
) {
    SphericalFit out;
    if (lags.size() < 3 || lags.size() != gamma.size()) return out;

    const double max_lag = lags.back();

    // Heuristic sill from the tail of the experimental curve
    if (sill_guess <= 0.0) {
        int n_tail = std::max(1, static_cast<int>(gamma.size() / 4));
        double s = 0.0;
        for (std::size_t i = gamma.size() - n_tail; i < gamma.size(); ++i) s += gamma[i];
        sill_guess = s / n_tail;
    }

    const double min_range = max_lag / 20.0;
    const double max_range = max_lag * 2.0;
    const double min_nug = 0.0;
    const double max_nug = std::max(sill_guess * 0.9, 1e-9);

    double range0 = (range_guess > 0.0) ? range_guess : max_lag * 0.6;
    range0 = std::clamp(range0, min_range, max_range);
    double nug0 = (nugget_guess >= 0.0) ? nugget_guess : gamma.front();
    nug0 = std::clamp(nug0, min_nug, max_nug);

    const int R_STEPS = 40;
    const int N_STEPS = 20;

    double best_rss = std::numeric_limits<double>::max();
    double best_r = range0, best_n = nug0, best_s = sill_guess;

    auto optimal_sill = [&](double r, double nug) -> double {
        double num = 0.0, den = 0.0;
        for (std::size_t k = 0; k < lags.size(); ++k) {
            const double c = 1.0 - sph_corr(lags[k], r);
            const double y = gamma[k] - nug;
            num += c * y;
            den += c * c;
        }
        if (den < 1e-14) return std::max(nug + 1e-9, sill_guess);
        return nug + std::max(num / den, 0.0);
    };

    for (int i = 0; i <= R_STEPS; ++i) {
        const double r = min_range + (max_range - min_range) * i / R_STEPS;
        for (int j = 0; j <= N_STEPS; ++j) {
            const double n = min_nug + (max_nug - min_nug) * j / N_STEPS;
            const double s = optimal_sill(r, n);
            double rss = 0.0;
            for (std::size_t k = 0; k < lags.size(); ++k) {
                const double th = n + (s - n) * (1.0 - sph_corr(lags[k], r));
                const double d = gamma[k] - th;
                rss += d * d;
            }
            if (rss < best_rss) {
                best_rss = rss; best_r = r; best_n = n; best_s = s;
            }
        }
    }

    // Local refinement
    double r_step = (max_range - min_range) / (R_STEPS * 2);
    double n_step = (max_nug - min_nug) / (N_STEPS * 2);
    for (int iter = 0; iter < 3; ++iter) {
        const double r_lo = std::max(min_range, best_r - r_step);
        const double r_hi = std::min(max_range, best_r + r_step);
        const double n_lo = std::max(min_nug, best_n - n_step);
        const double n_hi = std::min(max_nug, best_n + n_step);
        bool improved = false;
        const int LS = 8;
        for (int i = 0; i <= LS; ++i) {
            const double r = r_lo + (r_hi - r_lo) * i / LS;
            for (int j = 0; j <= LS; ++j) {
                const double n = n_lo + (n_hi - n_lo) * j / LS;
                const double s = optimal_sill(r, n);
                double rss = 0.0;
                for (std::size_t k = 0; k < lags.size(); ++k) {
                    const double th = n + (s - n) * (1.0 - sph_corr(lags[k], r));
                    const double d = gamma[k] - th;
                    rss += d * d;
                }
                if (rss < best_rss - 1e-12) {
                    best_rss = rss; best_r = r; best_n = n; best_s = s;
                    improved = true;
                }
            }
        }
        if (!improved) break;
        r_step *= 0.5;
        n_step *= 0.5;
    }

    // R^2
    double mean_g = std::accumulate(gamma.begin(), gamma.end(), 0.0) / gamma.size();
    double ss_tot = 0.0;
    for (double g : gamma) ss_tot += (g - mean_g) * (g - mean_g);

    out.sill = best_s;
    out.range = best_r;
    out.nugget = best_n;
    out.rss = best_rss;
    out.r_squared = (ss_tot > 1e-14) ? (1.0 - best_rss / ss_tot) : 0.0;
    out.ok = true;
    return out;
}

} // namespace

static void validate_local_inputs(const std::vector<Point3>& samples,const std::vector<double>& values,
    const std::vector<Point3>& targets,double radius,int min_points,int n_lags) {
    if(samples.size()!=values.size()||!std::isfinite(radius)||radius<=0||min_points<3||n_lags<3||n_lags>256)
        throw std::invalid_argument("Invalid local variogram arrays or controls");
    for(const auto& group:{&samples,&targets})for(const auto& p:*group)for(double v:p)
        if(!std::isfinite(v)||std::abs(v)>1e100)throw std::invalid_argument("Invalid local variogram coordinate");
    for(double v:values)if(!std::isfinite(v)||std::abs(v)>1e100)
        throw std::invalid_argument("Invalid local variogram value");
}

LocalVariogramParams fit_local_variogram_spherical(
    const std::vector<Point3>& samples,
    const std::vector<double>& values,
    const Point3& target,
    double search_radius,
    int min_points,
    int n_lags,
    double sill_guess,
    double range_guess,
    double nugget_guess
) {
    LocalVariogramParams result;
    result.valid = false;
    validate_local_inputs(samples,values,{target},search_radius,min_points,n_lags);
    for(double guess:{sill_guess,range_guess,nugget_guess})if(!std::isfinite(guess))
        throw std::invalid_argument("Nonfinite variogram initial guess");

    std::vector<Point3> local_pts;
    std::vector<double> local_vals;
    local_pts.reserve(samples.size());
    local_vals.reserve(samples.size());

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (euclidean_dist(samples[i], target) <= search_radius) {
            local_pts.push_back(samples[i]);
            local_vals.push_back(values[i]);
        }
    }

    if (static_cast<int>(local_pts.size()) < min_points) return result;
    if (n_lags < 2) n_lags = 2;

    const double max_lag = search_radius * 0.8;
    const double lag_width = max_lag / n_lags;

    std::vector<double> gamma(static_cast<std::size_t>(n_lags), 0.0);
    std::vector<int> pairs(static_cast<std::size_t>(n_lags), 0);

    for (std::size_t i = 0; i < local_pts.size(); ++i) {
        for (std::size_t j = i + 1; j < local_pts.size(); ++j) {
            const double h = euclidean_dist(local_pts[i], local_pts[j]);
            if(h<=0||h>=max_lag)continue;
            const int li = static_cast<int>(h / lag_width);
            if (li < 0 || li >= n_lags) continue;
            const double d = local_vals[i] - local_vals[j];
            gamma[li] += d * d;
            pairs[li] += 1;
        }
    }

    std::vector<double> lags, gammas;
    lags.reserve(n_lags);
    gammas.reserve(n_lags);
    for (int i = 0; i < n_lags; ++i) {
        if (pairs[i] > 0) {
            lags.push_back((i + 0.5) * lag_width);
            gammas.push_back(gamma[i] / (2.0 * pairs[i]));
        }
    }

    if (lags.size() < 3) return result;

    const SphericalFit fit =
        fit_spherical_simple(lags, gammas, sill_guess, range_guess, nugget_guess);

    if (!fit.ok) return result;

    result.sill = fit.sill;
    result.range = fit.range;
    result.nugget = fit.nugget;
    result.rss = fit.rss;
    result.r_squared = fit.r_squared;
    result.valid = true;
    return result;
}

std::vector<LocalVariogramParams> fit_local_variograms_batch(
    const std::vector<Point3>& samples,
    const std::vector<double>& values,
    const std::vector<Point3>& targets,
    double search_radius,
    int min_points,
    int n_lags
) {
    validate_local_inputs(samples,values,targets,search_radius,min_points,n_lags);
    std::vector<LocalVariogramParams> out(targets.size());

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(targets.size()); ++i) {
        out[static_cast<std::size_t>(i)] = fit_local_variogram_spherical(
            samples, values, targets[static_cast<std::size_t>(i)],
            search_radius, min_points, n_lags);
    }
    return out;
}

} // namespace deeprbf
