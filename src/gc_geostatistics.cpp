#include <memory>
// deeprbf/gc_geostatistics.cpp
// Core geostatistics: OK, IK, SGS, SIS, anisotropic, experimental
// variograms, automatic variogram fitting, cross-validation.

#include "deeprbf/gc_geostatistics.hpp"
#include "deeprbf/kernels.hpp"
#include "deeprbf/spatial_hash.hpp"
#include "deeprbf/local_variogram.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace deeprbf {

// Exact conditioning coordinates: no implicit grid snapping or distance tolerance.
template<class T> static std::map<Point3,T> conditioning_map(
    const std::vector<Point3>& points, const std::vector<T>& values) {
    std::map<Point3,T> known;
    for(size_t i=0;i<points.size();++i) {
        auto [it, inserted]=known.emplace(points[i],values[i]);
        if(!inserted && it->second!=values[i])
            throw std::invalid_argument("Conflicting conditioning values at identical coordinates");
    }
    return known;
}

constexpr double PI = 3.14159265358979323846;
constexpr int MAX_LOCAL_NEIGHBORS = 32;
constexpr int MISSING_CATEGORY = -999;
constexpr double REGULARIZATION = 1e-10;

static void output_budget(size_t rows,size_t columns){
    if(static_cast<long double>(rows)*columns>32000000.L)throw std::length_error("Native output exceeds 32 million entries; use bounded batches");
}
static void indicator_spec(const IndicatorVariogram& v){
    for(double r:{v.range_major,v.range_minor,v.range_vertical})if(!std::isfinite(r)||r<=0)throw std::invalid_argument("Invalid indicator range");
    for(double a:{v.azimuth_deg,v.dip_deg,v.plunge_deg})if(!std::isfinite(a))throw std::invalid_argument("Invalid indicator angle");
    spherical_covariance(0.,v.sill,v.nugget);
}

inline double deg2rad(double v) { return v * PI / 180.0; }

// =============================================================================
// Utility
// =============================================================================
double spherical_covariance(double h_scaled, double sill, double nugget) {
    if(!std::isfinite(h_scaled)||h_scaled<0)throw std::invalid_argument("Finite nonnegative covariance distance required");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    double h = std::min(h_scaled, 1.0);
    if (h <= 0.0) return sill;
    double corr = 1.0 - 1.5*h + 0.5*h*h*h;
    return (sill - nugget) * corr;
}

double anisotropic_scaled_distance(
    const Point3& a, const Point3& b,
    double range_major, double range_minor, double range_vertical,
    double azimuth_deg, double dip_deg, double plunge_deg
) {
    double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    double az = deg2rad(azimuth_deg), dip = deg2rad(dip_deg), pl = deg2rad(plunge_deg);
    double caz = std::cos(az), saz = std::sin(az);
    double x1 =  caz*dx + saz*dy;
    double y1 = -saz*dx + caz*dy;
    double z1 = dz;
    double cdip = std::cos(dip), sdip = std::sin(dip);
    double x2 = x1;
    double y2 =  cdip*y1 + sdip*z1;
    double z2 = -sdip*y1 + cdip*z1;
    double cpl = std::cos(pl), spl = std::sin(pl);
    double x3 =  cpl*x2 - spl*z2;
    double y3 = y2;
    double z3 =  spl*x2 + cpl*z2;
    double r_maj = std::max(range_major, 1e-9);
    double r_min = std::max(range_minor, 1e-9);
    double r_ver = std::max(range_vertical, 1e-9);
    return std::sqrt((x3/r_maj)*(x3/r_maj) + (y3/r_min)*(y3/r_min) + (z3/r_ver)*(z3/r_ver));
}

static Point3 transform_point_anisotropic(const Point3& p, const IndicatorVariogram& vg) {
    double az = deg2rad(vg.azimuth_deg);
    double dip = deg2rad(vg.dip_deg);
    double pl = deg2rad(vg.plunge_deg);
    double ca = std::cos(az), sa = std::sin(az);
    double cd = std::cos(dip), sd = std::sin(dip);
    double cp = std::cos(pl), sp = std::sin(pl);
    double x1 =  ca * p[0] + sa * p[1];
    double y1 = -sa * p[0] + ca * p[1];
    double z1 = p[2];
    double x2 = x1;
    double y2 =  cd * y1 + sd * z1;
    double z2 = -sd * y1 + cd * z1;
    double x3 =  cp * x2 - sp * z2;
    double y3 = y2;
    double z3 =  sp * x2 + cp * z2;
    double rx = std::max(vg.range_major, 1e-6);
    double ry = std::max(vg.range_minor, 1e-6);
    double rz = std::max(vg.range_vertical, 1e-6);
    return {x3 / rx, y3 / ry, z3 / rz};
}

// =============================================================================
// Local kriging kernels
// =============================================================================
static bool local_ordinary_kriging(
    const Point3& target,
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
    double var = sill;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double h = euclidean_dist(cond_pts[i], cond_pts[j]) / range;
            A(i,j) = spherical_covariance(h, var, nugget);
        }
        A(i,m) = 1.0;
        A(m,i) = 1.0;
        double h0 = euclidean_dist(target, cond_pts[i]) / range;
        b(i) = spherical_covariance(h0, var, nugget);
    }
    A(m,m) = 0.0;
    b(m) = 1.0;

    A.topLeftCorner(m,m).diagonal().array() += REGULARIZATION;
    Eigen::FullPivLU<Eigen::MatrixXd> factor(A);
    if(factor.rank()!=A.rows())return false;
    Eigen::VectorXd lambda = factor.solve(b);
    if (!lambda.allFinite()||(A*lambda-b).norm()>1e-9*std::max(1.,b.norm())) return false;

    estimate = 0.0;
    for (int i = 0; i < m; ++i) estimate += lambda(i) * cond_vals[i];
    variance = var - lambda.head(m).dot(b.head(m)) - lambda(m);
    if(!std::isfinite(variance)||variance < -1e-9*std::max(1.,var))return false;
    variance = std::max(variance, 0.);
    return true;
}

static bool local_ordinary_kriging_anisotropic(
    const Point3& target,
    const std::vector<Point3>& cond_pts,
    const std::vector<double>& cond_vals,
    const IndicatorVariogram& vg,
    double& estimate, double& variance
) {
    const int m = static_cast<int>(cond_pts.size());
    if (m < 1 || m > MAX_LOCAL_NEIGHBORS) return false;

    Eigen::MatrixXd A(m+1, m+1);
    Eigen::VectorXd b(m+1);
    A.setZero(); b.setZero();
    double var = vg.sill;
    double nugget = vg.nugget;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double h = anisotropic_scaled_distance(
                cond_pts[i], cond_pts[j],
                vg.range_major, vg.range_minor, vg.range_vertical,
                vg.azimuth_deg, vg.dip_deg, vg.plunge_deg);
            A(i,j) = spherical_covariance(h, var, nugget);
        }
        A(i,m) = 1.0;
        A(m,i) = 1.0;
        double h0 = anisotropic_scaled_distance(
            target, cond_pts[i],
            vg.range_major, vg.range_minor, vg.range_vertical,
            vg.azimuth_deg, vg.dip_deg, vg.plunge_deg);
        b(i) = spherical_covariance(h0, var, nugget);
    }
    A(m,m) = 0.0;
    b(m) = 1.0;

    A.topLeftCorner(m,m).diagonal().array() += REGULARIZATION;
    Eigen::FullPivLU<Eigen::MatrixXd> factor(A);
    if(factor.rank()!=A.rows())return false;
    Eigen::VectorXd lambda = factor.solve(b);
    if (!lambda.allFinite()||(A*lambda-b).norm()>1e-9*std::max(1.,b.norm())) return false;

    estimate = 0.0;
    for (int i = 0; i < m; ++i) estimate += lambda(i) * cond_vals[i];
    variance = var - lambda.head(m).dot(b.head(m)) - lambda(m);
    if(!std::isfinite(variance)||variance < -1e-9*std::max(1.,var))return false;
    variance = std::max(variance, 0.);
    return true;
}

static bool local_indicator_kriging(
    const Point3& target,
    const std::vector<Point3>& cond_pts,
    const std::vector<double>& indicators,
    double global_prop,
    double range, double sill, double nugget,
    double& prob, double& variance
) {
    const int m = static_cast<int>(cond_pts.size());
    if (m < 1) return false;

    double data_var = global_prop * (1.0 - global_prop);
    if (data_var == 0.) { prob=global_prop; variance=0.; return true; }
    double eff_sill = data_var * sill;
    double eff_nugget = data_var * nugget;

    Eigen::MatrixXd A(m, m);
    Eigen::VectorXd b(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double h = euclidean_dist(cond_pts[i], cond_pts[j]) / range;
            A(i,j) = spherical_covariance(h, eff_sill, eff_nugget);
        }
        double h0 = euclidean_dist(target, cond_pts[i]) / range;
        b(i) = spherical_covariance(h0, eff_sill, eff_nugget);
    }
    A.diagonal().array() += REGULARIZATION;
    Eigen::LDLT<Eigen::MatrixXd> factor(A);
    if(factor.info()!=Eigen::Success||!factor.isPositive())return false;
    Eigen::VectorXd lambda = factor.solve(b);
    if (!lambda.allFinite()||(A*lambda-b).norm()>1e-9*std::max(1.,b.norm())) return false;

    prob = global_prop;
    for (int i = 0; i < m; ++i) prob += lambda(i) * (indicators[i] - global_prop);
    prob = std::clamp(prob, 0.0, 1.0);
    variance = eff_sill - lambda.dot(b);
    if(!std::isfinite(variance)||variance < -1e-9*std::max(1.,eff_sill))return false;
    variance = std::max(variance, 0.);
    return true;
}

static bool local_indicator_kriging_anisotropic(
    const Point3& target,
    const std::vector<Point3>& cond_pts,
    const std::vector<double>& indicators,
    double global_prop,
    const IndicatorVariogram& vg,
    double& prob, double& variance
) {
    const int m = static_cast<int>(cond_pts.size());
    if (m < 1) return false;

    double data_var = global_prop * (1.0 - global_prop);
    if (data_var == 0.) { prob=global_prop; variance=0.; return true; }
    double eff_sill = data_var * vg.sill;
    double eff_nugget = data_var * vg.nugget;

    Eigen::MatrixXd A(m, m);
    Eigen::VectorXd b(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double h = anisotropic_scaled_distance(
                cond_pts[i], cond_pts[j],
                vg.range_major, vg.range_minor, vg.range_vertical,
                vg.azimuth_deg, vg.dip_deg, vg.plunge_deg);
            A(i,j) = spherical_covariance(h, eff_sill, eff_nugget);
        }
        double h0 = anisotropic_scaled_distance(
            target, cond_pts[i],
            vg.range_major, vg.range_minor, vg.range_vertical,
            vg.azimuth_deg, vg.dip_deg, vg.plunge_deg);
        b(i) = spherical_covariance(h0, eff_sill, eff_nugget);
    }
    A.diagonal().array() += REGULARIZATION;
    Eigen::LDLT<Eigen::MatrixXd> factor(A);
    if(factor.info()!=Eigen::Success||!factor.isPositive())return false;
    Eigen::VectorXd lambda = factor.solve(b);
    if (!lambda.allFinite()||(A*lambda-b).norm()>1e-9*std::max(1.,b.norm())) return false;

    prob = global_prop;
    for (int i = 0; i < m; ++i) prob += lambda(i) * (indicators[i] - global_prop);
    prob = std::clamp(prob, 0.0, 1.0);
    variance = eff_sill - lambda.dot(b);
    if(!std::isfinite(variance)||variance < -1e-9*std::max(1.,eff_sill))return false;
    variance = std::max(variance, 0.);
    return true;
}

// =============================================================================
// Global proportions
// =============================================================================
std::vector<double> compute_global_category_proportions(
    const std::vector<int>& sample_categories,
    const std::vector<int>& categories
) {
    if(categories.empty())throw std::invalid_argument("categories must not be empty");
    std::set<int> catalogue(categories.begin(),categories.end());
    if(catalogue.size()!=categories.size())throw std::invalid_argument("Duplicate category catalogue");
    for(int c:sample_categories)if(!catalogue.count(c))throw std::invalid_argument("Unknown conditioning category");

    std::vector<double> props(categories.size(), 0.0);
    if (sample_categories.empty() || categories.empty()) return props;

    for (size_t c = 0; c < categories.size(); ++c) {
        int cnt = static_cast<int>(std::count(
            sample_categories.begin(), sample_categories.end(), categories[c]));
        props[c] = static_cast<double>(cnt) / sample_categories.size();
    }
    double sum = std::accumulate(props.begin(), props.end(), 0.0);
    if (sum <= 1e-12) {
        std::fill(props.begin(), props.end(), 1.0 / categories.size());
    } else {
        for (double& p : props) p /= sum;
    }
    return props;
}

// =============================================================================
// probabilities_from_local_ik_anisotropic
// =============================================================================
std::vector<double> probabilities_from_local_ik_anisotropic(
    const Point3& target,
    const std::vector<Point3>& cond_points,
    const std::vector<int>& cond_categories,
    const std::vector<int>& categories,
    const std::vector<IndicatorVariogram>& variograms,
    const std::vector<double>& fallback
) {
    if(categories.empty())throw std::invalid_argument("categories must not be empty");

    std::vector<double> global_props = fallback;
    double sum_fb = std::accumulate(global_props.begin(), global_props.end(), 0.0);
    if (global_props.empty() || sum_fb <= 1e-12) {
        global_props = compute_global_category_proportions(cond_categories, categories);
    }

    std::unordered_map<int, IndicatorVariogram> vg_map;
    for (int cat : categories) {
        auto it = std::find_if(variograms.begin(), variograms.end(),
                               [cat](const IndicatorVariogram& v) { return v.category == cat; });
        if (it != variograms.end()) {
            vg_map[cat] = *it;
        } else {
            IndicatorVariogram vg;
            vg.category = cat;
            vg.range_major = vg.range_minor = vg.range_vertical = 100.0;
            vg.sill = 1.0; vg.nugget = 0.05;
            vg_map[cat] = vg;
        }
    }

    std::vector<double> probs(categories.size(), 0.0);
    for (size_t c = 0; c < categories.size(); ++c) {
        int cat = categories[c];
        const auto& vg = vg_map[cat];
        double global_p = global_props[c];

        std::vector<Point3> trans_cond(cond_points.size());
        for (size_t i = 0; i < cond_points.size(); ++i)
            trans_cond[i] = transform_point_anisotropic(cond_points[i], vg);
        Point3 trans_target = transform_point_anisotropic(target, vg);

        // simple squared-distance neighbour selection in the transformed space
        std::vector<std::pair<size_t, double>> matches;
        matches.reserve(cond_points.size());
        for (size_t i = 0; i < trans_cond.size(); ++i) {
            const double dx = trans_target[0] - trans_cond[i][0];
            const double dy = trans_target[1] - trans_cond[i][1];
            const double dz = trans_target[2] - trans_cond[i][2];
            const double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 <= 1.0) matches.emplace_back(i, d2);
        }
        std::sort(matches.begin(), matches.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        const int max_neighbors = 32;
        if (matches.size() > static_cast<size_t>(max_neighbors))
            matches.resize(max_neighbors);

        std::vector<Point3> orig_cond_pts;
        std::vector<double> indicators(matches.size());
        for (size_t k = 0; k < matches.size(); ++k) {
            size_t idx = matches[k].first;
            orig_cond_pts.push_back(cond_points[idx]);
            indicators[k] = (cond_categories[idx] == cat) ? 1.0 : 0.0;
        }

        double prob = global_p, var = 0.0;
        if (local_indicator_kriging_anisotropic(target, orig_cond_pts, indicators,
                                                global_p, vg, prob, var)) {
            probs[c] = std::clamp(prob, 0.0, 1.0);
        } else {
            probs[c] = global_p;
        }
    }
    for (double& p : probs) p = std::max(p, 0.0);
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (sum > 1e-12) {
        for (double& p : probs) p /= sum;
    } else {
        std::fill(probs.begin(), probs.end(), 1.0 / categories.size());
    }
    return probs;
}

// =============================================================================
// Ordinary Kriging — public API
// =============================================================================
OKResult ordinary_kriging(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    return ordinary_kriging_with_range(sample_xyz, sample_values, target_xyz,
                                       max_neighbors, min_neighbors,
                                       search_radius, search_radius,
                                       sill, nugget);
}

OKResult ordinary_kriging_with_range(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int max_neighbors, int min_neighbors,
    double search_radius, double variogram_range,
    double sill, double nugget
) {
    if(!std::isfinite(variogram_range)||variogram_range<=0)throw std::invalid_argument("Finite positive variogram range required");
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    if (sample_xyz.size() != sample_values.size())
        throw std::runtime_error("ordinary_kriging_with_range: size mismatch");

    OKResult out;
    out.estimate.assign(target_xyz.size(), std::numeric_limits<double>::quiet_NaN());
    out.variance.assign(target_xyz.size(), std::numeric_limits<double>::quiet_NaN());
    out.n_samples.assign(target_xyz.size(), 0);
    if (sample_xyz.empty() || target_xyz.empty()) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);
    double range = std::max(variogram_range, 1e-9);

    SpatialHash hash(std::max(search_radius / 4.0, 1.0));
    hash.build(sample_xyz, sample_values);
    for(const auto& target:target_xyz)hash.validate_target(target);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(target_xyz.size()); ++i) {
        auto matches = hash.query(target_xyz[static_cast<std::size_t>(i)],
                                  search_radius, max_neighbors);
        if (static_cast<int>(matches.size()) < min_neighbors) continue;

        std::vector<Point3> cond_pts; cond_pts.reserve(matches.size());
        std::vector<double> cond_vals; cond_vals.reserve(matches.size());
        for (const auto& m : matches) {
            cond_pts.push_back(hash.points[static_cast<size_t>(m.index)]);
            cond_vals.push_back(hash.values[static_cast<size_t>(m.index)]);
        }

        double est, var;
        if (local_ordinary_kriging(target_xyz[static_cast<std::size_t>(i)],
                                   cond_pts, cond_vals, range, sill, nugget, est, var)) {
            out.estimate[static_cast<std::size_t>(i)] = est;
            out.variance[static_cast<std::size_t>(i)] = var;
            out.n_samples[static_cast<std::size_t>(i)] = static_cast<int>(matches.size());
        }
    }
    return out;
}

OKResult ordinary_kriging_anisotropic(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    const IndicatorVariogram& variogram,
    int max_neighbors, int min_neighbors,
    double search_radius
) {
    for(double v:{variogram.range_major,variogram.range_minor,variogram.range_vertical})
        if(!std::isfinite(v)||v<=0)throw std::invalid_argument("Invalid anisotropic covariance range");
    for(double v:{variogram.azimuth_deg,variogram.dip_deg,variogram.plunge_deg})
        if(!std::isfinite(v))throw std::invalid_argument("Invalid anisotropic covariance angle");
    spherical_covariance(0.,variogram.sill,variogram.nugget);
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");

    if (sample_xyz.size() != sample_values.size())
        throw std::runtime_error("ordinary_kriging_anisotropic: size mismatch");

    OKResult out;
    out.estimate.assign(target_xyz.size(), std::numeric_limits<double>::quiet_NaN());
    out.variance.assign(target_xyz.size(), std::numeric_limits<double>::quiet_NaN());
    out.n_samples.assign(target_xyz.size(), 0);
    if (sample_xyz.empty() || target_xyz.empty()) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);

    SpatialHash hash(std::max(search_radius / 4.0, 1.0));
    hash.build(sample_xyz, sample_values);
    for(const auto& target:target_xyz)hash.validate_target(target);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(target_xyz.size()); ++i) {
        auto matches = hash.query(target_xyz[static_cast<std::size_t>(i)],
                                  search_radius, max_neighbors);
        if (static_cast<int>(matches.size()) < min_neighbors) continue;

        std::vector<Point3> cond_pts; cond_pts.reserve(matches.size());
        std::vector<double> cond_vals; cond_vals.reserve(matches.size());
        for (const auto& m : matches) {
            cond_pts.push_back(hash.points[static_cast<size_t>(m.index)]);
            cond_vals.push_back(hash.values[static_cast<size_t>(m.index)]);
        }

        double est, var;
        if (local_ordinary_kriging_anisotropic(target_xyz[static_cast<std::size_t>(i)],
                                               cond_pts, cond_vals, variogram, est, var)) {
            out.estimate[static_cast<std::size_t>(i)] = est;
            out.variance[static_cast<std::size_t>(i)] = var;
            out.n_samples[static_cast<std::size_t>(i)] = static_cast<int>(matches.size());
        }
    }
    return out;
}

OKResult ordinary_kriging_local(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    double search_radius,
    int max_neighbors,
    int min_neighbors,
    int min_points_for_variogram,
    int n_lags,
    double sill_guess,
    double range_guess,
    double nugget_guess
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");

    if (sample_xyz.size() != sample_values.size())
        throw std::runtime_error("ordinary_kriging_local: size mismatch");

    const size_t n_target = target_xyz.size();
    OKResult result;
    result.estimate.assign(n_target, std::numeric_limits<double>::quiet_NaN());
    result.variance.assign(n_target, std::numeric_limits<double>::quiet_NaN());
    result.n_samples.assign(n_target, 0);
    if (sample_xyz.empty() || target_xyz.empty()) return result;

    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n_target); ++i) {
        const Point3& target = target_xyz[static_cast<std::size_t>(i)];

        LocalVariogramParams params = fit_local_variogram_spherical(
            sample_xyz, sample_values, target,
            search_radius, min_points_for_variogram, n_lags,
            sill_guess, range_guess, nugget_guess);

        if (!params.valid) continue;

        std::vector<Point3> one_target = {target};
        OKResult local_res = ordinary_kriging_with_range(
            sample_xyz, sample_values, one_target,
            max_neighbors, min_neighbors,
            search_radius, params.range,
            params.sill, params.nugget);

        result.estimate[static_cast<std::size_t>(i)] = local_res.estimate[0];
        result.variance[static_cast<std::size_t>(i)] = local_res.variance[0];
        result.n_samples[static_cast<std::size_t>(i)] = local_res.n_samples[0];
    }
    return result;
}

// =============================================================================
// Nested variogram OK
// =============================================================================
bool local_ordinary_kriging_nested(
    const Point3& target,
    const std::vector<Point3>& cond_pts,
    const std::vector<double>& cond_vals,
    const NestedVariogram& vg,
    double& estimate,
    double& variance
) {
    const int m = static_cast<int>(cond_pts.size());
    if (m < 1 || m > MAX_LOCAL_NEIGHBORS) return false;

    Eigen::MatrixXd A(m+1, m+1);
    Eigen::VectorXd b(m+1);
    A.setZero(); b.setZero();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double dist = euclidean_dist(cond_pts[i], cond_pts[j]);
            A(i,j) = vg.covariance(dist);
        }
        A(i,m) = 1.0;
        A(m,i) = 1.0;
        double dist0 = euclidean_dist(target, cond_pts[i]);
        b(i) = vg.covariance(dist0);
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
    variance = vg.sill() - lambda.head(m).dot(b.head(m)) - lambda(m);
    if(!std::isfinite(variance)||variance < -1e-9*std::max(1.,vg.sill()))return false;
    variance = std::max(variance, 0.);
    return true;
}

OKResult ordinary_kriging_nested(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    const NestedVariogram& variogram,
    int max_neighbors,
    int min_neighbors,
    double search_radius
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");

    if (sample_xyz.size() != sample_values.size())
        throw std::runtime_error("ordinary_kriging_nested: size mismatch");

    OKResult out;
    out.estimate.assign(target_xyz.size(), std::numeric_limits<double>::quiet_NaN());
    out.variance.assign(target_xyz.size(), std::numeric_limits<double>::quiet_NaN());
    out.n_samples.assign(target_xyz.size(), 0);
    if (sample_xyz.empty() || target_xyz.empty()) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);

    SpatialHash hash(std::max(search_radius / 4.0, 1.0));
    hash.build(sample_xyz, sample_values);
    for(const auto& target:target_xyz)hash.validate_target(target);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(target_xyz.size()); ++i) {
        auto matches = hash.query(target_xyz[static_cast<std::size_t>(i)],
                                  search_radius, max_neighbors);
        if (static_cast<int>(matches.size()) < min_neighbors) continue;

        std::vector<Point3> cond_pts; cond_pts.reserve(matches.size());
        std::vector<double> cond_vals; cond_vals.reserve(matches.size());
        for (const auto& m : matches) {
            cond_pts.push_back(hash.points[static_cast<size_t>(m.index)]);
            cond_vals.push_back(hash.values[static_cast<size_t>(m.index)]);
        }

        double est, var;
        if (local_ordinary_kriging_nested(target_xyz[static_cast<std::size_t>(i)],
                                          cond_pts, cond_vals, variogram, est, var)) {
            out.estimate[static_cast<std::size_t>(i)] = est;
            out.variance[static_cast<std::size_t>(i)] = var;
            out.n_samples[static_cast<std::size_t>(i)] = static_cast<int>(matches.size());
        }
    }
    return out;
}

// =============================================================================
// Indicator Kriging
// =============================================================================
IKResult indicator_kriging(
    const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_categories.size())throw std::invalid_argument("sample/value size mismatch");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");
    if(categories.empty())throw std::invalid_argument("categories must not be empty");

    if (sample_xyz.size() != sample_categories.size())
        throw std::runtime_error("indicator_kriging: size mismatch");
    const auto known=conditioning_map(sample_xyz,sample_categories);
    const auto global_props=compute_global_category_proportions(sample_categories,categories);

    IKResult out;
    out.n_targets = target_xyz.size();
    out.n_categories = categories.size();
    out.categories = categories;
    output_budget(out.n_targets,out.n_categories);
    out.probabilities.assign(out.n_targets * out.n_categories, 0.0);
    out.n_samples.assign(out.n_targets, 0);
    if (sample_xyz.empty() || target_xyz.empty()) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);

    std::vector<double> sample_vals_double;
    sample_vals_double.reserve(sample_categories.size());
    for (int c : sample_categories)
        sample_vals_double.push_back(static_cast<double>(c));

    SpatialHash hash(std::max(search_radius / 4.0, 1.0));
    hash.build(sample_xyz, sample_vals_double);
    for(const auto& target:target_xyz)hash.validate_target(target);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(out.n_targets); ++i) {
        const auto hard=known.find(target_xyz[static_cast<size_t>(i)]);
        if(hard!=known.end()) {
            for(size_t c=0;c<categories.size();++c)out.probabilities[i*out.n_categories+c]=double(categories[c]==hard->second);
            out.n_samples[i]=1;continue;
        }
        auto matches = hash.query(target_xyz[static_cast<std::size_t>(i)],
                                  search_radius, max_neighbors);

        std::vector<double> local_probs = global_props;

        if (static_cast<int>(matches.size()) >= min_neighbors) {
            std::vector<Point3> cond_pts; cond_pts.reserve(matches.size());
            std::vector<int> cond_cats; cond_cats.reserve(matches.size());
            for (const auto& m : matches) {
                cond_pts.push_back(hash.points[static_cast<size_t>(m.index)]);
                cond_cats.push_back(static_cast<int>(
                    std::llround(hash.values[static_cast<size_t>(m.index)])));
            }

            for (size_t c = 0; c < categories.size(); ++c) {
                std::vector<double> indicators(matches.size());
                for (size_t k = 0; k < matches.size(); ++k)
                    indicators[k] = (cond_cats[k] == categories[c]) ? 1.0 : 0.0;

                double prob, var;
                if (local_indicator_kriging(target_xyz[static_cast<std::size_t>(i)],
                                            cond_pts, indicators, global_props[c],
                                            search_radius, sill, nugget, prob, var)) {
                    local_probs[c] = prob;
                }
            }
            out.n_samples[static_cast<std::size_t>(i)] = static_cast<int>(matches.size());
        }

        for (double& p : local_probs) p = std::max(p, 0.0);
        double sum = std::accumulate(local_probs.begin(), local_probs.end(), 0.0);
        if (sum > 1e-12) {
            for (double& p : local_probs) p /= sum;
        } else {
            std::fill(local_probs.begin(), local_probs.end(), 1.0 / categories.size());
        }

        for (size_t c = 0; c < categories.size(); ++c)
            out.probabilities[i * out.n_categories + c] = local_probs[c];
    }
    return out;
}

IKResult indicator_kriging_anisotropic(
    const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    const std::vector<IndicatorVariogram>& variograms,
    int max_neighbors, int min_neighbors,
    double search_radius_aniso
) {
    for(const auto& v:variograms)indicator_spec(v);
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_categories.size())throw std::invalid_argument("sample/value size mismatch");
    if(!std::isfinite(search_radius_aniso)||search_radius_aniso<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(categories.empty())throw std::invalid_argument("categories must not be empty");

    if (sample_xyz.size() != sample_categories.size())
        throw std::runtime_error("indicator_kriging_anisotropic: size mismatch");
    const auto known=conditioning_map(sample_xyz,sample_categories);
    const auto global_props=compute_global_category_proportions(sample_categories,categories);

    IKResult out;
    out.n_targets = target_xyz.size();
    out.n_categories = categories.size();
    out.categories = categories;
    output_budget(out.n_targets,out.n_categories);
    out.probabilities.assign(out.n_targets * out.n_categories, 0.0);
    out.n_samples.assign(out.n_targets, 0);
    if (sample_xyz.empty() || target_xyz.empty()) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);

    std::vector<int> sample_cats_int(sample_categories.begin(), sample_categories.end());
    std::vector<double> sample_vals_double(sample_cats_int.begin(), sample_cats_int.end());

    // Legacy public API search_radius is in coordinate units. Covariance is
    // anisotropic; use the audited NeighborhoodSpec API for ellipsoidal search.
    SpatialHash hash(std::max(search_radius_aniso / 4.0, 1.0));
    hash.build(sample_xyz, sample_vals_double);
    for(const auto& target:target_xyz)hash.validate_target(target);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 32)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(out.n_targets); ++i) {
        Point3 target = target_xyz[static_cast<std::size_t>(i)];
        const auto hard=known.find(target);
        if(hard!=known.end()) {
            for(size_t c=0;c<categories.size();++c)out.probabilities[i*out.n_categories+c]=double(categories[c]==hard->second);
            out.n_samples[i]=1;continue;
        }
        auto matches = hash.query(target, search_radius_aniso, max_neighbors);

        std::vector<double> local_probs = global_props;

        if (static_cast<int>(matches.size()) >= min_neighbors) {
            std::vector<Point3> cond_pts; cond_pts.reserve(matches.size());
            std::vector<int> cond_cats; cond_cats.reserve(matches.size());
            for (const auto& m : matches) {
                cond_pts.push_back(sample_xyz[static_cast<size_t>(m.index)]);
                cond_cats.push_back(static_cast<int>(
                    std::llround(hash.values[static_cast<size_t>(m.index)])));
            }

            local_probs = probabilities_from_local_ik_anisotropic(
                target, cond_pts, cond_cats, categories, variograms, global_props);

            out.n_samples[static_cast<std::size_t>(i)] = static_cast<int>(matches.size());
        }

        for (size_t c = 0; c < categories.size(); ++c)
            out.probabilities[i * out.n_categories + c] = local_probs[c];
    }
    return out;
}

// =============================================================================
// Experimental indicator variograms
// =============================================================================
struct PairEntry {
    size_t i, j;
    double distance;
};

// Bounded reservoir sampling per directional lag; counts describe the pairs
// actually used. Azimuth is counterclockwise from +X; dip is positive toward +Z.
static std::vector<VariogramResult> indicator_variograms(
    const std::vector<Point3>& points,const std::vector<int>& labels,
    const std::vector<std::pair<int,int>>& pairs,const std::vector<Direction>& directions,
    int n_lags,double max_lag,double lag_width,int max_pairs_per_lag) {
    if(points.size()!=labels.size()||n_lags<1||n_lags>256||max_pairs_per_lag<1||
       !std::isfinite(max_lag)||!std::isfinite(lag_width)||
       std::min(static_cast<long double>(n_lags)*max_pairs_per_lag,
                static_cast<long double>(points.size())*(points.size()-static_cast<size_t>(!points.empty()))/2)>2000000.L)
        throw std::invalid_argument("Invalid indicator variogram arrays/lag controls");
    if(static_cast<long double>(points.size())*(points.size()-static_cast<size_t>(!points.empty()))/2>50000000.L)
        throw std::invalid_argument("Exact pair scan exceeds 50 million pairs; use explicit sampled variography");
    for(const auto& p:points)for(double v:p)if(!std::isfinite(v)||std::abs(v)>1e100)throw std::invalid_argument("Invalid variogram coordinate");
    for(const auto& d:directions)if(!std::isfinite(d.azimuth_deg)||!std::isfinite(d.dip_deg)||!std::isfinite(d.tolerance_deg)||d.tolerance_deg<0||d.tolerance_deg>90)
        throw std::invalid_argument("Invalid variogram direction");
    std::vector<VariogramResult> out;
    if(points.size()<2||pairs.empty())return out;
    if(max_lag<=0){Point3 low=points.front(),high=low;
        for(const auto& p:points)for(int d=0;d<3;++d){low[d]=std::min(low[d],p[d]);high[d]=std::max(high[d],p[d]);}
        max_lag=.5*euclidean_dist(low,high);
    }
    if(max_lag<=0)return out;
    if(lag_width<=0)lag_width=max_lag/n_lags;
    for(const auto& direction:directions){
        double az=deg2rad(direction.azimuth_deg),dip=deg2rad(direction.dip_deg);
        const Point3 unit{std::cos(dip)*std::cos(az),std::cos(dip)*std::sin(az),std::sin(dip)};
        const double threshold=std::cos(deg2rad(direction.tolerance_deg));
        std::vector<std::vector<PairEntry>> bins(n_lags);std::vector<size_t> counts(n_lags,0);
        std::mt19937_64 rng(1729);
        for(size_t i=0;i<points.size();++i)for(size_t j=i+1;j<points.size();++j){
            const double h=euclidean_dist(points[i],points[j]);if(h<=0||h>max_lag)continue;
            double projection=0;for(int d=0;d<3;++d)projection+=(points[j][d]-points[i][d])*unit[d];
            if(std::abs(projection)/h+1e-14<threshold)continue;
            const double position=h/lag_width;if(position>n_lags)continue;
            const size_t lag=std::min(static_cast<size_t>(n_lags-1),static_cast<size_t>(std::ceil(position)-1));
            const size_t seen=++counts[lag];auto& selected=bins[lag];
            if(selected.size()<static_cast<size_t>(max_pairs_per_lag))selected.push_back({i,j,h});
            else {std::uniform_int_distribution<size_t> draw(0,seen-1);size_t index=draw(rng);if(index<selected.size())selected[index]={i,j,h};}
        }
        for(const auto& cats:pairs){VariogramResult result;result.categories={cats.first,cats.second};
            for(int lag=0;lag<n_lags;++lag){double gamma=0;
                for(const auto& pair:bins[lag]){
                    const double a=double(labels[pair.i]==cats.first)-double(labels[pair.j]==cats.first);
                    const double b=double(labels[pair.i]==cats.second)-double(labels[pair.j]==cats.second);
                    gamma+=.5*a*b;
                }
                const auto count=bins[lag].size();result.lags.push_back({(lag+.5)*lag_width,count?gamma/count:0.,static_cast<int>(count)});
            }out.push_back(std::move(result));
        }
    }return out;
}

std::vector<VariogramResult> compute_direct_indicator_variograms_fast(
    const std::vector<Point3>& points,const std::vector<int>& categories,const std::vector<int>& all_categories,
    const std::vector<Direction>& directions,int n_lags,double max_lag,double lag_width,int max_pairs_per_lag) {
    std::vector<std::pair<int,int>> pairs;for(int category:all_categories)pairs.emplace_back(category,category);
    return indicator_variograms(points,categories,pairs,directions,n_lags,max_lag,lag_width,max_pairs_per_lag);
}
std::vector<VariogramResult> compute_cross_indicator_variograms_fast(
    const std::vector<Point3>& points,const std::vector<int>& categories,const std::vector<std::pair<int,int>>& cross_pairs,
    const std::vector<Direction>& directions,int n_lags,double max_lag,double lag_width,int max_pairs_per_lag) {
    return indicator_variograms(points,categories,cross_pairs,directions,n_lags,max_lag,lag_width,max_pairs_per_lag);
}

// =============================================================================
// Variogram fitting
// =============================================================================
static inline double spherical_corr(double h, double range) {
    if (range <= 0.0 || h >= range) return 1.0;
    double hr = h / range;
    return 1.5 * hr - 0.5 * hr * hr * hr;
}

static inline double exponential_corr(double h, double range) {
    if (range <= 0.0) return 1.0;
    return 1.0 - std::exp(-h / range);
}

static inline double gaussian_corr(double h, double range) {
    if (range <= 0.0) return 1.0;
    double hr = h / range;
    return 1.0 - std::exp(-hr * hr);
}

static inline double spherical_model(double h, double sill, double range, double nugget) {
    return nugget + (sill - nugget) * spherical_corr(h, range);
}
static inline double exponential_model(double h, double sill, double range, double nugget) {
    return nugget + (sill - nugget) * exponential_corr(h, range);
}
static inline double gaussian_model(double h, double sill, double range, double nugget) {
    return nugget + (sill - nugget) * gaussian_corr(h, range);
}

static VariogramFitResult fit_variogram_model_impl(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_exp,
    const std::vector<double>& weights,
    const std::string& model_type,
    double sill_guess,
    double range_guess,
    double nugget_guess,
    double min_range = -1.0,
    double max_range = -1.0,
    double min_nugget = -1.0,
    double max_nugget = -1.0
) {
    VariogramFitResult res;
    res.model_type = model_type;
    size_t n = lag_centers.size();
    if(n==0||gamma_exp.size()!=n||(!weights.empty()&&weights.size()!=n))throw std::invalid_argument("invalid variogram arrays");
    for(double v:lag_centers)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid lag");
    for(double v:gamma_exp)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid semivariance");
    for(double v:weights)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid weight");
    if(*std::max_element(lag_centers.begin(),lag_centers.end())<=0)throw std::invalid_argument("positive lag required");
    if(!std::is_sorted(lag_centers.begin(),lag_centers.end()))throw std::invalid_argument("Variogram lags must be sorted");
    for(double v:{sill_guess,range_guess,nugget_guess,min_range,max_range,min_nugget,max_nugget})if(!std::isfinite(v))throw std::invalid_argument("Nonfinite variogram fit control");
    if(!weights.empty()&&std::accumulate(weights.begin(),weights.end(),0.)<=0)throw std::invalid_argument("positive weight required");

    std::vector<double> w = weights;
    if (w.empty() || w.size() != n) w.assign(n, 1.0);

    double max_lag = lag_centers.back();

    if (min_range <= 0.0) min_range = max_lag / 20.0;
    if (max_range <= 0.0) max_range = max_lag * 3.0;

    int n_sill = std::max(1, static_cast<int>(n / 5));
    double exp_sill = 0.0;
    for (size_t i = n - n_sill; i < n; ++i) exp_sill += gamma_exp[i];
    exp_sill /= n_sill;
    if (sill_guess <= 0.0) sill_guess = exp_sill;

    if (min_nugget < 0.0) min_nugget = 0.0;
    if (max_nugget <= 0.0) max_nugget = sill_guess;

    if (range_guess <= 0.0) range_guess = max_lag * 0.6;
    if (nugget_guess < 0.0) nugget_guess = gamma_exp[0];
    if(min_range<=0||max_range<min_range||max_nugget<min_nugget)throw std::invalid_argument("Inverted variogram fit bounds");

    range_guess = std::clamp(range_guess, min_range, max_range);
    nugget_guess = std::clamp(nugget_guess, min_nugget, max_nugget);

    const int n_range_steps = 40;
    const int n_nugget_steps = 20;
    double best_rss = std::numeric_limits<double>::max();
    double best_range = range_guess;
    double best_nugget = nugget_guess;
    double best_sill = sill_guess;

    auto compute_optimal_sill = [&](double r, double nug) -> double {
        double sum_w_base2 = 0.0, sum_w_base_y = 0.0;
        for (size_t k = 0; k < n; ++k) {
            double corr;
            if (model_type == "spherical") corr = spherical_corr(lag_centers[k], r);
            else if (model_type == "exponential") corr = exponential_corr(lag_centers[k], r);
            else corr = gaussian_corr(lag_centers[k], r);
            double y = gamma_exp[k] - nug;
            double wgt = w[k];
            sum_w_base2 += wgt * corr * corr;
            sum_w_base_y += wgt * corr * y;
        }
        if (sum_w_base2 < 1e-12) return std::max(nug + 1e-6, exp_sill);
        double sill = nug + sum_w_base_y / sum_w_base2;
        return std::max(sill, nug + 1e-6);
    };

    for (int i = 0; i <= n_range_steps; ++i) {
        double r = min_range + (max_range - min_range) * i / n_range_steps;
        for (int j = 0; j <= n_nugget_steps; ++j) {
            double nug = min_nugget + (max_nugget - min_nugget) * j / n_nugget_steps;
            double sill = compute_optimal_sill(r, nug);
            double rss = 0.0;
            for (size_t k = 0; k < n; ++k) {
                double gamma_th;
                if (model_type == "spherical") gamma_th = spherical_model(lag_centers[k], sill, r, nug);
                else if (model_type == "exponential") gamma_th = exponential_model(lag_centers[k], sill, r, nug);
                else gamma_th = gaussian_model(lag_centers[k], sill, r, nug);
                double diff = gamma_exp[k] - gamma_th;
                rss += w[k] * diff * diff;
            }
            if (rss < best_rss) {
                best_rss = rss;
                best_range = r;
                best_nugget = nug;
                best_sill = sill;
            }
        }
    }

    double range_step = (max_range - min_range) / (n_range_steps * 2);
    double nug_step = (max_nugget - min_nugget) / (n_nugget_steps * 2);
    for (int iter = 0; iter < 3; ++iter) {
        double r_low = std::max(min_range, best_range - range_step);
        double r_high = std::min(max_range, best_range + range_step);
        double n_low = std::max(min_nugget, best_nugget - nug_step);
        double n_high = std::min(max_nugget, best_nugget + nug_step);
        bool improved = false;
        const int local_steps = 8;
        for (int i = 0; i <= local_steps; ++i) {
            double r = r_low + (r_high - r_low) * i / local_steps;
            for (int j = 0; j <= local_steps; ++j) {
                double nug = n_low + (n_high - n_low) * j / local_steps;
                double sill = compute_optimal_sill(r, nug);
                double rss = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    double gamma_th;
                    if (model_type == "spherical") gamma_th = spherical_model(lag_centers[k], sill, r, nug);
                    else if (model_type == "exponential") gamma_th = exponential_model(lag_centers[k], sill, r, nug);
                    else gamma_th = gaussian_model(lag_centers[k], sill, r, nug);
                    double diff = gamma_exp[k] - gamma_th;
                    rss += w[k] * diff * diff;
                }
                if (rss < best_rss - 1e-12) {
                    best_rss = rss;
                    best_range = r;
                    best_nugget = nug;
                    best_sill = sill;
                    improved = true;
                }
            }
        }
        if (!improved) break;
        range_step *= 0.5;
        nug_step *= 0.5;
    }

    res.sill = best_sill;
    res.range = best_range;
    res.nugget = best_nugget;
    res.rss = best_rss;
    res.fitted_values.resize(n);

    double total_weight = std::accumulate(w.begin(), w.end(), 0.0);
    double mean_gamma = 0.0;
    for (size_t k = 0; k < n; ++k) mean_gamma += w[k] * gamma_exp[k];
    mean_gamma /= total_weight;

    double ss_tot = 0.0;
    for (size_t k = 0; k < n; ++k) {
        double diff = gamma_exp[k] - mean_gamma;
        ss_tot += w[k] * diff * diff;
        if (model_type == "spherical")
            res.fitted_values[k] = spherical_model(lag_centers[k], best_sill, best_range, best_nugget);
        else if (model_type == "exponential")
            res.fitted_values[k] = exponential_model(lag_centers[k], best_sill, best_range, best_nugget);
        else
            res.fitted_values[k] = gaussian_model(lag_centers[k], best_sill, best_range, best_nugget);
    }
    res.r_squared = 1.0 - (best_rss / (ss_tot + 1e-12));
    return res;
}

VariogramFitResult fit_spherical_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget
) {
    return fit_variogram_model_impl(lag_centers, gamma_values, weights, "spherical",
                                    sill_guess, range_guess, nugget_guess,
                                    min_range, max_range, min_nugget, max_nugget);
}

VariogramFitResult fit_exponential_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget
) {
    return fit_variogram_model_impl(lag_centers, gamma_values, weights, "exponential",
                                    sill_guess, range_guess, nugget_guess,
                                    min_range, max_range, min_nugget, max_nugget);
}

VariogramFitResult fit_gaussian_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget
) {
    return fit_variogram_model_impl(lag_centers, gamma_values, weights, "gaussian",
                                    sill_guess, range_guess, nugget_guess,
                                    min_range, max_range, min_nugget, max_nugget);
}

VariogramFitResult fit_spherical_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess
) {
    return fit_spherical_variogram(lag_centers, gamma_values, {},
                                   sill_guess, range_guess, nugget_guess,
                                   -1, -1, -1, -1);
}

VariogramFitResult fit_exponential_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess
) {
    return fit_exponential_variogram(lag_centers, gamma_values, {},
                                     sill_guess, range_guess, nugget_guess,
                                     -1, -1, -1, -1);
}

VariogramFitResult fit_gaussian_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess
) {
    return fit_gaussian_variogram(lag_centers, gamma_values, {},
                                  sill_guess, range_guess, nugget_guess,
                                  -1, -1, -1, -1);
}

VariogramFitResult auto_fit_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget
) {
    std::vector<std::string> models = {"spherical", "exponential", "gaussian"};
    VariogramFitResult best;
    best.rss = std::numeric_limits<double>::max();
    for (const auto& m : models) {
        VariogramFitResult res = fit_variogram_model_impl(
            lag_centers, gamma_values, weights, m,
            sill_guess, range_guess, nugget_guess,
            min_range, max_range, min_nugget, max_nugget);
        if (res.rss < best.rss) best = res;
    }
    return best;
}

VariogramFitResult auto_fit_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess
) {
    return auto_fit_variogram(lag_centers, gamma_values, {},
                              sill_guess, range_guess, nugget_guess,
                              -1, -1, -1, -1);
}

// =============================================================================
// Nested variogram fitting (single-structure fallback)
// =============================================================================
static double evaluate_nested_variogram_model(
    const NestedVariogram& vg,
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_exp,
    const std::vector<double>& weights
) {
    double rss = 0.0;
    for (size_t i = 0; i < lag_centers.size(); ++i) {
        double gamma_th = vg.sill() - vg.covariance(lag_centers[i]);
        double diff = gamma_exp[i] - gamma_th;
        double w = (i < weights.size()) ? weights[i] : 1.0;
        rss += w * diff * diff;
    }
    return rss;
}

static NestedVariogram fit_single_structure(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_exp,
    const std::vector<double>& weights,
    VariogramModelType model_type,
    double sill_guess,
    double range_guess,
    double nugget_guess
) {
    auto fit_res = fit_variogram_model_impl(
        lag_centers, gamma_exp, weights,
        (model_type == VariogramModelType::Spherical) ? "spherical" :
        (model_type == VariogramModelType::Exponential) ? "exponential" : "gaussian",
        sill_guess, range_guess, nugget_guess,
        -1, -1, -1, -1);

    NestedVariogram vg(fit_res.nugget);
    VariogramStructure s;
    s.type = model_type;
    s.range = fit_res.range * (model_type == VariogramModelType::Exponential ? 3.0 : (model_type == VariogramModelType::Gaussian ? std::sqrt(3.0) : 1.0));
    s.contribution = fit_res.sill - fit_res.nugget;
    if (s.contribution > 0) vg.add_structure(s);
    return vg;
}

NestedVariogram fit_nested_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_exp,
    const std::vector<double>& weights,
    int max_structures,
    double sill_guess,
    double range_guess
) {
    if(lag_centers.empty()||lag_centers.size()!=gamma_exp.size()||(!weights.empty()&&weights.size()!=lag_centers.size()))throw std::invalid_argument("invalid variogram arrays");
    for(double v:lag_centers)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid lag");
    for(double v:gamma_exp)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid semivariance");
    for(double v:weights)if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid fit weight");

    if (lag_centers.size() != gamma_exp.size()) {
        throw std::runtime_error("fit_nested_variogram: lag_centers and gamma_exp size mismatch");
    }
    if (max_structures != 1) throw std::invalid_argument("The recovered fitter supports one structure only; construct NestedVariogram explicitly for multiple structures");

    double max_lag = lag_centers.back();
    if (range_guess <= 0) range_guess = max_lag * 0.8;
    if (sill_guess <= 0) {
        int n_last = std::max(1, static_cast<int>(gamma_exp.size()) / 4);
        double sum = 0.0;
        for (size_t i = gamma_exp.size() - n_last; i < gamma_exp.size(); ++i) sum += gamma_exp[i];
        sill_guess = sum / n_last;
    }

    NestedVariogram best;
    double best_rss = std::numeric_limits<double>::max();

    for (auto type : {VariogramModelType::Spherical,
                      VariogramModelType::Exponential,
                      VariogramModelType::Gaussian}) {
        NestedVariogram vg = fit_single_structure(
            lag_centers, gamma_exp, weights, type, sill_guess, range_guess, 0.0);
        double rss = evaluate_nested_variogram_model(vg, lag_centers, gamma_exp, weights);
        if (rss < best_rss) { best_rss = rss; best = vg; }
    }
    return best;
}

// =============================================================================
// Cross-validation
// =============================================================================
CrossValidationResult cross_validate_ordinary_kriging(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    int max_neighbors,
    int min_neighbors,
    double search_radius,
    double sill,
    double nugget
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    CrossValidationResult res;
    const size_t n = sample_xyz.size();
    if (n == 0) return res;
    res.errors.resize(n);
    res.squared_errors.resize(n);

    double sum_err = 0.0, sum_sq = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        std::vector<Point3> train_xyz;
        std::vector<double> train_vals;
        train_xyz.reserve(n-1);
        train_vals.reserve(n-1);
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            train_xyz.push_back(sample_xyz[j]);
            train_vals.push_back(sample_values[j]);
        }
        OKResult ok = ordinary_kriging(train_xyz, train_vals, {sample_xyz[i]},
                                       max_neighbors, min_neighbors,
                                       search_radius, sill, nugget);
        double pred = ok.estimate[0];
        double actual = sample_values[i];
        double err = actual - pred;
        res.errors[i] = err;
        res.squared_errors[i] = err * err;
        sum_err += err;
        sum_sq += err * err;
        sum_abs += std::abs(err);
    }
    res.mean_error = sum_err / static_cast<double>(n);
    res.mean_squared_error = sum_sq / static_cast<double>(n);
    res.root_mean_squared_error = std::sqrt(res.mean_squared_error);
    res.mean_absolute_error = sum_abs / static_cast<double>(n);
    return res;
}

CrossValidationResult cross_validate_indicator_kriging(
    const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<int>& categories,
    int max_neighbors,
    int min_neighbors,
    double search_radius,
    double sill,
    double nugget
) {
    if(std::unordered_set<int>(categories.begin(),categories.end()).size()!=categories.size())throw std::invalid_argument("Duplicate category catalogue");
    for(int c:sample_categories)if(std::find(categories.begin(),categories.end(),c)==categories.end())throw std::invalid_argument("Unknown validation category");
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_categories.size())throw std::invalid_argument("sample/value size mismatch");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");
    if(categories.empty())throw std::invalid_argument("categories must not be empty");

    CrossValidationResult res;
    const size_t n = sample_xyz.size();
    if (n == 0) return res;
    res.errors.resize(n);
    res.squared_errors.resize(n);

    double sum_err = 0.0, sum_sq = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        std::vector<Point3> train_xyz;
        std::vector<int> train_cats;
        train_xyz.reserve(n-1);
        train_cats.reserve(n-1);
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            train_xyz.push_back(sample_xyz[j]);
            train_cats.push_back(sample_categories[j]);
        }
        IKResult ik = indicator_kriging(train_xyz, train_cats, {sample_xyz[i]},
                                        categories, max_neighbors, min_neighbors,
                                        search_radius, sill, nugget);
        int true_cat = sample_categories[i];
        size_t cat_idx = 0;
        for (; cat_idx < categories.size(); ++cat_idx)
            if (categories[cat_idx] == true_cat) break;
        double prob = ik.probabilities[cat_idx];
        double err = 1.0 - prob;
        res.errors[i] = err;
        res.squared_errors[i] = err * err;
        sum_err += err;
        sum_sq += err * err;
        sum_abs += std::abs(err);
    }
    res.mean_error = sum_err / static_cast<double>(n);
    res.mean_squared_error = sum_sq / static_cast<double>(n);
    res.root_mean_squared_error = std::sqrt(res.mean_squared_error);
    res.mean_absolute_error = sum_abs / static_cast<double>(n);
    return res;
}

// =============================================================================
// Sequential Gaussian Simulation
// =============================================================================
SGSResult run_sgs(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget, int seed
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(n_realizations<=0)throw std::invalid_argument("realization count must be positive");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    return run_sgs_with_range(sample_xyz, sample_values, target_xyz,
                              n_realizations, max_neighbors, min_neighbors,
                              search_radius, search_radius, sill, nugget, seed);
}

SGSResult run_sgs_with_range(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double variogram_range,
    double sill, double nugget, int seed, bool standardize
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(n_realizations<=0)throw std::invalid_argument("realization count must be positive");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    if (sample_xyz.size() != sample_values.size())
        throw std::runtime_error("run_sgs_with_range: size mismatch");

    if(!std::isfinite(variogram_range)||variogram_range<=0)throw std::invalid_argument("variogram range must be positive");
    const auto hard_values=conditioning_map(sample_xyz,sample_values);
    SGSResult out;
    out.n_targets = target_xyz.size();
    out.n_realizations = static_cast<size_t>(std::max(n_realizations, 0));
    output_budget(out.n_targets,out.n_realizations);
    out.simulations.assign(out.n_targets * out.n_realizations,
                           std::numeric_limits<double>::quiet_NaN());
    if (sample_xyz.empty() || target_xyz.empty() || n_realizations <= 0) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);
    double range = std::max(variogram_range, 1e-9);
    double search = std::max(search_radius, 1e-9);
    double s = std::max(sill, 1e-12);
    double n = std::max(nugget, 0.0);

    double global_mean = std::accumulate(sample_values.begin(), sample_values.end(), 0.0) / sample_values.size();
    double global_var = 0.0;
    for (double v : sample_values) global_var += (v - global_mean) * (v - global_mean);
    global_var /= std::max<size_t>(sample_values.size(), 1);
    double global_std = std::sqrt(std::max(global_var, 1e-10));

    if(!standardize){global_mean=0.;global_std=1.;}

    std::vector<double> normal_values(sample_values.size());
    for (size_t i = 0; i < sample_values.size(); ++i)
        normal_values[i] = (sample_values[i] - global_mean) / global_std;

    double cell_size = std::max(1.0, search / 4.0);

    for (int sim = 0; sim < n_realizations; ++sim) {
        auto known=hard_values;
        std::mt19937 rng(static_cast<std::uint32_t>(seed) + static_cast<std::uint32_t>(sim) * 1298731u);

        std::vector<int> path(target_xyz.size());
        std::iota(path.begin(), path.end(), 0);
        std::shuffle(path.begin(), path.end(), rng);

        SpatialHash hash(cell_size);
        for (size_t i = 0; i < sample_xyz.size(); ++i)
            hash.add(sample_xyz[i], normal_values[i]);

        for (int idx : path) {
            const Point3& target = target_xyz[static_cast<std::size_t>(idx)];
            if(auto it=known.find(target);it!=known.end()) {
                out.simulations[static_cast<size_t>(sim)*out.n_targets+idx]=it->second;
                continue;
            }
            auto matches = hash.query(target, search, max_neighbors);

            std::sort(matches.begin(), matches.end(),
                      [](const auto& a, const auto& b) { return a.distance < b.distance; });

            double sim_val_norm;
            if (static_cast<int>(matches.size()) < min_neighbors) {
                std::normal_distribution<double> nd(0.0, std::sqrt(s));
                sim_val_norm = nd(rng);
            } else {
                std::vector<Point3> cond_pts; cond_pts.reserve(matches.size());
                std::vector<double> cond_vals; cond_vals.reserve(matches.size());
                for (const auto& m : matches) {
                    cond_pts.push_back(hash.points[static_cast<size_t>(m.index)]);
                    cond_vals.push_back(hash.values[static_cast<size_t>(m.index)]);
                }
                double est, var;
                if (local_ordinary_kriging(target, cond_pts, cond_vals, range, s, n, est, var)) {
                    std::normal_distribution<double> nd(est, std::sqrt(std::max(var, 1e-10)));
                    sim_val_norm = nd(rng);
                } else {
                    std::normal_distribution<double> nd(0.0, std::sqrt(s));
                    sim_val_norm = nd(rng);
                }
            }
            double sim_val_orig = sim_val_norm * global_std + global_mean;
            out.simulations[static_cast<size_t>(sim) * out.n_targets + idx] = sim_val_orig;
            known.emplace(target,sim_val_orig);
            hash.add(target, sim_val_norm);
        }
    }
    return out;
}

SGSSummaryResult run_sgs_summary(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget,
    double cutoff, int seed
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_values.size())throw std::invalid_argument("sample/value size mismatch");
    for(double v:sample_values)if(!std::isfinite(v))throw std::invalid_argument("non-finite sample value");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(n_realizations<=0)throw std::invalid_argument("realization count must be positive");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");

    SGSResult sims = run_sgs_with_range(sample_xyz, sample_values, target_xyz,
                                        n_realizations, max_neighbors, min_neighbors,
                                        search_radius, search_radius,
                                        sill, nugget, seed);

    SGSSummaryResult sum;
    sum.n_targets = static_cast<int>(target_xyz.size());
    sum.n_realizations = n_realizations;
    sum.mean.assign(sum.n_targets, 0.0);
    sum.stddev.assign(sum.n_targets, 0.0);
    sum.p10.assign(sum.n_targets, 0.0);
    sum.p50.assign(sum.n_targets, 0.0);
    sum.p90.assign(sum.n_targets, 0.0);
    sum.prob_above_cutoff.assign(sum.n_targets, 0.0);
    sum.n_samples.assign(sum.n_targets, 0);

    for (size_t i = 0; i < static_cast<size_t>(sum.n_targets); ++i) {
        std::vector<double> vals; vals.reserve(n_realizations);
        for (int sim = 0; sim < n_realizations; ++sim)
            vals.push_back(sims.simulations[static_cast<size_t>(sim) * sum.n_targets + i]);

        double m = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        sum.mean[i] = m;
        double ss = 0.0;
        int above = 0;
        for (double v : vals) {
            ss += (v-m)*(v-m);
            if (v >= cutoff) ++above;
        }
        sum.stddev[i] = std::sqrt(ss / vals.size());
        sum.prob_above_cutoff[i] = static_cast<double>(above) / vals.size();

        std::sort(vals.begin(), vals.end());
        auto percentile = [&](double p) {
            double pos = p * (vals.size()-1);
            size_t lo = static_cast<size_t>(std::floor(pos));
            size_t hi = static_cast<size_t>(std::ceil(pos));
            double t = pos - lo;
            return vals[lo] * (1.0-t) + vals[hi] * t;
        };
        sum.p10[i] = percentile(0.10);
        sum.p50[i] = percentile(0.50);
        sum.p90[i] = percentile(0.90);
    }
    return sum;
}

// =============================================================================
// Sequential Indicator Simulation
// =============================================================================
SISResult run_sis(
    const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget, int seed
) {
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_categories.size())throw std::invalid_argument("sample/value size mismatch");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(n_realizations<=0)throw std::invalid_argument("realization count must be positive");
    if(!std::isfinite(sill)||sill<=0||!std::isfinite(nugget)||nugget<0||nugget>sill)throw std::invalid_argument("invalid sill/nugget");
    if(categories.empty())throw std::invalid_argument("categories must not be empty");

    if (sample_xyz.size() != sample_categories.size())
        throw std::runtime_error("run_sis: size mismatch");

    const auto hard_values=conditioning_map(sample_xyz,sample_categories);
    if(std::unordered_set<int>(categories.begin(),categories.end()).size()!=categories.size())throw std::invalid_argument("duplicate category labels");
    for(int c:sample_categories)if(std::find(categories.begin(),categories.end(),c)==categories.end())throw std::invalid_argument("unknown sample category");
    SISResult out;
    out.n_targets = target_xyz.size();
    out.n_realizations = static_cast<size_t>(std::max(n_realizations, 0));
    output_budget(out.n_targets,out.n_realizations);
    out.simulations.assign(out.n_targets * out.n_realizations, MISSING_CATEGORY);
    if (sample_xyz.empty() || target_xyz.empty() || n_realizations <= 0) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);

    auto global_props = compute_global_category_proportions(sample_categories, categories);

    for (int sim = 0; sim < n_realizations; ++sim) {
        auto known=hard_values;
        std::mt19937 rng(static_cast<std::uint32_t>(seed) + static_cast<std::uint32_t>(sim) * 1298731u);

        std::vector<int> path(target_xyz.size());
        std::iota(path.begin(), path.end(), 0);
        std::shuffle(path.begin(), path.end(), rng);

        std::vector<Point3> current_pts = sample_xyz;
        std::vector<int> current_cats = sample_categories;

        for (int idx : path) {
            const Point3& target = target_xyz[static_cast<std::size_t>(idx)];
            if(auto it=known.find(target);it!=known.end()){out.simulations[static_cast<size_t>(sim)*out.n_targets+idx]=it->second;continue;}

            std::vector<std::pair<size_t, double>> matches;
            for (size_t j = 0; j < current_pts.size(); ++j) {
                double d = euclidean_dist(target, current_pts[j]);
                if (d <= search_radius) matches.emplace_back(j, d);
            }
            std::sort(matches.begin(), matches.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

            std::vector<double> probs = global_props;

            if (static_cast<int>(matches.size()) >= min_neighbors) {
                if (static_cast<int>(matches.size()) > max_neighbors)
                    matches.resize(max_neighbors);

                std::vector<Point3> cond_pts; cond_pts.reserve(matches.size());
                std::vector<int> cond_cats; cond_cats.reserve(matches.size());
                for (const auto& m : matches) {
                    cond_pts.push_back(current_pts[m.first]);
                    cond_cats.push_back(current_cats[m.first]);
                }

                for (size_t c = 0; c < categories.size(); ++c) {
                    std::vector<double> indicators(matches.size());
                    for (size_t k = 0; k < matches.size(); ++k)
                        indicators[k] = (cond_cats[k] == categories[c]) ? 1.0 : 0.0;
                    double prob, var;
                    if (local_indicator_kriging(target, cond_pts, indicators,
                                                global_props[c], search_radius,
                                                sill, nugget, prob, var)) {
                        probs[c] = prob;
                    }
                }
                for (double& p : probs) p = std::max(p, 0.0);
                double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
                if (sum > 1e-12) {
                    for (double& p : probs) p /= sum;
                } else {
                    std::fill(probs.begin(), probs.end(), 1.0 / categories.size());
                }
            }
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int drawn = categories[dist(rng)];
            out.simulations[static_cast<size_t>(sim) * out.n_targets + idx] = drawn;
            known.emplace(target,drawn);
            current_pts.push_back(target);
            current_cats.push_back(drawn);
        }
    }
    return out;
}

SISResult run_sis_anisotropic(
    const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    const std::vector<IndicatorVariogram>& variograms,
    int n_realizations,
    int max_neighbors,
    int min_neighbors,
    double search_radius_aniso,
    int seed
) {
    for(const auto& v:variograms)indicator_spec(v);
    for(const auto& p:sample_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_xyz)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_xyz.size()!=sample_categories.size())throw std::invalid_argument("sample/value size mismatch");
    if(!std::isfinite(search_radius_aniso)||search_radius_aniso<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(n_realizations<=0)throw std::invalid_argument("realization count must be positive");
    if(categories.empty())throw std::invalid_argument("categories must not be empty");

    if (sample_xyz.size() != sample_categories.size())
        throw std::runtime_error("run_sis_anisotropic: size mismatch");

    const auto hard_values=conditioning_map(sample_xyz,sample_categories);
    if(std::unordered_set<int>(categories.begin(),categories.end()).size()!=categories.size())throw std::invalid_argument("duplicate category labels");
    for(int c:sample_categories)if(std::find(categories.begin(),categories.end(),c)==categories.end())throw std::invalid_argument("unknown sample category");
    SISResult out;
    out.n_targets = target_xyz.size();
    out.n_realizations = static_cast<size_t>(std::max(n_realizations, 0));
    output_budget(out.n_targets,out.n_realizations);
    out.simulations.assign(out.n_targets * out.n_realizations, MISSING_CATEGORY);
    if (sample_xyz.empty() || target_xyz.empty() || n_realizations <= 0) return out;

    max_neighbors = std::clamp(max_neighbors, 1, MAX_LOCAL_NEIGHBORS);
    min_neighbors = std::max(min_neighbors, 1);

    std::vector<double> global_props(categories.size(), 0.0);
    for (size_t c = 0; c < categories.size(); ++c)
        global_props[c] = std::count(sample_categories.begin(),
                                     sample_categories.end(),
                                     categories[c]) /
                          static_cast<double>(sample_categories.size());

    std::unordered_map<int, IndicatorVariogram> vg_map;
    std::unordered_map<int, std::vector<Point3>> trans_sample;
    for (int cat : categories) {
        IndicatorVariogram vg;
        auto it = std::find_if(variograms.begin(), variograms.end(),
                               [cat](const IndicatorVariogram& v) { return v.category == cat; });
        if (it != variograms.end()) vg = *it;
        else {
            vg.category = cat;
            vg.range_major = vg.range_minor = vg.range_vertical = 100.0;
            vg.sill = 1.0; vg.nugget = 0.05;
            vg.azimuth_deg = vg.dip_deg = vg.plunge_deg = 0.0;
        }
        vg_map[cat] = vg;

        std::vector<Point3> trans(sample_xyz.size());
        for (size_t i = 0; i < sample_xyz.size(); ++i)
            trans[i] = transform_point_anisotropic(sample_xyz[i], vg);
        trans_sample[cat] = std::move(trans);
    }

    double hash_cell_size = std::max(search_radius_aniso / 4.0, 1.0);

    std::unordered_map<int, std::vector<Point3>> trans_target;
    for (int cat : categories) {
        const auto& vg = vg_map[cat];
        std::vector<Point3> tt(target_xyz.size());
        for (size_t i = 0; i < target_xyz.size(); ++i)
            tt[i] = transform_point_anisotropic(target_xyz[i], vg);
        trans_target[cat] = std::move(tt);
    }

SpatialHash preflight(hash_cell_size);
    for(const auto& item:trans_sample)for(const auto& p:item.second)preflight.validate_target(p);
    for(const auto& item:trans_target)for(const auto& p:item.second)preflight.validate_target(p);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int sim = 0; sim < n_realizations; ++sim) {
        auto known=hard_values;
        std::mt19937 rng(static_cast<std::uint32_t>(seed) + static_cast<std::uint32_t>(sim) * 1298731u);

        std::vector<int> path(target_xyz.size());
        std::iota(path.begin(), path.end(), 0);
        std::shuffle(path.begin(), path.end(), rng);

        std::vector<int> sim_vals(target_xyz.size(), MISSING_CATEGORY);

        std::unordered_map<int, std::unique_ptr<SpatialHash>> sim_hashes;
        std::unordered_map<int, std::vector<Point3>> sim_orig_points;
        std::unordered_map<int, std::vector<int>>    sim_cats;

        for (int cat : categories) {
            sim_hashes[cat] = std::make_unique<SpatialHash>(hash_cell_size);
            for (size_t i = 0; i < sample_xyz.size(); ++i) {
                const Point3& trans_pt = trans_sample[cat][i];
                sim_hashes[cat]->add(trans_pt, sample_categories[i] == cat ? 1.0 : 0.0);
                sim_orig_points[cat].push_back(sample_xyz[i]);
                sim_cats[cat].push_back(sample_categories[i]);
            }
        }

        for (int idx : path) {
            const Point3& target = target_xyz[static_cast<std::size_t>(idx)];
            if(auto it=known.find(target);it!=known.end()){sim_vals[idx]=it->second;continue;}
            std::vector<double> probs(categories.size(), 0.0);

            for (size_t cidx = 0; cidx < categories.size(); ++cidx) {
                int cat = categories[cidx];
                const auto& vg = vg_map[cat];
                const Point3& q = trans_target[cat][idx];

                auto& hash_ptr = sim_hashes[cat];
                auto neighbors = hash_ptr->query(q, search_radius_aniso, max_neighbors);

                double prob = global_props[cidx];

                if (static_cast<int>(neighbors.size()) >= min_neighbors) {
                    if (static_cast<int>(neighbors.size()) > max_neighbors) {
                        std::sort(neighbors.begin(), neighbors.end(),
                            [](const auto& a, const auto& b) { return a.distance < b.distance; });
                        neighbors.resize(max_neighbors);
                    }
                    std::vector<Point3> cond_pts(neighbors.size());
                    std::vector<double> cond_vals(neighbors.size());
                    for (size_t k = 0; k < neighbors.size(); ++k) {
                        int local_idx = neighbors[k].index;
                        cond_pts[k] = sim_orig_points[cat][local_idx];
                        cond_vals[k] = (sim_cats[cat][local_idx] == cat) ? 1.0 : 0.0;
                    }
                    double mean = 0.0, var = 0.0;
                    if (local_indicator_kriging_anisotropic(target, cond_pts, cond_vals,
                                                            global_props[cidx], vg, mean, var)) {
                        prob = std::clamp(mean, 0.0, 1.0);
                    }
                }
                probs[cidx] = prob;
            }

            for (double& p : probs) p = std::max(p, 0.0);
            double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
            if (sum > 0.0) {
                for (double& p : probs) p /= sum;
            } else {
                std::fill(probs.begin(), probs.end(), 1.0 / categories.size());
            }
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int drawn_cat = categories[dist(rng)];
            sim_vals[idx] = drawn_cat;
            known.emplace(target,drawn_cat);

            for (int cat : categories) {
                Point3 trans_pt = transform_point_anisotropic(target, vg_map[cat]);
                sim_hashes[cat]->add(trans_pt, drawn_cat == cat ? 1.0 : 0.0);
                sim_orig_points[cat].push_back(target);
                sim_cats[cat].push_back(drawn_cat);
            }
        }

        for (size_t i = 0; i < target_xyz.size(); ++i)
            out.simulations[static_cast<size_t>(sim) * out.n_targets + i] = sim_vals[i];
    }
    return out;
}

// =============================================================================
// Topological annealing stub
// =============================================================================
std::vector<int> run_topological_annealing(
    const std::vector<Point3>&, const std::vector<int>&, double,
    const std::unordered_set<int>&, const std::unordered_map<int, std::vector<int>>&,
    const std::vector<bool>&, int, int, double, int) {
    throw std::runtime_error("Topological annealing is not implemented in the supplied archive");
}

} // namespace deeprbf
