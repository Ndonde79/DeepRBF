#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "deeprbf/spatial_hash.hpp"
#include "deeprbf/nested_variogram.hpp"

namespace deeprbf {

// ----------------------------------------------------------------------
// Struct definitions (declaration only)
// ----------------------------------------------------------------------
struct OKResult {
    std::vector<double> estimate, variance;
    std::vector<int> n_samples;
};

struct IKResult {
    std::vector<double> probabilities;
    std::vector<int> categories, n_samples;
    size_t n_targets = 0, n_categories = 0;
};

struct SGSResult {
    std::vector<double> simulations;
    size_t n_targets = 0, n_realizations = 0;
};

struct SGSSummaryResult {
    std::vector<double> mean, stddev, p10, p50, p90, prob_above_cutoff;
    std::vector<int> n_samples;
    int n_targets = 0, n_realizations = 0;
};

struct SISResult {
    std::vector<int> simulations;
    size_t n_targets = 0, n_realizations = 0;
};

struct IndicatorVariogram {
    int category = 0;
    double range_major = 100.0, range_minor = 100.0, range_vertical = 100.0;
    double azimuth_deg = 0.0, dip_deg = 0.0, plunge_deg = 0.0;
    double sill = 1.0, nugget = 0.0;
};

struct CrossVariogram {
    int category_i = 0, category_j = 0;
    double sill = 0.0, nugget = 0.0, range = 100.0;
};

struct LMCStructure {
    int structure_id = 0;
    double range_major = 100.0, range_minor = 100.0, range_vertical = 50.0;
    double azimuth_deg = 0.0, dip_deg = 0.0, plunge_deg = 0.0;
    std::vector<std::vector<double>> covariance_matrix;
    std::string family = "spherical";
};

double lmc_correlation(double scaled_distance,const std::string& family);

class LinearModelOfCoregionalization {
public:
    void add_structure(const LMCStructure& s);
    double covariance(int cat_i, int cat_j, double h_scaled) const;
    const std::vector<LMCStructure>& structures() const { return structures_; }
private:
    std::vector<LMCStructure> structures_;
};

struct CoKrigingResult {
    std::vector<double> estimates, variances;
    std::vector<int> n_samples;
    std::vector<std::vector<int>> samples_per_category;
};

struct SICoKResult {
    std::vector<std::vector<int>> realizations;
    size_t n_targets = 0, n_realizations = 0;
};

struct Direction {
    double azimuth_deg = 0.0, dip_deg = 0.0, tolerance_deg = 0.0;
};

struct VariogramLag {
    double lag_center = 0.0, gamma_value = 0.0;
    int n_pairs = 0;
};

struct VariogramResult {
    std::vector<int> categories;
    std::vector<VariogramLag> lags;
};

struct VariogramFitResult {
    double sill = 1.0, range = 100.0, nugget = 0.0;
    double rss = 0.0, r_squared = 0.0;
    std::string model_type = "spherical";
    std::vector<double> fitted_values;
};

struct CrossValidationResult {
    std::vector<double> errors, squared_errors;
    double mean_error = 0.0, mean_squared_error = 0.0;
    double root_mean_squared_error = 0.0, mean_absolute_error = 0.0;
};

// ----------------------------------------------------------------------
// Function declarations (exactly as in gc_geostatistics.cpp)
// ----------------------------------------------------------------------
double spherical_covariance(double h_scaled, double sill, double nugget);
double anisotropic_scaled_distance(const Point3& a, const Point3& b,
    double range_major, double range_minor, double range_vertical,
    double azimuth_deg, double dip_deg, double plunge_deg);

std::vector<double> compute_global_category_proportions(
    const std::vector<int>& sample_categories,
    const std::vector<int>& categories);

std::vector<double> probabilities_from_local_ik_anisotropic(
    const Point3& target,
    const std::vector<Point3>& cond_points,
    const std::vector<int>& cond_categories,
    const std::vector<int>& categories,
    const std::vector<IndicatorVariogram>& variograms,
    const std::vector<double>& fallback);

// Ordinary Kriging
OKResult ordinary_kriging(const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget);

OKResult ordinary_kriging_with_range(const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int max_neighbors, int min_neighbors,
    double search_radius, double variogram_range,
    double sill, double nugget);

OKResult ordinary_kriging_anisotropic(const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    const IndicatorVariogram& variogram,
    int max_neighbors, int min_neighbors,
    double search_radius);

OKResult ordinary_kriging_local(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    double search_radius,
    int max_neighbors = 20,
    int min_neighbors = 5,
    int min_points_for_variogram = 10,
    int n_lags = 5,
    double sill_guess = -1.0,
    double range_guess = -1.0,
    double nugget_guess = -1.0
);

// Indicator Kriging
IKResult indicator_kriging(const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget);

IKResult indicator_kriging_anisotropic(const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    const std::vector<IndicatorVariogram>& variograms,
    int max_neighbors, int min_neighbors,
    double search_radius_aniso);

// Variogram computation
std::vector<VariogramResult> compute_direct_indicator_variograms_fast(
    const std::vector<Point3>& points,
    const std::vector<int>& categories,
    const std::vector<int>& all_categories,
    const std::vector<Direction>& directions,
    int n_lags, double max_lag, double lag_width, int max_pairs_per_lag);

std::vector<VariogramResult> compute_cross_indicator_variograms_fast(
    const std::vector<Point3>& points,
    const std::vector<int>& categories,
    const std::vector<std::pair<int,int>>& cross_pairs,
    const std::vector<Direction>& directions,
    int n_lags, double max_lag, double lag_width, int max_pairs_per_lag);

// Variogram fitting (all 8‑parameter versions, plus convenience overloads)
VariogramFitResult fit_spherical_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget);

VariogramFitResult fit_exponential_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget);

VariogramFitResult fit_gaussian_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget);

VariogramFitResult auto_fit_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    const std::vector<double>& weights,
    double sill_guess, double range_guess, double nugget_guess,
    double min_range, double max_range, double min_nugget, double max_nugget);

// Add this after the existing auto_fit_variogram overloads:

/**
 * Automatically fit a nested variogram (1, 2, or 3 structures) using a combination
 * of grid search over ranges and linear least squares for contributions.
 *
 * @param lag_centers     Centre distances of experimental variogram lags.
 * @param gamma_exp       Experimental variogram values.
 * @param weights         Weights for each lag (e.g., number of pairs). Can be empty.
 * @param max_structures  Maximum number of structures to try (1, 2, or 3).
 * @param sill_guess      Optional initial guess for total sill (default = -1 for auto).
 * @param range_guess     Optional initial guess for the longest range (default = -1 for auto).
 * @return NestedVariogram object with fitted nugget and structures.
 */
NestedVariogram fit_nested_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_exp,
    const std::vector<double>& weights,
    int max_structures = 1,
    double sill_guess = -1.0,
    double range_guess = -1.0
);
// Convenience overloads without weights/bounds
VariogramFitResult fit_spherical_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess);

VariogramFitResult fit_exponential_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess);

VariogramFitResult fit_gaussian_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess);

VariogramFitResult auto_fit_variogram(
    const std::vector<double>& lag_centers,
    const std::vector<double>& gamma_values,
    double sill_guess, double range_guess, double nugget_guess);

// Cross-validation
CrossValidationResult cross_validate_ordinary_kriging(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget);

CrossValidationResult cross_validate_indicator_kriging(
    const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<int>& categories,
    int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget);

// Sequential Gaussian Simulation (11 arguments, including seed)
SGSResult run_sgs(const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget, int seed);

SGSResult run_sgs_with_range(const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double variogram_range,
    double sill, double nugget, int seed, bool standardize = true);

SGSSummaryResult run_sgs_summary(const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget,
    double cutoff, int seed);

// Sequential Indicator Simulation
SISResult run_sis(const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, double sill, double nugget, int seed);

SISResult run_sis_anisotropic(const std::vector<Point3>& sample_xyz,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_xyz,
    const std::vector<int>& categories,
    const std::vector<IndicatorVariogram>& variograms,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius_aniso, int seed);

// Cokriging (implemented in gc_cokriging.cpp)
LinearModelOfCoregionalization build_lmc_from_variograms(
    const std::vector<CrossVariogram>& variograms, int n_structures);

CoKrigingResult ordinary_cokriging(
    const std::vector<Point3>& sample_points,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_points,
    const std::vector<int>& categories,
    const LinearModelOfCoregionalization& lmc,
    int max_neighbors, int min_neighbors, double search_radius);

SICoKResult sequential_indicator_cokriging(
    const std::vector<Point3>& sample_points,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_points,
    const std::vector<int>& categories,
    const LinearModelOfCoregionalization& lmc,
    int n_realizations, int max_neighbors, int min_neighbors,
    double search_radius, int seed);

bool is_positive_definite(const LinearModelOfCoregionalization& lmc);

// Topological annealing (stub)
std::vector<int> run_topological_annealing(
    const std::vector<Point3>& block_coords,
    const std::vector<int>& initial_realization,
    double search_radius,
    const std::unordered_set<int>& ore_domains,
    const std::unordered_map<int, std::vector<int>>& domain_groups,
    const std::vector<bool>& fixed_mask,
    int target_components,
    int iterations, double initial_temp, int seed);
	
// Replace the old local kriging with a nested version
bool local_ordinary_kriging_nested(
    const Point3& target,
    const std::vector<Point3>& cond_pts,
    const std::vector<double>& cond_vals,
    const NestedVariogram& variogram,
    double& estimate,
    double& variance
);

// Public API
OKResult ordinary_kriging_nested(
    const std::vector<Point3>& sample_xyz,
    const std::vector<double>& sample_values,
    const std::vector<Point3>& target_xyz,
    const NestedVariogram& variogram,
    int max_neighbors,
    int min_neighbors,
    double search_radius
);


} // namespace deeprbf
