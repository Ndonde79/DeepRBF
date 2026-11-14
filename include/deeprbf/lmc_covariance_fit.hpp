#pragma once
#include "deeprbf/gc_geostatistics.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>
namespace deeprbf {
struct LMCCovarianceObservation {
 double lag=0.0;
 Eigen::MatrixXd covariance;
 double weight=1.0;
};
struct LMCCovarianceFit {
 LinearModelOfCoregionalization model;
 double weighted_rss=0.0;
 int iterations=0;
 bool converged=false;
};
// Fit jointly to observed covariance matrices. The zero-lag covariance must
// be supplied or estimated independently; cross-semivariances alone cannot
// identify zero-lag cross-covariance.
LMCCovarianceFit fit_lmc_to_covariances(
 const std::vector<LMCCovarianceObservation>& observations,
 const std::vector<double>& ranges,
 const std::string& model_type="spherical",
 double tolerance=1e-9, int max_iterations=500);
}
