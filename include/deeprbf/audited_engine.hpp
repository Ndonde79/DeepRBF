#pragma once
#include "deeprbf/core.hpp"
#include <array>
#include <string>
#include <memory>
#include <vector>

namespace deeprbf {
using Axes3 = std::array<Point3,3>;
inline Axes3 identity_axes() { return {{{1,0,0},{0,1,0},{0,0,1}}}; }

// Rows are local axes in world coordinates. Exponential/Gaussian ranges use
// range_convention; spherical ranges are compact-support distances.
struct CovarianceSpec {
    std::string family="spherical";
    // length_scale preserves the historical API; practical uses exp(-3h)
    // or exp(-3h^2). Spherical support is unchanged in either convention.
    std::string range_convention="length_scale";
    Point3 ranges{100,100,100};
    Axes3 axes=identity_axes();
    double sill=1, nugget=0;
    void validate() const;
    double covariance(const Point3& a,const Point3& b,bool same_observation=false) const;
};
struct NeighborhoodSpec {
    Point3 radii{300,300,300};
    Axes3 axes=identity_axes();
    int max_neighbors=24,min_neighbors=3,min_holes=3,max_per_hole=3;
    int max_per_octant=0,candidate_neighbors=128;
    bool restrict_domain=false;
    void validate() const;
};
struct SolveDiagnostics {
    std::string status="NOT_RUN";
    int rank=0,dimension=0,n_samples=0,n_holes=0;
    double relative_residual=0,reciprocal_condition=0,regularization=0;
    // Defined only for successful solves; negative weights are not clipped.
    double minimum_weight=0,maximum_weight=0,weight_sum=0,weight_l1=0;
    int negative_weights=0;
    bool success=false;
};
struct AuditedResult {
    std::vector<double> estimate,variance;
    std::vector<SolveDiagnostics> diagnostics;
};
struct RotatedSupport {
    Point3 center{0,0,0},size{0,0,0};
    Axes3 axes=identity_axes();
    std::array<int,3> discretization{1,1,1};
    int domain=0;
};
struct NeighborhoodSelection {
    std::vector<int> indices;
    int n_holes=0;
    bool supported=false;
};
NeighborhoodSelection select_neighborhood(const std::vector<Point3>& xyz,
    const std::vector<int>& holes,const std::vector<int>& domains,
    const Point3& target,int target_domain,const NeighborhoodSpec& search);

AuditedResult audited_ordinary_kriging(const std::vector<Point3>& xyz,
    const std::vector<double>& values,const std::vector<int>& holes,
    const std::vector<int>& domains,const std::vector<RotatedSupport>& targets,
    const CovarianceSpec& covariance,const NeighborhoodSpec& search,
    double regularization=0,double residual_tolerance=1e-9,
    const std::vector<RotatedSupport>& observation_support={});

struct NativeCategoricalResult {
    std::vector<double> probabilities;
    std::vector<int> support_holes;
    std::vector<int> supported,exact_conditioned;
    int n_targets=0,n_categories=0;
};
// Sparse local simple indicator kriging / sequential indicator simulation.
// Coordinates are already rotated and divided by practical covariance ranges.
// hard contains encoded categories 0..C-1, or -1 for unknown nodes.
struct NativeIndicatorResult {
    std::vector<double> probabilities;
    std::vector<int> simulations,direct_hard_neighbors,direct_hard_holes,supported,prior_only_draws;
    int n_nodes=0,n_categories=0,n_realizations=0;
    double maximum_solve_residual=0;
};
NativeIndicatorResult indicator_lithology_native(const std::vector<Point3>& coordinates,
    const std::vector<int>& hard,const std::vector<double>& prior,int max_neighbors=20,
    int realizations=0,unsigned long long seed=42,int candidate_neighbors=96,
    double search_radius=2.,const std::vector<int>& hole_ids={},int max_per_hole=0,int min_holes=0);
std::vector<double> gibbs_truncated_native(const std::vector<double>& precision,
    const std::vector<double>& lower,const std::vector<double>& upper,
    int sweeps=30,unsigned long long seed=42);

struct SparseLatentResult {
    std::vector<double> values;
    long long precision_nonzeros=0,estimated_workspace_bytes=0;
    double maximum_local_residual=0;
};
SparseLatentResult vecchia_truncated_native(const std::vector<Point3>& coordinates,
    const std::vector<double>& lower,const std::vector<double>& upper,int neighbors=12,
    int sweeps=30,unsigned long long seed=42,double nugget=.01,double max_workspace_mb=512);
NativeCategoricalResult categorical_rbf_native(const std::vector<Point3>& xyz,
    const std::vector<int>& labels,const std::vector<int>& holes,
    const std::vector<double>& support,const std::vector<int>& categories,
    const std::vector<int>& category_domains,const std::vector<Point3>& targets,
    const Point3& ranges,const Axes3& axes,const Point3& gate_scale,
    const NeighborhoodSpec& search,bool hierarchical=false,double prior_mix=0);

// Owns an immutable sample snapshot and reuses both search indexes across chunks.
class NativeCategoricalModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    NativeCategoricalModel(const std::vector<Point3>& xyz,const std::vector<int>& labels,
        const std::vector<int>& holes,const std::vector<double>& support,
        const std::vector<int>& categories,const std::vector<int>& category_domains,
        const Point3& ranges,const Axes3& axes,const Point3& gate_scale,const NeighborhoodSpec& search);
    ~NativeCategoricalModel();
    NativeCategoricalResult predict(const std::vector<Point3>& targets,bool hierarchical=false,double prior_mix=0)const;
};

struct NativeGaussianResult {
    std::vector<double> simulations,conditional_mean,conditional_covariance;
    int n_targets=0,n_realizations=0;
    double minimum_conditional_eigenvalue=0,conditioning_residual=0;
};
// Exact dense Gaussian-space reference, with explicit bounded problem size.
// It is a validation/reference backend, not an unbounded large-grid simulator.
NativeGaussianResult simulate_gaussian_native(const std::vector<Point3>& xyz,
    const std::vector<double>& values,const std::vector<Point3>& targets,
    const CovarianceSpec& covariance,int realizations,unsigned long long seed,
    double mean=0,int dense_limit=2048);
} // namespace deeprbf
