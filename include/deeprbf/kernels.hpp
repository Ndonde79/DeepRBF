#pragma once

#include "deeprbf/core.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace deeprbf {

inline double point_distance(const Point3& a, const Point3& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::hypot(dx, dy, dz);
}

inline double euclidean_dist(const Point3& a, const Point3& b) {
    return point_distance(a, b);
}

// Generalised Cauchy covariance (unit sill), the long-tailed "spheroidal"-style kernel:
//   C(h) = (1 + (h/b)^p)^(-a/p),   b = epsilon / (0.04^(-p/a) - 1)^(1/p)
// so epsilon is the range at which C = 0.04 (variogram at 96% of sill, the Leapfrog spheroidal convention).
// Defaults a = 9, p = 1.667 are a fit to the Leapfrog spheroidal alpha = 9 curve. Positive definite in 3D for
// a > 0 and 0 < p <= 2. Shape is chosen through the kernel string: "gen_cauchy" (defaults) or
// "gen_cauchy:<alpha>:<power>", e.g. "gen_cauchy:9:1.667". The string is parsed once (resolve_kernel); inner
// loops use ResolvedKernel and never compare strings.
constexpr double GEN_CAUCHY_DEFAULT_ALPHA = 9.0;
constexpr double GEN_CAUCHY_DEFAULT_POWER = 1.667;
constexpr double GEN_CAUCHY_RANGE_COVARIANCE = 0.04;

inline bool is_gen_cauchy_name(const std::string& kernel) {
    return kernel=="gen_cauchy" || kernel.rfind("gen_cauchy:",0)==0;
}

// Parses "gen_cauchy" / "gen_cauchy:a:p"; throws std::invalid_argument on malformed or invalid shapes.
inline void parse_gen_cauchy(const std::string& kernel, double& alpha, double& power) {
    alpha=GEN_CAUCHY_DEFAULT_ALPHA; power=GEN_CAUCHY_DEFAULT_POWER;
    if(kernel=="gen_cauchy") return;
    const std::string rest=kernel.substr(11);
    const std::size_t colon=rest.find(':');
    if(colon==std::string::npos) throw std::invalid_argument("gen_cauchy kernel string must be gen_cauchy or gen_cauchy:<alpha>:<power>, got "+kernel);
    const std::string sa=rest.substr(0,colon), sp=rest.substr(colon+1);
    char* end=nullptr;
    alpha=std::strtod(sa.c_str(),&end);
    if(sa.empty()||*end!='\0') throw std::invalid_argument("gen_cauchy: cannot parse alpha in "+kernel);
    power=std::strtod(sp.c_str(),&end);
    if(sp.empty()||*end!='\0') throw std::invalid_argument("gen_cauchy: cannot parse power in "+kernel);
    if(!std::isfinite(alpha)||!(alpha>0.0)) throw std::invalid_argument("gen_cauchy: alpha must be > 0");
    if(!std::isfinite(power)||!(power>0.0)||power>2.0) throw std::invalid_argument("gen_cauchy: power must be in (0, 2]");
}

// Scale b such that C(epsilon) = 0.04.
inline double gen_cauchy_scale(double epsilon, double alpha, double power) {
    return epsilon/std::pow(std::pow(GEN_CAUCHY_RANGE_COVARIANCE,-power/alpha)-1.0,1.0/power);
}

inline double gen_cauchy_value(double d, double inv_b, double power, double neg_a_over_p) {
    return std::pow(1.0+std::pow(d*inv_b,power),neg_a_over_p);
}

inline double evaluate_kernel(const std::string& kernel, double d, double epsilon) {
    if(!std::isfinite(d)||d<0||!std::isfinite(epsilon)||epsilon<=0)
        throw std::invalid_argument("kernel requires finite nonnegative distance and positive epsilon");
    const double q=d/epsilon;
    if(kernel=="gaussian") return std::exp(-q*q);
    if(kernel=="multiquadric") return std::hypot(d,epsilon);
    if(kernel=="inverse_multiquadric") return 1.0/std::hypot(d,epsilon);
    if(kernel=="thin_plate"||kernel=="thin_plate_spline") return d==0?0:q*q*std::log(q+1e-12);
    if(kernel=="biharmonic"||kernel=="cubic") return d*d*d;
    if(kernel=="linear") return d;
    if(kernel=="spherical") return q<1.0?1.0-1.5*q+0.5*q*q*q:0.0;
    if(kernel=="exponential") return std::exp(-3.0*q);
    if(is_gen_cauchy_name(kernel)){
        double a,p; parse_gen_cauchy(kernel,a,p);
        return gen_cauchy_value(d,1.0/gen_cauchy_scale(epsilon,a,p),p,-a/p);
    }
    throw std::invalid_argument("unknown RBF kernel: "+kernel);
}

// 3.3.2: kernel resolved once to an id, so inner loops avoid string comparisons. Same formulas as evaluate_kernel.
enum class KernelId { Gaussian, Multiquadric, InverseMultiquadric, ThinPlate, Cubic, Linear, Spherical, Exponential, GenCauchy };

inline KernelId kernel_id(const std::string& kernel) {
    if(kernel=="gaussian") return KernelId::Gaussian;
    if(kernel=="multiquadric") return KernelId::Multiquadric;
    if(kernel=="inverse_multiquadric") return KernelId::InverseMultiquadric;
    if(kernel=="thin_plate"||kernel=="thin_plate_spline") return KernelId::ThinPlate;
    if(kernel=="biharmonic"||kernel=="cubic") return KernelId::Cubic;
    if(kernel=="linear") return KernelId::Linear;
    if(kernel=="spherical") return KernelId::Spherical;
    if(kernel=="exponential") return KernelId::Exponential;
    if(is_gen_cauchy_name(kernel)) return KernelId::GenCauchy;
    throw std::invalid_argument("unknown RBF kernel: "+kernel);
}

// No argument checks: callers validate epsilon once and pass finite distances.
inline double evaluate_kernel_id(KernelId id, double d, double epsilon) {
    switch(id){
    case KernelId::Gaussian: { const double q=d/epsilon; return std::exp(-q*q); }
    case KernelId::Multiquadric: return std::hypot(d,epsilon);
    case KernelId::InverseMultiquadric: return 1.0/std::hypot(d,epsilon);
    case KernelId::ThinPlate: { const double q=d/epsilon; return d==0?0:q*q*std::log(q+1e-12); }
    case KernelId::Cubic: return d*d*d;
    case KernelId::Linear: return d;
    case KernelId::Spherical: { const double q=d/epsilon; return q<1.0?1.0-1.5*q+0.5*q*q*q:0.0; }
    case KernelId::Exponential: return std::exp(-3.0*d/epsilon);
    // The id alone carries no shape: default alpha/power. Use ResolvedKernel for a configured shape.
    case KernelId::GenCauchy:
        return gen_cauchy_value(d,1.0/gen_cauchy_scale(epsilon,GEN_CAUCHY_DEFAULT_ALPHA,GEN_CAUCHY_DEFAULT_POWER),
                                GEN_CAUCHY_DEFAULT_POWER,-GEN_CAUCHY_DEFAULT_ALPHA/GEN_CAUCHY_DEFAULT_POWER);
    }
    return 0.0;
}

// Kernel resolved once (id, epsilon, shape constants) for inner loops. Same formulas as evaluate_kernel.
struct ResolvedKernel {
    KernelId id = KernelId::Gaussian;
    double epsilon = 1.0;
    double gc_inv_b = 1.0, gc_power = 1.0, gc_neg_a_over_p = -1.0;
    double operator()(double d) const {
        if(id==KernelId::GenCauchy) return gen_cauchy_value(d,gc_inv_b,gc_power,gc_neg_a_over_p);
        return evaluate_kernel_id(id,d,epsilon);
    }
};

inline ResolvedKernel resolve_kernel(const std::string& kernel, double epsilon) {
    evaluate_kernel(kernel,0.0,epsilon);   // validates name, shape and epsilon
    ResolvedKernel k; k.id=kernel_id(kernel); k.epsilon=epsilon;
    if(k.id==KernelId::GenCauchy){
        double a,p; parse_gen_cauchy(kernel,a,p);
        k.gc_inv_b=1.0/gen_cauchy_scale(epsilon,a,p); k.gc_power=p; k.gc_neg_a_over_p=-a/p;
    }
    return k;
}

// Strictly positive definite kernels: may be used with no polynomial drift.
inline bool kernel_is_positive_definite(KernelId id) {
    return id==KernelId::Gaussian||id==KernelId::InverseMultiquadric||id==KernelId::Spherical||
           id==KernelId::Exponential||id==KernelId::GenCauchy;
}
// Conditionally positive definite of order 2 (cubic, thin plate): need at least the linear drift.
inline bool kernel_needs_linear_drift(KernelId id) {
    return id==KernelId::Cubic||id==KernelId::ThinPlate;
}

// Sign of the regularisation on the diagonal of the RBF matrix. A kernel that is conditionally positive definite of
// order 1 with a positive sign (linear r, multiquadric) is conditionally NEGATIVE definite on the drift-orthogonal
// subspace, so smoothing needs -nu there; +nu would move the spectrum towards zero (3.4.2 fix). Positive definite and
// order-2 kernels (cubic r^3, thin plate) take +nu.
inline double regularization_sign(KernelId id) {
    return (id==KernelId::Linear||id==KernelId::Multiquadric) ? -1.0 : 1.0;
}

inline Matrix build_phi_matrix(const std::vector<Point3>& centers, const RBFConfig& cfg) {
    Matrix out(centers.size(), Vector(centers.size(), 0.0));

    for (std::size_t i = 0; i < centers.size(); ++i) {
        for (std::size_t j = 0; j < centers.size(); ++j) {
            out[i][j] = evaluate_kernel(
                cfg.kernel,
                point_distance(centers[i], centers[j]),
                cfg.epsilon
            );
        }
        out[i][i] += regularization_sign(kernel_id(cfg.kernel)) * cfg.regularization;
    }

    return out;
}

} // namespace deeprbf
