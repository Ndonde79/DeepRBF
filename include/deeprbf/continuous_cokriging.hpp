#pragma once
#include "deeprbf/core.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace deeprbf {
// Continuous-variable SIMPLE cokriging: known stationary means, spherical nested LMC.
// This is deliberately distinct from categorical indicator cokriging and ordinary cokriging.
struct ContinuousObservation { Point3 xyz; int variable; double value; };
struct ContinuousLMCStructure {
    double range;
    Eigen::MatrixXd coregionalization;
    // Transform geographic offsets into anisotropic distance coordinates.
    // Identity retains backward compatibility with isotropic models.
    Eigen::Matrix3d anisotropy = Eigen::Matrix3d::Identity();
};
struct ContinuousCoKrigingEstimate { double estimate; double variance; int samples; };

inline double continuous_spherical(double h) {
    if (h >= 1.) return 0.;
    return 1. - 1.5*h + 0.5*h*h*h;
}
inline double continuous_lmc_covariance(const Point3& a, const Point3& b,
                                        int i, int j,
                                        const std::vector<ContinuousLMCStructure>& structures) {
    const Eigen::Vector3d delta(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
    double c = 0.;
    for (const auto& s : structures) {
        const double d = (s.anisotropy * delta).norm();
        c += s.coregionalization(i,j) * continuous_spherical(d/s.range);
    }
    return c;
}
inline ContinuousCoKrigingEstimate continuous_simple_cokriging(
    const std::vector<ContinuousObservation>& observations, const Point3& target,
    int target_variable, const std::vector<double>& means,
    const std::vector<ContinuousLMCStructure>& structures, double diagonal_jitter=0.) {
    const int nv = static_cast<int>(means.size());
    if (nv == 0 || target_variable < 0 || target_variable >= nv || structures.empty() ||
        observations.empty() || !std::isfinite(diagonal_jitter) || diagonal_jitter < 0.)
        throw std::invalid_argument("invalid continuous cokriging configuration");
    for (double v : target) if (!std::isfinite(v)) throw std::invalid_argument("invalid target");
    for (double m : means) if (!std::isfinite(m)) throw std::invalid_argument("invalid mean");
    for (const auto& s : structures) {
        if (!std::isfinite(s.range) || s.range <= 0. || s.coregionalization.rows()!=nv || s.coregionalization.cols()!=nv ||
            !s.coregionalization.allFinite() || !s.anisotropy.allFinite() ||
            std::abs(s.anisotropy.determinant()) < 1e-12 || !s.coregionalization.isApprox(s.coregionalization.transpose(),1e-10))
            throw std::invalid_argument("invalid LMC structure");
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(s.coregionalization);
        if (eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff() < -1e-10)
            throw std::invalid_argument("LMC must be positive semidefinite");
    }
    const Eigen::Index n=static_cast<Eigen::Index>(observations.size());
    Eigen::MatrixXd A(n,n);
    Eigen::VectorXd b(n), residual(n);
    for (Eigen::Index i=0;i<n;++i) {
        const auto& oi=observations[static_cast<size_t>(i)];
        if (oi.variable<0 || oi.variable>=nv || !std::isfinite(oi.value)) throw std::invalid_argument("invalid observation");
        for(double x:oi.xyz) if(!std::isfinite(x)) throw std::invalid_argument("invalid observation coordinate");
        residual(i)=oi.value-means[static_cast<size_t>(oi.variable)];
        b(i)=continuous_lmc_covariance(oi.xyz,target,oi.variable,target_variable,structures);
        for (Eigen::Index j=0;j<=i;++j) {
            const auto& oj=observations[static_cast<size_t>(j)];
            if (oj.variable<0 || oj.variable>=nv) throw std::invalid_argument("invalid variable index");
            A(i,j)=continuous_lmc_covariance(oi.xyz,oj.xyz,oi.variable,oj.variable,structures);
            A(j,i)=A(i,j);
        }
        A(i,i)+=diagonal_jitter;
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if(ldlt.info()!=Eigen::Success || (ldlt.vectorD().array()<=0.).any())
        throw std::runtime_error("singular or nonpositive conditioning covariance; check duplicates and jitter");
    Eigen::VectorXd w=ldlt.solve(b);
    if(ldlt.info()!=Eigen::Success || !w.allFinite() || (A*w-b).norm()>1e-7*(1.+b.norm()))
        throw std::runtime_error("cokriging linear solve failed");
    double prior=continuous_lmc_covariance(target,target,target_variable,target_variable,structures);
    double variance=prior-b.dot(w);
    if(variance < -1e-7*std::max(1.,prior)) throw std::runtime_error("negative cokriging variance");
    return {means[static_cast<size_t>(target_variable)]+w.dot(residual),std::max(0.,variance),static_cast<int>(n)};
}

// Ordinary cokriging with one unbiasedness constraint per represented variable.
// Each variable has an unknown but locally constant mean. The target-variable
// weights sum to one; all other variable weights sum to zero.
inline ContinuousCoKrigingEstimate continuous_ordinary_cokriging(
    const std::vector<ContinuousObservation>& observations, const Point3& target,
    int target_variable, int variable_count,
    const std::vector<ContinuousLMCStructure>& structures, double diagonal_jitter=0.) {
    if (variable_count<=0 || target_variable<0 || target_variable>=variable_count ||
        observations.empty() || structures.empty() || !std::isfinite(diagonal_jitter) || diagonal_jitter<0.)
        throw std::invalid_argument("invalid ordinary cokriging configuration");
    for(double x:target) if(!std::isfinite(x)) throw std::invalid_argument("nonfinite target");
    for(const auto& s:structures) {
        if(!std::isfinite(s.range) || s.range<=0. ||
           s.coregionalization.rows()!=variable_count || s.coregionalization.cols()!=variable_count ||
           !s.coregionalization.allFinite() || !s.anisotropy.allFinite() ||
           std::abs(s.anisotropy.determinant()) < 1e-12 ||
           !s.coregionalization.isApprox(s.coregionalization.transpose(),1e-10))
            throw std::invalid_argument("invalid LMC structure");
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(s.coregionalization);
        if(eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff() < -1e-10)
            throw std::invalid_argument("non-PSD LMC structure");
    }
    const Eigen::Index n=static_cast<Eigen::Index>(observations.size());
    std::vector<int> represented(variable_count,0), mapping(variable_count,-1);
    for(const auto& o:observations) {
        if(o.variable<0 || o.variable>=variable_count || !std::isfinite(o.value))
            throw std::invalid_argument("invalid observation");
        for(double x:o.xyz) if(!std::isfinite(x)) throw std::invalid_argument("nonfinite coordinate");
        represented[o.variable]=1;
    }
    if(!represented[target_variable]) throw std::invalid_argument("target variable has no observations");
    int m=0;
    for(int i=0;i<variable_count;++i) if(represented[i]) mapping[i]=m++;
    Eigen::MatrixXd K=Eigen::MatrixXd::Zero(n+m,n+m);
    Eigen::VectorXd rhs=Eigen::VectorXd::Zero(n+m);
    for(Eigen::Index i=0;i<n;++i) {
        const auto& a=observations[static_cast<size_t>(i)];
        rhs(i)=continuous_lmc_covariance(a.xyz,target,a.variable,target_variable,structures);
        K(i,n+mapping[a.variable])=1.;
        K(n+mapping[a.variable],i)=1.;
        for(Eigen::Index j=0;j<=i;++j) {
            const auto& b=observations[static_cast<size_t>(j)];
            K(i,j)=continuous_lmc_covariance(a.xyz,b.xyz,a.variable,b.variable,structures);
            K(j,i)=K(i,j);
        }
        K(i,i)+=diagonal_jitter;
    }
    rhs(n+mapping[target_variable])=1.;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
    if(!lu.isInvertible()) throw std::runtime_error("singular ordinary cokriging system");
    Eigen::VectorXd solution=lu.solve(rhs);
    if(!solution.allFinite() || (K*solution-rhs).norm()>1e-7*(1.+rhs.norm()))
        throw std::runtime_error("ordinary cokriging solve failed");
    double estimate=0.;
    for(Eigen::Index i=0;i<n;++i) estimate+=solution(i)*observations[static_cast<size_t>(i)].value;
    const double prior=continuous_lmc_covariance(target,target,target_variable,target_variable,structures);
    // KKT: variance = C00 - w^T c - mu^T f0.
    const double variance=prior-solution.head(n).dot(rhs.head(n))-
        solution.tail(m).dot(rhs.tail(m));
    if(variance < -1e-7*std::max(1.,prior)) throw std::runtime_error("negative cokriging variance");
    return {estimate,std::max(0.,variance),static_cast<int>(n)};
}
} // namespace deeprbf
