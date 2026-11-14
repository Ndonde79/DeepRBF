#include "deeprbf/lmc_covariance_fit.hpp"
#include "deeprbf/lmc_psd.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace deeprbf {
namespace {
double correlation(double h,double range,const std::string& model) {
 const double r=h/range;
 if(model=="spherical") return r>=1.0?0.0:1.0-1.5*r+0.5*r*r*r;
 if(model=="exponential") return std::exp(-3.0*r);
 return std::exp(-3.0*r*r);
}
}
LMCCovarianceFit fit_lmc_to_covariances(const std::vector<LMCCovarianceObservation>& obs,
 const std::vector<double>& ranges,const std::string& model,double tolerance,int max_iterations) {
 if(obs.empty()||ranges.empty()||max_iterations<1||!std::isfinite(tolerance)||tolerance<0 ||
    (model!="spherical"&&model!="exponential"&&model!="gaussian"))
  throw std::invalid_argument("invalid LMC fit settings");
 const Eigen::Index n=obs.front().covariance.rows();
 if(n<1) throw std::invalid_argument("empty covariance matrix");
 for(double r:ranges) if(!std::isfinite(r)||r<=0) throw std::invalid_argument("invalid range");
 for(const auto& o:obs) {
  if(!std::isfinite(o.lag)||o.lag<0||!std::isfinite(o.weight)||o.weight<=0||
    o.covariance.rows()!=n||o.covariance.cols()!=n||!o.covariance.allFinite()||
    !o.covariance.isApprox(o.covariance.transpose(),1e-9))
   throw std::invalid_argument("invalid covariance observation");
 }
 const size_t kmax=ranges.size();
 std::vector<Eigen::MatrixXd> b(kmax,Eigen::MatrixXd::Zero(n,n));
 std::vector<std::vector<double>> rho(kmax,std::vector<double>(obs.size()));
 for(size_t k=0;k<kmax;++k)for(size_t i=0;i<obs.size();++i)
  rho[k][i]=correlation(obs[i].lag,ranges[k],model);
 double previous=std::numeric_limits<double>::infinity();
 LMCCovarianceFit out;
 for(int iteration=0;iteration<max_iterations;++iteration) {
  for(size_t k=0;k<kmax;++k) {
   Eigen::MatrixXd numerator=Eigen::MatrixXd::Zero(n,n);
   double denominator=0;
   for(size_t i=0;i<obs.size();++i) {
    Eigen::MatrixXd residual=obs[i].covariance;
    for(size_t j=0;j<kmax;++j) if(j!=k) residual.noalias()-=rho[j][i]*b[j];
    numerator.noalias()+=obs[i].weight*rho[k][i]*residual;
    denominator+=obs[i].weight*rho[k][i]*rho[k][i];
   }
   if(denominator>1e-14) b[k]=project_coregionalization_psd(numerator/denominator);
  }
  double rss=0;
  for(size_t i=0;i<obs.size();++i) {
   Eigen::MatrixXd residual=obs[i].covariance;
   for(size_t k=0;k<kmax;++k) residual.noalias()-=rho[k][i]*b[k];
   rss+=obs[i].weight*residual.squaredNorm();
  }
  if(!std::isfinite(rss))throw std::runtime_error("LMC objective diverged");
  out.iterations=iteration+1;
  out.weighted_rss=rss;
  if(std::isfinite(previous)&&std::abs(previous-rss)<=tolerance*(1.0+previous)) {
   out.converged=true; break;
  }
  previous=rss;
 }
 for(size_t k=0;k<kmax;++k) {
  LMCStructure s;
  s.structure_id=static_cast<int>(k);
  s.family=model;
  s.range_major=s.range_minor=s.range_vertical=ranges[k];
  s.covariance_matrix.resize(n,std::vector<double>(n));
  for(Eigen::Index i=0;i<n;++i)for(Eigen::Index j=0;j<n;++j)s.covariance_matrix[i][j]=b[k](i,j);
  out.model.add_structure(s);
 }
 return out;
}
}
