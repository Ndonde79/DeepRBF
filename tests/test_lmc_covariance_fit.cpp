#include "deeprbf/lmc_covariance_fit.hpp"
#include "deeprbf/lmc_psd.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
 using namespace deeprbf;
 Eigen::Matrix2d a,b;
 a<<2.0,0.5,0.5,1.0;
 b<<0.6,-0.2,-0.2,0.8;
 for(const std::string family : {"spherical", "exponential", "gaussian"}) {
 std::vector<LMCCovarianceObservation> observations;
 for(int i=0;i<=30;++i) {
  double h=i*5.0;
  auto sph=[&](double range){double r=h/range;return lmc_correlation(r,family);};
  LMCCovarianceObservation o;
  o.lag=h;o.covariance=sph(60)*a+sph(150)*b;
  observations.push_back(o);
 }
 auto fit=fit_lmc_to_covariances(observations,{60,150},family,1e-12,800);
 if(fit.model.structures().size()!=2||fit.weighted_rss>1e-7)throw std::runtime_error("LMC covariance recovery failed");
 for(const auto& s:fit.model.structures()) {
  if(s.family!=family)throw std::runtime_error("LMC covariance family lost");
  Eigen::Matrix2d m;
  for(int i=0;i<2;++i)for(int j=0;j<2;++j)m(i,j)=s.covariance_matrix[i][j];
  if(!is_coregionalization_psd(m,1e-9))throw std::runtime_error("LMC PSD failure");
 }
 bool caught=false;
 try {fit_lmc_to_covariances(observations,{-1});}catch(const std::invalid_argument&){caught=true;}
 if(!caught)throw std::runtime_error("invalid range accepted");
 std::cout<<"LMC covariance fit PASS rss="<<fit.weighted_rss<<" iterations="<<fit.iterations<<"\n";
 }
}
