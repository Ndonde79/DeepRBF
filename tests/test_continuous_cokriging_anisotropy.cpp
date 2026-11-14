#include "deeprbf/continuous_cokriging.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
 using namespace deeprbf;
 Eigen::Matrix2d B; B << 2.,0.5,0.5,1.;
 ContinuousLMCStructure s{100.,B};
 const Point3 o{0.,0.,0.}, x{20.,0.,0.}, y{0.,20.,0.};
 auto cov=[&](const Point3& p){return continuous_lmc_covariance(o,p,0,0,{s});};
 if(std::abs(cov(x)-cov(y))>1e-12) return 1;
 s.anisotropy(1,1)=4.;
 if(!(cov(x)>cov(y))) return 2;
 std::vector<ContinuousObservation> obs{{o,0,3.},{x,1,4.}};
 auto sk=continuous_simple_cokriging(obs,o,0,{2.,4.},{s});
 auto ok=continuous_ordinary_cokriging(obs,o,0,2,{s});
 if(std::abs(sk.estimate-3.)>1e-8 || std::abs(ok.estimate-3.)>1e-8) return 3;
 s.anisotropy(2,2)=0.;
 bool rejected=false;
 try { (void)continuous_simple_cokriging(obs,o,0,{2.,4.},{s}); }
 catch(const std::invalid_argument&) {rejected=true;}
 if(!rejected) return 4;
 std::cout << "PASS: anisotropic LMC covariance, simple/ordinary cokriging exactitude, invalid metric rejection\n";
}
