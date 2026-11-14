#include "deeprbf/continuous_cokriging.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace deeprbf;
int main() {
  Eigen::Matrix2d B; B << 4.,1.2,1.2,2.;
  std::vector<ContinuousLMCStructure> model{{30.,B}};
  std::vector<double> means{2.,5.};
  std::vector<ContinuousObservation> obs{{{0.,0.,0.},0,3.},{{10.,0.,0.},1,7.}};
  auto exact=continuous_simple_cokriging(obs,{0.,0.,0.},0,means,model);
  assert(std::abs(exact.estimate-3.)<1e-10 && exact.variance<1e-10);
  auto remote=continuous_simple_cokriging(obs,{100.,0.,0.},0,means,model);
  assert(std::abs(remote.estimate-2.)<1e-10 && std::abs(remote.variance-4.)<1e-10);
  auto cross=continuous_simple_cokriging(obs,{0.,0.,0.},1,means,model);
  assert(std::isfinite(cross.estimate) && cross.variance>=0. && cross.variance<=2.);
  bool rejected=false;
  try { Eigen::Matrix2d bad;bad << 1.,2.,2.,1.;continuous_simple_cokriging(obs,{0.,0.,0.},0,means,{{30.,bad}}); }
  catch(const std::invalid_argument&) {rejected=true;}
  assert(rejected);
  std::cout << "PASS continuous simple cokriging: exactitude, remote mean, cross-variable, PSD rejection\n";
}
