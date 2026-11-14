#include "deeprbf/continuous_cokriging.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace deeprbf;
int main(){
  std::vector<ContinuousLMCStructure> model={{12.,(Eigen::Matrix2d()<<2.,0.5,0.5,1.).finished()}};
  std::vector<ContinuousObservation> data={{{0.,0.,0.},0,3.},{{4.,0.,0.},0,5.},{{0.,4.,0.},0,7.},{{0.,0.,0.},1,10.},{{4.,0.,0.},1,12.},{{0.,4.,0.},1,14.}};
  auto exact=continuous_ordinary_cokriging(data,{0.,0.,0.},0,2,model);
  assert(std::abs(exact.estimate-3.)<1e-8);
  assert(exact.variance<1e-7);
  for(auto& o:data) o.value=o.variable==0?6.:25.;
  auto constant=continuous_ordinary_cokriging(data,{2.,2.,0.},0,2,model);
  assert(std::abs(constant.estimate-6.)<1e-8);
  auto secondary=continuous_ordinary_cokriging(data,{2.,2.,0.},1,2,model);
  assert(std::abs(secondary.estimate-25.)<1e-8);
  bool rejected=false;
  try {continuous_ordinary_cokriging(data,{2.,2.,0.},2,2,model);} catch(const std::invalid_argument&){rejected=true;}
  assert(rejected);
  std::cout<<"ordinary cokriging exactitude, constraints, and invalid inputs passed\n";
}
