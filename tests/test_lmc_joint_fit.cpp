#include "deeprbf/gc_lmc_fit.hpp"
#include "deeprbf/lmc_psd.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
 using namespace deeprbf;
 VariogramResult a,b,c;
 a.categories={1,1}; b.categories={2,2}; c.categories={1,2};
 for(int i=1;i<=12;++i) {
  double h=10.0*i;
  double r=h>=100?0.:1.-1.5*h/100.+0.5*std::pow(h/100.,3.);
  a.lags.push_back({h,2.*(1.-r),100});
  b.lags.push_back({h,1.*(1.-r),100});
  c.lags.push_back({h,.5*(1.-r),100});
 }
 auto fitted=fit_lmc_from_experimental_variograms({a,b},{c},{1,2},2,"spherical",1e-9,80);
 assert(fitted.structures().size()==2);
 assert(fitted.covariance(0,1,0.)>0.);
 for(const auto& s:fitted.structures()) {
  Eigen::Matrix2d m;
  for(int i=0;i<2;++i)for(int j=0;j<2;++j)m(i,j)=s.covariance_matrix[i][j];
  assert(is_coregionalization_psd(m,1e-8));
 }
 bool caught=false;
 try { (void)fit_lmc_from_experimental_variograms({a,b},{c},{1,2},2,"unsupported"); }
 catch(const std::invalid_argument&) {caught=true;}
 assert(caught);
 std::cout<<"Joint LMC fitting: 2 PSD structures; invalid model rejected\n";
}
