#include "deeprbf/lmc_psd.hpp"
#include <iostream>
#include <cassert>
#include <stdexcept>
int main(){
 using namespace deeprbf;
 Eigen::Matrix2d m; m<<1,2,2,1;
 assert(!is_coregionalization_psd(m));
 auto p=project_coregionalization_psd(m);
 assert(is_coregionalization_psd(p));
 assert((p-p.transpose()).norm()<1e-12);
 assert((p*m.col(0)).allFinite());
 Eigen::Matrix2d valid; valid<<2,.3,.3,1;
 assert(is_coregionalization_psd(valid));
 assert((project_coregionalization_psd(valid)-valid).norm()<1e-12);
 bool caught=false;try{project_coregionalization_psd(Eigen::MatrixXd::Constant(2,2,NAN));}catch(const std::invalid_argument&){caught=true;}assert(caught);
 std::cout<<"LMC PSD projection and validation passed\n";
}
