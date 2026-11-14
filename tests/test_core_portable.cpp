#include "deeprbf/spatial_hash.hpp"
#include "deeprbf/gmres.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace deeprbf;
int main(){
 SpatialHash h(1.0); h.add({0,0,0},10); h.add({2,0,0},20);
 auto n=h.query({0.1,0,0},0.5,2);
 if(n.size()!=1 || n[0].index!=0) throw std::runtime_error("spatial hash regression");
 GMRESConfig cfg; cfg.restart=3; cfg.max_iter=30; cfg.tol=1e-11;
 std::vector<double> x(3,0), b{6,11,8}; std::string status;
 auto A=[](const auto& v,auto& out){out={4*v[0]+v[1],2*v[0]+3*v[1]+v[2],v[1]+2*v[2]};};
 auto M=[](const auto& v,auto& out){out={v[0]/4,v[1]/3,v[2]/2};};
 if(!gmres_solve(3,A,b,x,cfg,status,M)) throw std::runtime_error("GMRES convergence");
 if(std::abs(x[0]-1)>1e-8||std::abs(x[1]-2)>1e-8||std::abs(x[2]-3)>1e-8) throw std::runtime_error("GMRES accuracy");
 std::cout << "PASS: native spatial hash and preconditioned GMRES\n";
}
