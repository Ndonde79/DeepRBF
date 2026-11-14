#include "deeprbf/gc_geostatistics.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(){
 using namespace deeprbf;
 std::vector<Point3> p{{0,0,0},{10,0,0},{0,10,0},{0,0,10},{10,10,0},{10,0,10}};
 std::vector<double> z{1.2,2.3,0.8,3.1,2.0,4.2};
 auto at=ordinary_kriging_with_range(p,z,p,6,3,30,20,1.0,0.0);
 if(at.estimate.size()!=p.size())throw std::runtime_error("result size");
 double err=0;
 for(size_t i=0;i<p.size();++i){
  if(at.n_samples[i]<3||!std::isfinite(at.estimate[i])||!std::isfinite(at.variance[i]))throw std::runtime_error("invalid result");
  err=std::max(err,std::abs(at.estimate[i]-z[i]));
  if(at.variance[i]<-1e-8)throw std::runtime_error("negative kriging variance");
 }
 std::cout<<"native OK conditioning maximum error="<<err<<"\n";
 if(err>1e-5)throw std::runtime_error("OK failed conditioning exactitude");
 std::vector<double> constant(p.size(),7.25);
 auto mid=ordinary_kriging_with_range(p,constant,{{5,5,5}},6,3,30,20,1.0,0.0);
 if(mid.n_samples[0]<3||std::abs(mid.estimate[0]-7.25)>1e-6)throw std::runtime_error("OK failed constant reproduction");
 std::cout<<"native OK constant reproduction="<<mid.estimate[0]<<" variance="<<mid.variance[0]<<"\n";
}
