#include "deeprbf/cross_variogram.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 using namespace deeprbf;
 const std::vector<Point3> xyz{{0,0,0},{10,0,0},{20,0,0},{30,0,0}};
 const std::vector<double> a{0,1,2,3},b{0,2,4,6};
 DirectionalVariogramOptions opt;opt.direction={1,0,0};opt.angular_tolerance_deg=0;opt.lag_width=10;opt.lag_count=3;
 auto direct=experimental_directional_variogram(xyz,a,opt);
 auto cross=experimental_directional_cross_variogram(xyz,a,a,opt);
 for(std::size_t i=0;i<3;++i){assert(cross[i].pairs==direct[i].pairs);assert(std::abs(cross[i].semivariance-direct[i].semivariance)<1e-12);}
 cross=experimental_directional_cross_variogram(xyz,a,b,opt);
 assert(cross[0].pairs==3 && std::abs(cross[0].semivariance-1.0)<1e-12);
 assert(cross[1].pairs==2 && std::abs(cross[1].semivariance-4.0)<1e-12);
 const std::vector<std::int64_t> domains{1,1,2,2};
 cross=experimental_directional_cross_variogram(xyz,a,b,opt,domains);
 assert(cross[0].pairs==2 && cross[1].pairs==0);
 bool rejected=false;
 try{experimental_directional_cross_variogram(xyz,a,std::vector<double>{1},opt);}catch(const std::invalid_argument&){rejected=true;}
 assert(rejected);
 std::cout<<"cross-variogram direct equivalence, cross covariance, domain exclusion and invalid input passed\n";
}
