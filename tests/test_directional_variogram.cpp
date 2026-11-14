#include "deeprbf/directional_variogram.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 using namespace deeprbf;
 std::vector<Point3> xyz{{0,0,0},{10,0,0},{20,0,0},{0,10,0}};
 std::vector<double> grade{0,2,4,100};
 DirectionalVariogramOptions opt;opt.direction={1,0,0};opt.angular_tolerance_deg=0;opt.lag_width=10;opt.lag_count=2;
 auto b=experimental_directional_variogram(xyz,grade,opt);
 assert(b[0].pairs==2 && std::abs(b[0].semivariance-2)<1e-12);
 assert(b[1].pairs==1 && std::abs(b[1].semivariance-8)<1e-12);
 opt.direction={0,1,0};b=experimental_directional_variogram(xyz,grade,opt);
 assert(b[0].pairs==1 && std::abs(b[0].semivariance-5000)<1e-12);
 opt.direction={0,0,0};bool rejected=false;
 try{experimental_directional_variogram(xyz,grade,opt);}catch(const std::invalid_argument&){rejected=true;}
 assert(rejected);
 std::cout<<"directional variogram: 3 directional checks and invalid-input check passed\n";
}
