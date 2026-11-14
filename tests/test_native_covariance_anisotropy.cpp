#include "deeprbf/gc_geostatistics.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>
using namespace deeprbf;
static void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
static bool close(double a,double b,double tol=1e-12){return std::abs(a-b)<=tol;}
int main(){
 check(close(spherical_covariance(0,2,0.2),2),"covariance origin");
 check(close(spherical_covariance(0.5,2,0.2),1.8*(1-0.75+0.0625)),"covariance interior");
 check(close(spherical_covariance(1,2,0.2),0),"covariance range");
 check(close(spherical_covariance(2,2,0.2),0),"covariance outside range");
 Point3 o{0,0,0},x{10,0,0},y{0,10,0},z{0,0,10};
 check(close(anisotropic_scaled_distance(o,x,20,10,5,0,0,0),0.5),"major axis");
 check(close(anisotropic_scaled_distance(o,y,20,10,5,0,0,0),1),"minor axis");
 check(close(anisotropic_scaled_distance(o,z,20,10,5,0,0,0),2),"vertical axis");
 check(close(anisotropic_scaled_distance(o,x,20,10,5,90,0,0),1),"rotated minor axis");
 check(close(anisotropic_scaled_distance(o,y,20,10,5,90,0,0),0.5),"rotated major axis");
 bool threw=false;try{(void)spherical_covariance(0.2,1,2);}catch(const std::invalid_argument&){threw=true;}
 check(threw,"invalid covariance parameters not rejected");
 std::cout<<"covariance and anisotropy: 10 checks passed\n";
}
