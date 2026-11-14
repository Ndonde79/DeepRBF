#pragma once
#include "deeprbf/core.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace deeprbf {
struct VariogramBin { double mean_lag=0, semivariance=0; std::size_t pairs=0; };
struct DirectionalVariogramOptions {
  Point3 direction{1,0,0};
  double angular_tolerance_deg=22.5;
  double lag_width=10;
  std::size_t lag_count=12;
  double bandwidth=std::numeric_limits<double>::infinity();
  bool axial=true; // Treat positive and negative directions as equivalent
};
inline std::vector<VariogramBin> experimental_directional_variogram(
    const std::vector<Point3>& xyz, const std::vector<double>& grade,
    const DirectionalVariogramOptions& opt) {
  if(xyz.size()!=grade.size() || opt.lag_count==0 || !(opt.lag_width>0) ||
     !std::isfinite(opt.lag_width) || !std::isfinite(opt.angular_tolerance_deg) ||
     opt.angular_tolerance_deg<0 || opt.angular_tolerance_deg>90 ||
     !(opt.bandwidth>=0) || std::isnan(opt.bandwidth))
    throw std::invalid_argument("invalid directional variogram input");
  const auto d=opt.direction;
  const double norm=std::hypot(std::hypot(d[0],d[1]),d[2]);
  if(!(norm>0) || !std::isfinite(norm)) throw std::invalid_argument("invalid direction");
  const Point3 u{d[0]/norm,d[1]/norm,d[2]/norm};
  for(std::size_t i=0;i<xyz.size();++i){
    if(!std::isfinite(grade[i])) throw std::invalid_argument("nonfinite grade");
    for(double c:xyz[i]) if(!std::isfinite(c)) throw std::invalid_argument("nonfinite coordinate");
  }
  const double cos_tol=std::cos(opt.angular_tolerance_deg*3.14159265358979323846/180.0);
  std::vector<VariogramBin> bins(opt.lag_count);
  for(std::size_t i=0;i<xyz.size();++i) for(std::size_t j=i+1;j<xyz.size();++j){
    const double dx=xyz[j][0]-xyz[i][0],dy=xyz[j][1]-xyz[i][1],dz=xyz[j][2]-xyz[i][2];
    const double distance=std::hypot(std::hypot(dx,dy),dz);
    if(distance==0)continue;
    double projection=dx*u[0]+dy*u[1]+dz*u[2];
    if(opt.axial) projection=std::abs(projection);
    if(projection/distance+1e-14<cos_tol)continue;
    const double transverse=std::sqrt(std::max(0.0,distance*distance-projection*projection));
    if(transverse>opt.bandwidth)continue;
    const double position=distance/opt.lag_width;
    if(!(position>0) || position>static_cast<double>(opt.lag_count))continue;
    const auto index=std::min(opt.lag_count-1,static_cast<std::size_t>(std::ceil(position)-1));
    const double delta=grade[j]-grade[i];
    auto& b=bins[index];b.mean_lag+=distance;b.semivariance+=0.5*delta*delta;++b.pairs;
  }
  for(auto& b:bins)if(b.pairs){b.mean_lag/=b.pairs;b.semivariance/=b.pairs;}
  return bins;
}
} // namespace deeprbf
