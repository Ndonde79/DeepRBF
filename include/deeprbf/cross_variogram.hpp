#pragma once
#include "deeprbf/directional_variogram.hpp"
#include <cstdint>
#include <numeric>
namespace deeprbf {
// Cross-semivariogram: gamma_ab(h) = 1/(2N) sum (a_i-a_j)(b_i-b_j).
// Only within-domain pairs are used when domain labels are provided.
inline std::vector<VariogramBin> experimental_directional_cross_variogram(
    const std::vector<Point3>& xyz,
    const std::vector<double>& a, const std::vector<double>& b,
    const DirectionalVariogramOptions& opt,
    const std::vector<std::int64_t>& domains = {}) {
  if (xyz.size()!=a.size() || xyz.size()!=b.size() ||
      (!domains.empty() && domains.size()!=xyz.size()))
    throw std::invalid_argument("cross-variogram dimensions mismatch");
  // Delegate parameter and coordinate validation to the tested direct variogram.
  (void)experimental_directional_variogram(xyz,a,opt);
  for (double v:b) if (!std::isfinite(v)) throw std::invalid_argument("nonfinite cross grade");
  const auto d=opt.direction;
  const double norm=std::hypot(std::hypot(d[0],d[1]),d[2]);
  const Point3 u{d[0]/norm,d[1]/norm,d[2]/norm};
  const double ct=std::cos(opt.angular_tolerance_deg*3.14159265358979323846/180.0);
  std::vector<VariogramBin> bins(opt.lag_count);
  for(std::size_t i=0;i<xyz.size();++i) for(std::size_t j=i+1;j<xyz.size();++j){
    if(!domains.empty() && domains[i]!=domains[j]) continue;
    const double dx=xyz[j][0]-xyz[i][0],dy=xyz[j][1]-xyz[i][1],dz=xyz[j][2]-xyz[i][2];
    const double dist=std::hypot(std::hypot(dx,dy),dz);
    if(!(dist>0))continue;
    double projection=dx*u[0]+dy*u[1]+dz*u[2];
    if(opt.axial) projection=std::abs(projection);
    if(projection/dist+1e-14<ct)continue;
    const double transverse=std::sqrt(std::max(0.0,dist*dist-projection*projection));
    if(transverse>opt.bandwidth)continue;
    const double position=dist/opt.lag_width;
    if(!(position>0) || position>static_cast<double>(opt.lag_count))continue;
    const auto k=std::min(opt.lag_count-1,static_cast<std::size_t>(std::ceil(position)-1));
    auto& bin=bins[k];
    bin.mean_lag+=dist;
    bin.semivariance+=0.5*(a[j]-a[i])*(b[j]-b[i]);
    ++bin.pairs;
  }
  for(auto& bin:bins) if(bin.pairs){bin.mean_lag/=bin.pairs;bin.semivariance/=bin.pairs;}
  return bins;
}
} // namespace deeprbf
