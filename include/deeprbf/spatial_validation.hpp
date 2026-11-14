#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <map>
#include <set>
namespace deeprbf {
struct ValidationPoint { double x,y,z; int domain; };
struct SpatialFold { std::vector<std::size_t> train, test; };
// Deterministic contiguous coordinate folds. Test points never participate in training.
inline std::vector<SpatialFold> coordinate_folds(const std::vector<ValidationPoint>& p,
                                                  std::size_t folds, int axis=0,
                                                  double buffer=0.0, bool same_domain_only=false) {
 if (folds<2 || folds>p.size() || axis<0 || axis>2 || !std::isfinite(buffer) || buffer<0)
   throw std::invalid_argument("invalid spatial cross-validation parameters");
 auto coord=[&](std::size_t i){return axis==0?p[i].x:(axis==1?p[i].y:p[i].z);};
 for(std::size_t i=0;i<p.size();++i)
   if(!std::isfinite(p[i].x)||!std::isfinite(p[i].y)||!std::isfinite(p[i].z))
     throw std::invalid_argument("non-finite spatial coordinate");
 std::vector<std::size_t> order(p.size());
 for(std::size_t i=0;i<p.size();++i)order[i]=i;
 std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return coord(a)<coord(b);});
 std::vector<SpatialFold> result(folds);
 for(std::size_t k=0;k<folds;++k){
   auto lo=k*p.size()/folds, hi=(k+1)*p.size()/folds;
   auto& f=result[k];
   f.test.insert(f.test.end(),order.begin()+lo,order.begin()+hi);
   for(std::size_t i=0;i<p.size();++i){
     if(std::find(f.test.begin(),f.test.end(),i)!=f.test.end())continue;
     bool excluded=false;
     if(buffer>0){for(auto j:f.test){
       if(same_domain_only && p[i].domain!=p[j].domain)continue;
       const double dx=p[i].x-p[j].x,dy=p[i].y-p[j].y,dz=p[i].z-p[j].z;
       if(std::hypot(std::hypot(dx,dy),dz)<buffer){excluded=true;break;}
     }}
     if(!excluded)f.train.push_back(i);
   }
 }
 return result;
}
// Keep all intervals belonging to the same drillhole in one test fold.
// Groups are sorted by centroid along the selected spatial axis, then assigned
// in contiguous groups to folds. Buffers exclude whole training drillholes.
inline std::vector<SpatialFold> grouped_coordinate_folds(
    const std::vector<ValidationPoint>& p, const std::vector<std::size_t>& group,
    std::size_t folds, int axis=0, double buffer=0.0, bool same_domain_only=false) {
  if (p.size()!=group.size() || p.empty() || axis<0 || axis>2 ||
      !std::isfinite(buffer) || buffer<0)
    throw std::invalid_argument("invalid grouped spatial validation parameters");
  std::map<std::size_t, std::vector<std::size_t>> members;
  for(std::size_t i=0;i<p.size();++i) {
    if(!std::isfinite(p[i].x)||!std::isfinite(p[i].y)||!std::isfinite(p[i].z))
      throw std::invalid_argument("non-finite coordinate");
    members[group[i]].push_back(i);
  }
  if(folds<2 || folds>members.size())
    throw std::invalid_argument("invalid grouped fold count");
  struct G { std::size_t id; double center; };
  std::vector<G> sorted;
  for(const auto& [id, indices]:members) {
    double sum=0;
    for(auto i:indices)sum+=axis==0?p[i].x:(axis==1?p[i].y:p[i].z);
    sorted.push_back({id,sum/static_cast<double>(indices.size())});
  }
  std::stable_sort(sorted.begin(),sorted.end(),[](const G&a,const G&b){
    return a.center==b.center?a.id<b.id:a.center<b.center;
  });
  std::vector<SpatialFold> result(folds);
  for(std::size_t k=0;k<folds;++k) {
    auto& f=result[k];
    std::set<std::size_t> testgroups;
    for(std::size_t j=k*sorted.size()/folds;j<(k+1)*sorted.size()/folds;++j)
      testgroups.insert(sorted[j].id);
    for(std::size_t i=0;i<p.size();++i)
      if(testgroups.count(group[i])) f.test.push_back(i);
    for(const auto& [id,indices]:members) {
      if(testgroups.count(id))continue;
      bool exclude=false;
      for(auto i:indices) {
        for(auto j:f.test) {
          if(same_domain_only && p[i].domain!=p[j].domain)continue;
          if(std::hypot(std::hypot(p[i].x-p[j].x,p[i].y-p[j].y),p[i].z-p[j].z)<buffer) {
            exclude=true;break;
          }
        }
        if(exclude)break;
      }
      if(!exclude)f.train.insert(f.train.end(),indices.begin(),indices.end());
    }
  }
  return result;
}

}
