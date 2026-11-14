#include "deeprbf/spatial_validation.hpp"
#include <stdexcept>
#include <set>
#include <iostream>
using namespace deeprbf;
int main(){
 std::vector<ValidationPoint> p; std::vector<std::size_t> groups;
 for(std::size_t hole=0;hole<15;++hole)
  for(int depth=0;depth<3;++depth){p.push_back({double(hole*10),0,double(depth),int(hole%2)});groups.push_back(hole);}
 auto folds=grouped_coordinate_folds(p,groups,5,0,11.0);
 std::set<std::size_t> covered;
 for(const auto& f:folds){
  std::set<std::size_t> testholes;
  for(auto i:f.test){covered.insert(i);testholes.insert(groups[i]);}
  for(auto i:f.train){
   if(testholes.count(groups[i]))throw std::runtime_error("drillhole leakage");
   for(auto j:f.test)
    if(std::hypot(std::hypot(p[i].x-p[j].x,p[i].y-p[j].y),p[i].z-p[j].z)<11.0)
     throw std::runtime_error("buffer leakage");
  }
  if(f.test.empty()||f.train.empty())throw std::runtime_error("empty fold");
 }
 if(covered.size()!=p.size())throw std::runtime_error("incomplete coverage");
 bool invalid=false;try{grouped_coordinate_folds(p,groups,16);}catch(const std::invalid_argument&){invalid=true;}
 if(!invalid)throw std::runtime_error("invalid fold count accepted");
 std::cout<<"grouped drillhole folds: leakage, buffer, coverage, validation PASS\n";
}
