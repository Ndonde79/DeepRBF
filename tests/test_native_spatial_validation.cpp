#include "deeprbf/spatial_validation.hpp"
#include <iostream>
#include <stdexcept>
#include <set>
using namespace deeprbf;
int main(){
 std::vector<ValidationPoint> p;
 for(int i=0;i<30;++i)p.push_back({double(i),0,0,i%2});
 auto folds=coordinate_folds(p,5,0,2.0);
 std::set<std::size_t> all;
 for(const auto& f:folds){
  if(f.test.size()!=6)throw std::runtime_error("fold size");
  std::set<std::size_t> t(f.test.begin(),f.test.end());
  for(auto i:f.test)all.insert(i);
  for(auto i:f.train){
   if(t.count(i))throw std::runtime_error("train-test leakage");
   for(auto j:f.test)if(std::abs(p[i].x-p[j].x)<2.0)throw std::runtime_error("buffer leakage");
  }
 }
 if(all.size()!=p.size())throw std::runtime_error("incomplete fold coverage");
 bool threw=false;try{coordinate_folds(p,1);}catch(const std::invalid_argument&){threw=true;}
 if(!threw)throw std::runtime_error("invalid folds accepted");
 std::cout<<"spatial validation: partition, no leakage, buffer, invalid inputs passed\n";
}
