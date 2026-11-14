#include "deeprbf/rbf_model.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
int main() {
 using namespace deeprbf;
 const std::vector<std::array<double,3>> p{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1},{2,1,0}};
 const std::vector<double> v{1.0,2.5,-0.5,3.0,1.2,4.0};
 for(const auto& kernel: {"gaussian","multiquadric","inverse_multiquadric"}) {
  NativeExactRBFImplicitModel model(kernel,1.5,0.0);
  model.fit(p,v);
  if(!model.is_fitted() || model.n_centers()!=static_cast<int>(p.size())) throw std::runtime_error("fit metadata invalid");
  const auto y=model.predict(p);
  if(y.size()!=v.size()) throw std::runtime_error("prediction count mismatch");
  double max_error=0;
  for(size_t i=0;i<v.size();++i) {if(!std::isfinite(y[i])) throw std::runtime_error("nonfinite prediction");max_error=std::max(max_error,std::abs(y[i]-v[i]));}
  std::cout<<kernel<<" max training error="<<max_error<<" backend="<<model.solver_backend()<<'\n';
  if(max_error>1e-6) throw std::runtime_error("RBF failed interpolation exactitude");
 }
}
