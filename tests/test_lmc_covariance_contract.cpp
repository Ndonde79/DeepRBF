#include "deeprbf/gc_geostatistics.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <iostream>
int main() {
  deeprbf::LinearModelOfCoregionalization lmc;
  deeprbf::LMCStructure a; a.covariance_matrix={{2.0,0.5},{0.5,1.0}};
  lmc.add_structure(a);
  assert(std::abs(lmc.covariance(0,1,0.0)-0.5)<1e-12);
  assert(std::abs(lmc.covariance(0,1,1.0))<1e-12);
  assert(std::abs(lmc.covariance(0,0,0.5)-0.625)<1e-12);
  bool threw=false;
  try { (void)lmc.covariance(-1,0,0.0); } catch(const std::invalid_argument&) {threw=true;}
  assert(threw);
  threw=false;
  try { (void)lmc.covariance(0,0,-1.0); } catch(const std::invalid_argument&) {threw=true;}
  assert(threw);
  std::cout<<"LMC covariance contract passed\n";
}
