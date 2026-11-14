
#include "deeprbf/spatial_hash.hpp"
#include "deeprbf/kernels.hpp"
#include "deeprbf/gmres.hpp"
#include "deeprbf/scalfmm_adapter.hpp"
#include "deeprbf/operator.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
using namespace deeprbf;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){
 std::mt19937 rng(42);std::uniform_real_distribution<double> U(-10,10);
 SpatialHash hash(.25);for(int i=0;i<500;++i)hash.add({U(rng),U(rng),U(rng)},i);
 for(int k=0;k<50;++k){Point3 q{U(rng),U(rng),U(rng)};double radius=k%2?12.:.4;
  std::vector<std::pair<double,int>> ref;for(int i=0;i<500;++i){auto p=hash.points[i];double d=std::hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2]);if(d<=radius)ref.push_back({d,i});}
  std::sort(ref.begin(),ref.end());if(ref.size()>17)ref.resize(17);auto got=hash.query(q,radius,17);check(got.size()==ref.size(),"hash count");for(size_t i=0;i<got.size();++i)check(got[i].index==ref[i].second,"hash neighbor differs");
 }
 bool threw=false;try{hash.query({0,0,0},1,0);}catch(const std::invalid_argument&){threw=true;}check(threw,"zero neighbor count");
 GMRESConfig cfg;cfg.restart=3;cfg.max_iter=30;cfg.tol=1e-11;std::string status;std::vector<double>x(3,0),b{6,11,8};
 auto A=[](const auto& v,auto& out){out={4*v[0]+v[1],2*v[0]+3*v[1]+v[2],v[1]+2*v[2]};};
 auto M=[](const auto& v,auto& out){out={v[0]/4,v[1]/3,v[2]/2};};
 check(gmres_solve(3,A,b,x,cfg,status,M),"GMRES failed");check(std::abs(x[0]-1)<1e-8&&std::abs(x[1]-2)<1e-8&&std::abs(x[2]-3)<1e-8,"GMRES wrong solution");
 b={0,0,0};x={0,0,0};check(gmres_solve(3,A,b,x,cfg,status),"zero RHS");
 #ifdef DEEPRBF_TEST_REAL_SCALFMM
 // Exercise the actual FMM even below the production source-count threshold.
 std::vector<Point3> sources,targets;std::vector<double>w;for(int i=0;i<80;++i){sources.push_back({U(rng)/10,U(rng)/10,U(rng)/10});w.push_back(1.+i/80.);}
 for(int i=0;i<25;++i)targets.push_back({U(rng)/10,U(rng)/10,U(rng)/10});
 ScalFMMOptions opts;opts.minimum_fmm_sources=1;opts.interpolation_order=6;opts.tree_height=4;opts.validation_relative_tolerance=2e-3;
 ScalFMMAdapter adapter;adapter.setup(sources,"gaussian",1.4,opts);std::vector<double>y;adapter.evaluate(targets,w,y);
 auto info=adapter.execution_info();check(info.active&&!info.fallback_used&&info.validated,"FMM did not execute/validate");
 std::cout<<"ScalFMM executed="<<info.active<<" validated="<<info.validated<<" fallback="<<info.fallback_used<<" relative_L2="<<info.validation_relative_error<<"\n";
 double err=0,den=0;for(size_t i=0;i<targets.size();++i){double exact=0;for(size_t j=0;j<sources.size();++j){double r=point_distance(targets[i],sources[j]);exact+=w[j]*std::exp(-r*r/(1.4*1.4));}err+=(y[i]-exact)*(y[i]-exact);den+=exact*exact;}check(std::sqrt(err/den)<2e-3,"FMM accuracy");
 // Deliberately demand unattainable precision from a low-order approximation.
 opts.interpolation_order=3;opts.validation_relative_tolerance=1e-30;opts.fallback_on_validation_failure=false;
 adapter.setup(sources,"gaussian",1.4,opts);
 for(int repeat=0;repeat<2;++repeat){bool rejected=false;try{adapter.evaluate(targets,w,y);}catch(const std::runtime_error&){rejected=true;}check(rejected,"FMM validation failure silently accepted");}
 RBFOperatorConfig opcfg;opcfg.kernel="gaussian";opcfg.epsilon=1.4;opcfg.scalfmm_options=opts;
 RBFMatrixFreeOperator op(sources,opcfg);
 for(int repeat=0;repeat<2;++repeat){bool rejected=false;try{op.apply(w,y);}catch(const std::runtime_error&){rejected=true;}check(rejected,"operator swallowed FMM validation failure");}
 std::cout<<"ScalFMM adapter and operator reject failed validation on repeated calls\n";
 #else
 std::cout<<"ScalFMM backend unavailable: accelerator test SKIPPED (not a pass)\n";
 #endif
 std::cout<<"SpatialHash and GMRES PASS\n";
}
