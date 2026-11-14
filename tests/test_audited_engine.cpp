#include "deeprbf/audited_engine.hpp"
#include "deeprbf/kd_tree.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace deeprbf;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int main(){
    KDTree3 kd;std::mt19937_64 rng(42);std::normal_distribution<double> normal;
    for(int i=0;i<1000;++i)kd.add({normal(rng),normal(rng),normal(rng)},0);
    kd.add(kd.points[0],0);kd.build();
    for(int query=0;query<30;++query){
        Point3 q{normal(rng),normal(rng),normal(rng)};
        if(query==0)q=kd.points[0];
        auto nearest=kd.query(q,1.5,17);std::vector<std::pair<double,int>> oracle;
        for(size_t i=0;i<kd.points.size();++i){auto p=kd.points[i];double d=std::hypot(q[0]-p[0],q[1]-p[1],q[2]-p[2]);if(d<=1.5)oracle.emplace_back(d,static_cast<int>(i));}
        std::sort(oracle.begin(),oracle.end());if(oracle.size()>17)oracle.resize(17);
        require(nearest.size()==oracle.size(),"KD radius result size");
        for(size_t i=0;i<oracle.size();++i)require(nearest[i].index==oracle[i].second&&nearest[i].distance==oracle[i].first,"KD brute force parity including ties");
    }
    CovarianceSpec c;c.family="exponential";c.ranges={10,20,30};c.validate();
    require(std::abs(c.covariance({0,0,0},{10,0,0})-std::exp(-1.))<1e-12,"exponential length scale");
    c.range_convention="practical";
    for(const auto& family:{"exponential","gaussian"}){
        c.family=family;
        for(int axis=0;axis<3;++axis){Point3 lag{0,0,0};lag[axis]=c.ranges[axis];
            require(std::abs(c.covariance({0,0,0},lag)-std::exp(-3.))<1e-12,"practical axis range");}
    }
    c.range_convention="invalid";bool rejected=false;try{c.validate();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"reject ambiguous convention");c.range_convention="length_scale";c.family="exponential";
    c.nugget=.2;require(std::abs(c.covariance({0,0,0},{0,0,0},true)-1)<1e-12,"total sill");
    require(std::abs(c.covariance({0,0,0},{0,0,0},false)-.8)<1e-12,"observation identity nugget");c.nugget=0;
    NeighborhoodSpec s;s.radii={50,50,50};s.max_per_hole=1;s.min_holes=3;s.max_neighbors=3;
    std::vector<Point3> xyz{{0,0,0},{1,0,0},{3,0,0},{0,3,0}};std::vector<int> holes{0,0,1,2},domains(4,0);
    auto near=select_neighborhood(xyz,holes,domains,{0,0,1},0,s);require(near.supported&&near.n_holes==3&&near.indices.size()==3,"hole quota");
    RotatedSupport target;target.center={0,0,1};std::vector<double> values(4,7.);
    auto result=audited_ordinary_kriging(xyz,values,holes,domains,{target},c,s);
    require(result.diagnostics[0].success,"qualified solve");require(std::abs(result.estimate[0]-7)<1e-10,"constant reproduction");
    require(result.variance[0]>=0&&result.diagnostics[0].relative_residual<1e-10,"residual and variance");
    require(std::abs(result.diagnostics[0].weight_sum-1)<1e-10,"unbiased weight sum diagnostics");
    require(result.diagnostics[0].weight_l1>=1-1e-10,"weight absolute sum diagnostics");
    target.center={1000,0,0};result=audited_ordinary_kriging(xyz,values,holes,domains,{target},c,s);
    require(result.diagnostics[0].status=="INSUFFICIENT_SUPPORT"&&std::isnan(result.estimate[0]),"unsupported explicit");
    s.max_per_hole=0;s.min_holes=1;s.min_neighbors=1;s.max_neighbors=10;xyz={{0,0,0},{0,0,0}};holes={0,1};domains={0,0};values={1,1};target.center={1,0,0};
    result=audited_ordinary_kriging(xyz,values,holes,domains,{target},c,s);require(result.diagnostics[0].status=="RANK_DEFICIENT","singular failure explicit");
    result=audited_ordinary_kriging(xyz,values,holes,domains,{target},c,s,1e-5);require(result.diagnostics[0].success&&result.diagnostics[0].regularization==1e-5,"explicit regularization");
    std::cout<<"PASS audited covariance, neighborhoods, solver diagnostics and failure controls\n";
}
