#include "deeprbf/audited_engine.hpp"
#include "deeprbf/spatial_hash.hpp"
#include "deeprbf/kd_tree.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

namespace deeprbf {
namespace {
std::vector<Neighbor> canonical_query(const KDTree3& tree,const Point3& q,double radius,int count){
    auto found=tree.query(q,radius+1e-12,std::min(count+1,static_cast<int>(tree.points.size())));
    auto key=[](double value){return std::nearbyint(value*1e12);};
    if(found.size()>static_cast<size_t>(count)&&key(found[count-1].distance)==key(found[count].distance))
        found=tree.query(q,std::min(radius+1e-12,found[count-1].distance+1e-12),static_cast<int>(tree.points.size()));
    std::sort(found.begin(),found.end(),[&](const Neighbor& a,const Neighbor& b){auto da=key(a.distance),db=key(b.distance);return da==db?a.index<b.index:da<db;});
    if(found.size()>static_cast<size_t>(count))found.resize(count);return found;
}
double separation(const Point3& a,const Point3& b) {
    return std::hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
}
std::vector<double> local_probability(int node,const std::vector<int>& neighbors,
    const std::vector<Point3>& x,const std::vector<int>& labels,
    const std::vector<double>& prior,double& maximum_residual) {
    if(neighbors.empty())return prior;
    const int m=static_cast<int>(neighbors.size());
    Eigen::MatrixXd cov(m,m);Eigen::VectorXd cross(m);
    for(int i=0;i<m;++i){
        const double d=separation(x[node],x[neighbors[i]]);
        if(d==0){std::vector<double> p(prior.size(),0.);p[labels[neighbors[i]]]=1.;return p;}
        cross[i]=std::exp(-3*d);
        for(int j=0;j<m;++j)cov(i,j)=std::exp(-3*separation(x[neighbors[i]],x[neighbors[j]]));
        cov(i,i)+=1e-7; // Explicit reference numerical diagonal, not a fitted nugget.
    }
    Eigen::LLT<Eigen::MatrixXd> factor(cov);
    if(factor.info()!=Eigen::Success)throw std::runtime_error("Indicator covariance factorization failed");
    Eigen::VectorXd weights=factor.solve(cross);
    const double residual=(cov*weights-cross).norm()/std::max(1.,cross.norm());
    if(!weights.allFinite()||!std::isfinite(residual)||residual>1e-9)
        throw std::runtime_error("Indicator solve residual failed");
    maximum_residual=std::max(maximum_residual,residual);
    std::vector<double> p=prior;const double sum=weights.sum();
    for(auto& value:p)value*=1-sum;
    for(int i=0;i<m;++i)p[labels[neighbors[i]]]+=weights[i];
    double total=0;for(auto& value:p){value=std::max(0.,value);total+=value;}
    if(total<=1e-12)std::fill(p.begin(),p.end(),1./p.size());
    else for(auto& value:p)value/=total;
    return p;
}
// Explicit bounded-integer rejection and 53-bit uniform draws avoid implementation
// dependent std::shuffle/discrete_distribution behavior across C++ runtimes.
std::uint64_t bounded(std::mt19937_64& rng,std::uint64_t count){
    const auto threshold=static_cast<std::uint64_t>(-count)%count;
    for(;;){auto value=rng();if(value>=threshold)return value%count;}
}
}

NativeIndicatorResult indicator_lithology_native(const std::vector<Point3>& x,
    const std::vector<int>& hard,const std::vector<double>& prior,int k,
    int realizations,unsigned long long seed,int candidates,double radius,const std::vector<int>& hole_ids,int max_per_hole,int min_holes){
    if(x.empty()||x.size()!=hard.size()||prior.size()<2||prior.size()>512||
       k<1||k>128||candidates<k||candidates>512||realizations<0||
       !std::isfinite(radius)||radius<=0||radius>100)
        throw std::invalid_argument("Invalid native indicator controls");
    if((!hole_ids.empty()&&hole_ids.size()!=x.size())||max_per_hole<0||min_holes<0||min_holes>k||
       (hole_ids.empty()&&(max_per_hole||min_holes)))throw std::invalid_argument("Invalid indicator drillhole controls");
    // Bound both adjacency and outputs before allocating any large buffers.
    if(x.size()>static_cast<size_t>(std::numeric_limits<int>::max())||
       static_cast<long double>(x.size())*(candidates+k)>32000000.L||
       static_cast<long double>(x.size())*prior.size()>32000000.L||
       static_cast<long double>(x.size())*realizations>32000000.L)
        throw std::invalid_argument("Native indicator workspace/output memory limit exceeded");
    double total=0;for(double p:prior){if(!std::isfinite(p)||p<0)throw std::invalid_argument("Invalid indicator prior");total+=p;}
    if(std::abs(total-1)>1e-10)throw std::invalid_argument("Indicator prior must sum to one");
    KDTree3 all,fixed;all.reserve(x.size());std::vector<int> hard_index,unknown;
    std::map<Point3,int> unique;
    for(size_t i=0;i<x.size();++i){
        for(double value:x[i])if(!std::isfinite(value))throw std::invalid_argument("Nonfinite indicator coordinates");
        if(hard[i]<-1||hard[i]>=static_cast<int>(prior.size()))throw std::invalid_argument("Invalid encoded category");
        if(!hole_ids.empty()&&hard[i]>=0&&hole_ids[i]<0)throw std::invalid_argument("Hard samples require nonnegative hole IDs");
        if(!unique.emplace(x[i],static_cast<int>(i)).second)throw std::invalid_argument("Native indicator nodes must be unique; merge coincident data first");
        all.add(x[i],0);
        if(hard[i]>=0){fixed.add(x[i],0);hard_index.push_back(static_cast<int>(i));}
        else unknown.push_back(static_cast<int>(i));
    }
    if(hard_index.empty())throw std::invalid_argument("At least one hard indicator node required");
    all.build();fixed.build();
    NativeIndicatorResult out;out.n_nodes=static_cast<int>(x.size());out.n_categories=static_cast<int>(prior.size());out.n_realizations=realizations;
    out.direct_hard_neighbors.resize(x.size());out.direct_hard_holes.resize(x.size());out.supported.assign(x.size(),1);out.probabilities.assign(x.size()*prior.size(),0.);
    std::vector<std::vector<int>> graph(x.size()),hard_graph(x.size());
    for(size_t i=0;i<x.size();++i){
        if(hard[i]>=0){
            // No search/solve is needed at an exact constraint. -1 means the
            // neighbor count was not assessed, rather than zero support.
            out.direct_hard_neighbors[i]=-1;
            out.direct_hard_holes[i]=-1;
            if(!realizations)out.probabilities[i*prior.size()+hard[i]]=1.;
            continue;
        }
        std::map<int,int> per_hole;
        for(const auto& n:canonical_query(fixed,x[i],radius,hole_ids.empty()?k:candidates)){
            const int index=hard_index[n.index];const int hole=hole_ids.empty()?index:hole_ids[index];
            if(max_per_hole&&per_hole[hole]>=max_per_hole)continue;
            hard_graph[i].push_back(index);++per_hole[hole];if(static_cast<int>(hard_graph[i].size())==k)break;
        }
        out.direct_hard_neighbors[i]=static_cast<int>(hard_graph[i].size());
        out.direct_hard_holes[i]=hole_ids.empty()?-1:static_cast<int>(per_hole.size());
        out.supported[i]=!hard_graph[i].empty()&&static_cast<int>(per_hole.size())>=min_holes;
        if(realizations&&hard[i]<0)for(const auto& n:canonical_query(all,x[i],radius,candidates))graph[i].push_back(n.index);
        if(!realizations){
            auto p=out.supported[i]?local_probability(static_cast<int>(i),hard_graph[i],x,hard,prior,out.maximum_solve_residual):prior;
            if(hard[i]>=0)p[hard[i]]=1.;
            std::copy(p.begin(),p.end(),out.probabilities.begin()+i*prior.size());
        }
    }
    if(!realizations)return out;
    out.simulations.resize(x.size()*realizations);out.prior_only_draws.assign(realizations,0);
    for(int r=0;r<realizations;++r){
        std::mt19937_64 rng(seed+104729ULL*r);auto path=unknown;auto labels=hard;
        for(size_t i=path.size();i>1;--i)std::swap(path[i-1],path[bounded(rng,i)]);
        for(int node:path){
            std::vector<int> selected=hard_graph[node];
            for(int neighbor:graph[node])if(labels[neighbor]>=0)selected.push_back(neighbor);
            std::sort(selected.begin(),selected.end());selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
            std::sort(selected.begin(),selected.end(),[&](int a,int b){const auto da=std::nearbyint(separation(x[node],x[a])*1e12),db=std::nearbyint(separation(x[node],x[b])*1e12);return da==db?a<b:da<db;});
            if(max_per_hole){
                std::map<int,int> used;std::vector<int> filtered;
                for(int index:selected){
                    if(hard[index]>=0&&used[hole_ids[index]]>=max_per_hole)continue;
                    filtered.push_back(index);if(hard[index]>=0)++used[hole_ids[index]];
                }
                selected=std::move(filtered);
            }
            if(selected.size()>static_cast<size_t>(k))selected.resize(k);
            const bool use_prior=selected.empty()||(min_holes&&out.direct_hard_holes[node]<min_holes);
            if(use_prior)++out.prior_only_draws[r];
            auto p=use_prior?prior:local_probability(node,selected,x,labels,prior,out.maximum_solve_residual);
            const double draw=static_cast<double>(rng()>>11)*0x1.0p-53;
            double cumulative=0;int label=static_cast<int>(prior.size())-1;
            for(size_t c=0;c<p.size();++c){cumulative+=p[c];if(draw<cumulative){label=static_cast<int>(c);break;}}
            labels[node]=label;
        }
        for(size_t i=0;i<x.size();++i){
            if(labels[i]<0||(hard[i]>=0&&labels[i]!=hard[i]))throw std::runtime_error("Native SIS conditioning invariant failed");
            out.simulations[static_cast<size_t>(r)*x.size()+i]=labels[i];
            out.probabilities[i*prior.size()+labels[i]]+=1.;
        }
    }
    for(auto& value:out.probabilities)value/=realizations;
    return out;
}
} // namespace deeprbf
