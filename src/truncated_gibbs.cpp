#include "deeprbf/audited_engine.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "deeprbf/kd_tree.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace deeprbf {
namespace {
double uniform(std::mt19937_64& rng){return (static_cast<double>(rng()>>11)+.5)*0x1.0p-53;}
double truncated(double a,double b,std::mt19937_64& rng){
    if(b<0)return -truncated(-b,-a,rng);
    // Uniform rejection handles narrow intervals without subtracting nearly
    // identical Gaussian CDFs, including extreme tails.
    const bool narrow=std::isfinite(a)&&std::isfinite(b)&&b-a<1.;
    const double mode=std::max(a,std::min(b,0.));
    std::normal_distribution<double> normal;
    const double rate=a>0?(a+std::hypot(a,2.))/2:0.;
    for(int attempt=0;attempt<100000;++attempt){
        double z;
        if(narrow){z=a+(b-a)*uniform(rng);if(std::log(uniform(rng))>-.5*(z-mode)*(z+mode))continue;}
        else if(a>0){z=a-std::log(uniform(rng))/rate;if(z>b||std::log(uniform(rng))>-.5*(z-rate)*(z-rate))continue;}
        else{z=normal(rng);if(z<a||z>b)continue;}
        return std::max(std::nextafter(a,b),std::min(std::nextafter(b,a),z));
    }
    throw std::runtime_error("Truncated Gaussian rejection limit exceeded");
}
}

std::vector<double> gibbs_truncated_native(const std::vector<double>& values,
    const std::vector<double>& lower,const std::vector<double>& upper,int sweeps,unsigned long long seed){
    const size_t n=lower.size();
    if(!n||n>3000||upper.size()!=n||values.size()!=n*n||sweeps<1||sweeps>10000||n*static_cast<size_t>(sweeps)>3000000)
        throw std::invalid_argument("Invalid/bounded native Gibbs dimensions or sweeps");
    Eigen::MatrixXd precision(n,n);
    for(size_t i=0;i<n;++i){
        if(std::isnan(lower[i])||std::isnan(upper[i])||lower[i]>=upper[i])throw std::invalid_argument("Invalid Gibbs bounds");
        for(size_t j=0;j<n;++j){double value=values[i*n+j];if(!std::isfinite(value))throw std::invalid_argument("Nonfinite precision");precision(i,j)=value;}
    }
    if(!precision.isApprox(precision.transpose(),1e-10))throw std::invalid_argument("Gibbs precision must be symmetric");
    Eigen::LLT<Eigen::MatrixXd> factor(precision);
    if(factor.info()!=Eigen::Success)throw std::invalid_argument("Gibbs precision must be positive definite");
    std::mt19937_64 rng(seed);Eigen::VectorXd latent(n);
    for(size_t i=0;i<n;++i)latent[i]=truncated(lower[i],upper[i],rng);
    Eigen::VectorXd residual=precision*latent;
    std::vector<size_t> order(n);std::iota(order.begin(),order.end(),0);
    for(int sweep=0;sweep<sweeps;++sweep){
        // Rejection avoids modulo bias; seed streams are independent by field
        // and realization in the Python spectral-conditioning orchestrator.
        for(size_t count=n;count>1;--count){const auto threshold=static_cast<std::uint64_t>(-count)%count;std::uint64_t draw;do{draw=rng();}while(draw<threshold);std::swap(order[count-1],order[draw%count]);}
        for(size_t i:order){
            const double sd=1/std::sqrt(precision(i,i));const double mean=latent[i]-residual[i]/precision(i,i);
            double value=mean+sd*truncated((lower[i]-mean)/sd,(upper[i]-mean)/sd,rng);
            value=std::max(std::nextafter(lower[i],upper[i]),std::min(std::nextafter(upper[i],lower[i]),value));
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite Gibbs state");
            residual.noalias()+=precision.col(i)*(value-latent[i]);latent[i]=value;
        }
        if((sweep+1)%10==0)residual.noalias()=precision*latent; // Bound accumulated update roundoff.
    }
    return std::vector<double>(latent.data(),latent.data()+n);
}
SparseLatentResult vecchia_truncated_native(const std::vector<Point3>& x,
    const std::vector<double>& lower,const std::vector<double>& upper,int k,
    int sweeps,unsigned long long seed,double nugget,double max_workspace_mb){
    const size_t n=x.size();
    if(!n||n>200000||lower.size()!=n||upper.size()!=n||k<1||k>64||sweeps<1||sweeps>10000||n*static_cast<size_t>(sweeps)>20000000||
       !std::isfinite(nugget)||nugget<=0||nugget>=1||!std::isfinite(max_workspace_mb)||max_workspace_mb<=0)
        throw std::invalid_argument("Invalid sparse latent controls");
    const long long workspace=static_cast<long long>(n)*(k+1)*(k+1)*64+static_cast<long long>(n)*512;
    if(workspace>max_workspace_mb*1024*1024)throw std::invalid_argument("Sparse latent workspace budget exceeded");
    KDTree3 tree;tree.reserve(n);
    for(size_t i=0;i<n;++i){
        for(double v:x[i])if(!std::isfinite(v))throw std::invalid_argument("Nonfinite sparse coordinates");
        if(std::isnan(lower[i])||std::isnan(upper[i])||lower[i]>=upper[i])throw std::invalid_argument("Invalid sparse bounds");
        tree.add(x[i],0);
    }tree.build();
    // Fixed geometry ordering independent of draw seed. Candidate truncation is
    // an explicit approximation; no dense covariance/precision is allocated.
    std::mt19937_64 geometry_rng(1729),rng(seed);std::vector<int> order(n),rank(n);
    std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),geometry_rng);
    for(size_t j=0;j<n;++j)rank[order[j]]=static_cast<int>(j);
    auto covariance=[&](int a,int b){double h=0;for(int d=0;d<3;++d){double v=x[a][d]-x[b][d];h+=v*v;}return (1-nugget)*std::exp(-3*h)+(a==b?nugget:0);};
    std::vector<Eigen::Triplet<double>> entries;entries.reserve(n*static_cast<size_t>(k+1)*(k+1));
    Eigen::VectorXd latent(n);SparseLatentResult out;out.estimated_workspace_bytes=workspace;
    for(int i:order){
        auto candidates=tree.query(x[i],std::numeric_limits<double>::max()/4,std::min(static_cast<int>(n),std::max(256,k*8)));
        std::vector<int> parents;
        for(const auto& item:candidates)if(rank[item.index]<rank[i]){parents.push_back(item.index);if(static_cast<int>(parents.size())==k)break;}
        const int m=static_cast<int>(parents.size());Eigen::VectorXd beta(m);double variance=1,mean=0;
        if(m){Eigen::MatrixXd c(m,m);Eigen::VectorXd rhs(m);
            for(int a=0;a<m;++a){rhs[a]=covariance(i,parents[a]);for(int b=0;b<m;++b)c(a,b)=covariance(parents[a],parents[b]);}
            Eigen::LLT<Eigen::MatrixXd> factor(c);if(factor.info()!=Eigen::Success)throw std::runtime_error("Sparse local covariance failed");
            beta=factor.solve(rhs);double residual=(c*beta-rhs).norm()/std::max(rhs.norm(),1e-12);
            out.maximum_local_residual=std::max(out.maximum_local_residual,residual);
            if(!beta.allFinite()||residual>1e-8)throw std::runtime_error("Sparse local solve residual failed");
            variance-=rhs.dot(beta);for(int a=0;a<m;++a)mean+=beta[a]*latent[parents[a]];
        }
        if(!std::isfinite(variance)||variance<=0)throw std::runtime_error("Nonpositive sparse conditional variance");
        const double sd=std::sqrt(variance);latent[i]=mean+sd*truncated((lower[i]-mean)/sd,(upper[i]-mean)/sd,rng);
        std::vector<int> ids=parents;ids.push_back(i);Eigen::VectorXd coefficients(m+1);coefficients.head(m)=-beta;coefficients[m]=1;
        for(int a=0;a<=m;++a)for(int b=0;b<=m;++b)entries.emplace_back(ids[a],ids[b],coefficients[a]*coefficients[b]/variance);
    }
    Eigen::SparseMatrix<double> precision(n,n);precision.setFromTriplets(entries.begin(),entries.end());entries.clear();entries.shrink_to_fit();
    precision.makeCompressed();out.precision_nonzeros=precision.nonZeros();Eigen::VectorXd residual=precision*latent;
    for(int sweep=0;sweep<sweeps;++sweep){
        std::shuffle(order.begin(),order.end(),rng);
        for(int i:order){double diagonal=precision.coeff(i,i);double sd=1/std::sqrt(diagonal);double mean=latent[i]-residual[i]/diagonal;
            double value=mean+sd*truncated((lower[i]-mean)/sd,(upper[i]-mean)/sd,rng);
            value=std::max(std::nextafter(lower[i],upper[i]),std::min(std::nextafter(upper[i],lower[i]),value));
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite sparse latent state");
            double delta=value-latent[i];latent[i]=value;
            for(Eigen::SparseMatrix<double>::InnerIterator it(precision,i);it;++it)residual[it.row()]+=it.value()*delta;
        }
        if((sweep+1)%10==0)residual=precision*latent;
    }
    out.values.assign(latent.data(),latent.data()+n);return out;
}
} // namespace deeprbf
