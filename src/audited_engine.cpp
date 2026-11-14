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
#include <set>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace deeprbf {
namespace {
void finite_point(const Point3& p) {for(double x:p)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite coordinate");}
void positive(const Point3& p) {finite_point(p);for(double x:p)if(x<=0)throw std::invalid_argument("Positive finite scale required");}
void valid_axes(const Axes3& a) {
    Eigen::Matrix3d m;for(int i=0;i<3;++i){finite_point(a[i]);for(int j=0;j<3;++j)m(i,j)=a[i][j];}
    if(!(m*m.transpose()).isApprox(Eigen::Matrix3d::Identity(),1e-9)||std::abs(m.determinant()-1)>1e-9)throw std::invalid_argument("Right-handed orthonormal axes required");
}
Point3 transform(const Point3& p,const Axes3& axes,const Point3& scales) {
    Point3 q{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)q[i]+=axes[i][j]*p[j]/scales[i];return q;
}
double distance(const Point3&a,const Point3&b) {return std::hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);}
void input(const std::vector<Point3>& xyz,const std::vector<int>& holes,const std::vector<int>& domains) {
    if(xyz.size()!=holes.size()||xyz.size()!=domains.size())throw std::invalid_argument("Sample metadata size mismatch");
    for(const auto& p:xyz)finite_point(p);
}
class SearchIndex {
    const std::vector<int>& holes_;const std::vector<int>& domains_;NeighborhoodSpec spec_;
    KDTree3 hash_;Point3 origin_{};
public:
    SearchIndex(const std::vector<Point3>& xyz,const std::vector<int>& holes,const std::vector<int>& domains,const NeighborhoodSpec& spec):holes_(holes),domains_(domains),spec_(spec) {
        spec.validate();input(xyz,holes,domains);if(!xyz.empty())origin_=xyz.front();
        hash_.reserve(xyz.size());for(const auto& p:xyz){Point3 d{};for(int j=0;j<3;++j)d[j]=p[j]-origin_[j];hash_.add(transform(d,spec.axes,spec.radii),0);}
        hash_.build();
    }
    NeighborhoodSelection query(const Point3& target,int domain)const {
        finite_point(target);Point3 d{};for(int j=0;j<3;++j)d[j]=target[j]-origin_[j];const auto q=transform(d,spec_.axes,spec_.radii);
        // Metadata quotas are applied after candidate selection. Candidate budget
        // is explicit, so insufficient distinct-hole support never passes silently.
        auto near=hash_.query(q,1.,spec_.restrict_domain?std::numeric_limits<int>::max():spec_.candidate_neighbors);std::map<int,int> per_hole;std::array<int,8> octants{};NeighborhoodSelection out;
        for(const auto& n:near){int i=n.index;if(spec_.restrict_domain&&domains_[i]!=domain)continue;
            const auto& p=hash_.points[i];int octant=(p[0]>=q[0]?1:0)+(p[1]>=q[1]?2:0)+(p[2]>=q[2]?4:0);
            if(spec_.max_per_hole>0&&per_hole[holes_[i]]>=spec_.max_per_hole)continue;
            if(spec_.max_per_octant>0&&octants[octant]>=spec_.max_per_octant)continue;
            out.indices.push_back(i);++per_hole[holes_[i]];++octants[octant];if(static_cast<int>(out.indices.size())==spec_.max_neighbors)break;
        }
        for(const auto& entry:per_hole)if(entry.second>0)++out.n_holes;
        out.supported=static_cast<int>(out.indices.size())>=spec_.min_neighbors&&out.n_holes>=spec_.min_holes;return out;
    }
};
std::vector<Point3> quadrature(const RotatedSupport& block) {
    finite_point(block.center);finite_point(block.size);valid_axes(block.axes);size_t n=1;
    for(int j=0;j<3;++j){if(block.size[j]<0||block.discretization[j]<1||block.discretization[j]>32)throw std::invalid_argument("Invalid support size/discretization (1..32 required)");n*=block.discretization[j];}
    if(n>4096)throw std::invalid_argument("Support quadrature exceeds 4096 points");
    std::vector<Point3> out;out.reserve(n);
    for(int k=0;k<block.discretization[2];++k)for(int j=0;j<block.discretization[1];++j)for(int i=0;i<block.discretization[0];++i){
        Point3 local{((i+.5)/block.discretization[0]-.5)*block.size[0],((j+.5)/block.discretization[1]-.5)*block.size[1],((k+.5)/block.discretization[2]-.5)*block.size[2]},p=block.center;
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)p[b]+=local[a]*block.axes[a][b];out.push_back(p);
    }return out;
}
}
void CovarianceSpec::validate()const {
    positive(ranges);valid_axes(axes);
    if(range_convention!="length_scale"&&range_convention!="practical")throw std::invalid_argument("Unknown covariance range convention");
    if(family!="spherical"&&family!="exponential"&&family!="gaussian")throw std::invalid_argument("Unknown covariance family");
    if(!std::isfinite(sill)||!std::isfinite(nugget)||sill<=0||nugget<0||nugget>sill)throw std::invalid_argument("Invalid total sill/nugget");
}
double CovarianceSpec::covariance(const Point3& a,const Point3& b,bool same_observation)const {
    Point3 d{};for(int i=0;i<3;++i)d[i]=a[i]-b[i];const auto q=transform(d,axes,ranges);double h=std::hypot(q[0],q[1],q[2]);
    const double factor=range_convention=="practical"?3.:1.;
    double c=family=="exponential"?std::exp(-factor*h):family=="gaussian"?std::exp(-factor*h*h):(h>=1?0:1-1.5*h+.5*h*h*h);
    return (sill-nugget)*c+(same_observation?nugget:0);
}
void NeighborhoodSpec::validate()const {
    positive(radii);valid_axes(axes);
    if(max_neighbors<1||max_neighbors>512||min_neighbors<1||min_neighbors>max_neighbors||min_holes<1||min_holes>max_neighbors||max_per_hole<0||max_per_octant<0||candidate_neighbors<max_neighbors)throw std::invalid_argument("Invalid neighborhood limits");
}
NeighborhoodSelection select_neighborhood(const std::vector<Point3>& xyz,const std::vector<int>& holes,const std::vector<int>& domains,const Point3& target,int domain,const NeighborhoodSpec& spec){return SearchIndex(xyz,holes,domains,spec).query(target,domain);}

AuditedResult audited_ordinary_kriging(const std::vector<Point3>& xyz,const std::vector<double>& values,const std::vector<int>& holes,const std::vector<int>& domains,const std::vector<RotatedSupport>& targets,const CovarianceSpec& cov,const NeighborhoodSpec& search,double regularization,double tolerance,const std::vector<RotatedSupport>& observation_support) {
    cov.validate();search.validate();input(xyz,holes,domains);
    if(values.size()!=xyz.size()||!std::isfinite(regularization)||regularization<0||!std::isfinite(tolerance)||tolerance<=0)throw std::invalid_argument("Invalid values/solve controls");
    if(!observation_support.empty()&&observation_support.size()!=xyz.size())throw std::invalid_argument("One observation support per sample required");
    std::map<std::vector<double>,double> seen;std::vector<std::vector<Point3>> observations;observations.reserve(xyz.size());size_t observation_nodes=0;
    for(size_t i=0;i<xyz.size();++i){
        if(!std::isfinite(values[i]))throw std::invalid_argument("Nonfinite sample value");
        std::vector<double> key(xyz[i].begin(),xyz[i].end());
        if(observation_support.empty())observations.push_back({xyz[i]});
        else{
            const auto& support=observation_support[i];
            if(support.center!=xyz[i])throw std::invalid_argument("Observation support center differs from search coordinates");
            if(support.size!=Point3{0,0,0}&&cov.nugget!=0)throw std::invalid_argument("Finite observation support requires zero point nugget; a physical finite-support nugget model is not specified");
            observations.push_back(quadrature(support));
            if(observations.back().size()>64)throw std::invalid_argument("Observation support limited to 64 integration nodes");
            key.insert(key.end(),support.size.begin(),support.size.end());
            for(const auto& axis:support.axes)key.insert(key.end(),axis.begin(),axis.end());
            for(int order:support.discretization)key.push_back(order);
        }
        observation_nodes+=observations.back().size();if(observation_nodes>1000000)throw std::invalid_argument("Observation quadrature memory limit exceeded");
        auto [it,inserted]=seen.emplace(key,values[i]);if(!inserted&&it->second!=values[i])throw std::invalid_argument("Conflicting identical-support observations");
    }
    std::vector<std::vector<Point3>> supports;for(const auto& b:targets)supports.push_back(quadrature(b));
    SearchIndex index(xyz,holes,domains,search);AuditedResult out;double nan=std::numeric_limits<double>::quiet_NaN();
    out.estimate.assign(targets.size(),nan);out.variance.assign(targets.size(),nan);out.diagnostics.resize(targets.size());
    // One bounded cache per batch; full index vectors are keys (no hash collisions).
    using Factor=Eigen::FullPivLU<Eigen::MatrixXd>;
    struct Cached {Eigen::MatrixXd matrix;Factor factor;};std::map<std::vector<int>,Cached> cache;size_t cache_bytes=0;
    for(size_t t=0;t<targets.size();++t){auto near=index.query(targets[t].center,targets[t].domain);auto& diag=out.diagnostics[t];diag.n_samples=static_cast<int>(near.indices.size());diag.n_holes=near.n_holes;diag.regularization=regularization;
        if(!near.supported){diag.status="INSUFFICIENT_SUPPORT";continue;}
        const int m=diag.n_samples;diag.dimension=m+1;Eigen::VectorXd rhs=Eigen::VectorXd::Ones(m+1);
        const auto& points=supports[t];bool point=targets[t].size==Point3{0,0,0};
        size_t local_nodes=0;for(int index:near.indices)local_nodes+=observations[index].size();
        if(static_cast<long double>(local_nodes)*local_nodes+static_cast<long double>(local_nodes)*points.size()+static_cast<long double>(points.size())*points.size()>32000000.L){diag.status="QUADRATURE_WORK_LIMIT";continue;}
        for(int i=0;i<m;++i){double c=0;const auto& obs=observations[near.indices[i]];
            for(const auto& a:obs)for(const auto& p:points)c+=cov.covariance(a,p,point&&obs.size()==1&&a==p);
            rhs[i]=c/(obs.size()*points.size()*cov.sill);}
        auto it=cache.find(near.indices);
        if(it==cache.end()){
            // Account for varying neighborhood sizes, including keys and permutations.
            size_t entry_bytes=2ULL*(m+1)*(m+1)*sizeof(double)+4ULL*(m+1)*sizeof(int)+sizeof(Cached)+128;
            if(cache_bytes+entry_bytes>67108864ULL){cache.clear();cache_bytes=0;}Cached entry;entry.matrix=Eigen::MatrixXd::Zero(m+1,m+1);
            for(int i=0;i<m;++i){for(int j=0;j<m;++j){
                const auto& a=observations[near.indices[i]];const auto& b=observations[near.indices[j]];double covariance=0;
                for(const auto& u:a)for(const auto& v:b)covariance+=cov.covariance(u,v,i==j&&a.size()==1);
                entry.matrix(i,j)=covariance/(a.size()*b.size()*cov.sill);
            }entry.matrix(i,i)+=regularization/cov.sill;entry.matrix(i,m)=entry.matrix(m,i)=1;}
            entry.factor.compute(entry.matrix);it=cache.emplace(near.indices,std::move(entry)).first;cache_bytes+=entry_bytes;
        }
        const auto& matrix=it->second.matrix;const auto& factor=it->second.factor;diag.rank=static_cast<int>(factor.rank());diag.reciprocal_condition=factor.rcond();
        if(diag.rank!=m+1){diag.status="RANK_DEFICIENT";continue;}
        if(diag.reciprocal_condition<1e-12){diag.status="ILL_CONDITIONED";continue;}
        Eigen::VectorXd solution=factor.solve(rhs);diag.relative_residual=(matrix*solution-rhs).norm()/std::max(rhs.norm(),1e-300);
        if(!solution.allFinite()||!std::isfinite(diag.relative_residual)||diag.relative_residual>tolerance){diag.status="RESIDUAL_FAILURE";continue;}
        double estimate=0;for(int i=0;i<m;++i)estimate+=solution[i]*values[near.indices[i]];
        double cbb=0;for(size_t i=0;i<points.size();++i)for(size_t j=0;j<points.size();++j)cbb+=cov.covariance(points[i],points[j],point||i==j);
        cbb/=static_cast<double>(points.size())*points.size();double variance=cbb-cov.sill*(solution.head(m).dot(rhs.head(m))+solution[m]);
        if(!std::isfinite(estimate)||!std::isfinite(variance)||variance < -1e-9*cov.sill){diag.status="INVALID_VARIANCE";continue;}
        diag.minimum_weight=solution.head(m).minCoeff();diag.maximum_weight=solution.head(m).maxCoeff();
        diag.weight_sum=solution.head(m).sum();diag.weight_l1=solution.head(m).cwiseAbs().sum();
        for(int i=0;i<m;++i)if(solution[i]<-1e-12)++diag.negative_weights;
        out.estimate[t]=estimate;out.variance[t]=std::max(0.,variance);diag.success=true;diag.status=variance<0?"OK_ROUNDOFF_VARIANCE_TO_ZERO":"OK";
    }return out;
}

struct NativeCategoricalModel::Impl {
    std::vector<Point3> xyz;
    std::vector<int> labels,holes,categories,category_domains,domains,encoded;
    std::vector<double> support,prior;
    Point3 ranges,gate_scale;Axes3 axes;NeighborhoodSpec search;
    std::unique_ptr<SearchIndex> index;SpatialHash exact{1.};std::set<int> groups;
    Impl(const std::vector<Point3>& x,const std::vector<int>& l,const std::vector<int>& h,
         const std::vector<double>& w,const std::vector<int>& c,const std::vector<int>& cd,
         const Point3& r,const Axes3& a,const Point3& g,const NeighborhoodSpec& s):
         xyz(x),labels(l),holes(h),categories(c),category_domains(cd),support(w),ranges(r),gate_scale(g),axes(a),search(s){
    positive(ranges);positive(gate_scale);valid_axes(axes);search.validate();std::vector<int> domains(xyz.size(),0);input(xyz,holes,domains);
    const size_t nc=categories.size();if(xyz.empty()||nc<2||nc>512||labels.size()!=xyz.size()||support.size()!=xyz.size()||category_domains.size()!=nc||!std::is_sorted(categories.begin(),categories.end())||std::adjacent_find(categories.begin(),categories.end())!=categories.end())throw std::invalid_argument("Invalid categorical inputs");
    encoded.resize(xyz.size());std::map<int,double> totals;prior.assign(nc,.001);
    for(size_t i=0;i<xyz.size();++i){auto it=std::lower_bound(categories.begin(),categories.end(),labels[i]);if(it==categories.end()||*it!=labels[i]||!std::isfinite(support[i])||support[i]<=0)throw std::invalid_argument("Unknown category/invalid support");encoded[i]=static_cast<int>(it-categories.begin());domains[i]=category_domains[encoded[i]];totals[holes[i]]+=support[i];}
    if(search.restrict_domain)throw std::invalid_argument("Categorical domain gating requires unrestricted conditioning; target domains are not supplied");
    for(size_t i=0;i<xyz.size();++i)prior[encoded[i]]+=support[i]/totals[holes[i]];
    double total=std::accumulate(prior.begin(),prior.end(),0.);for(auto& v:prior)v/=total;
    this->domains=std::move(domains);index=std::make_unique<SearchIndex>(xyz,holes,this->domains,search);
    exact.reserve(xyz.size());for(const auto&p:xyz)exact.add(p,0);
    groups=std::set<int>(category_domains.begin(),category_domains.end());
    }
    NativeCategoricalResult predict(const std::vector<Point3>& targets,bool hierarchical,double mix)const {
    if(!std::isfinite(mix)||mix<0||mix>1)throw std::invalid_argument("Invalid categorical prior mixture");
    const size_t nc=categories.size();NativeCategoricalResult out;out.n_targets=static_cast<int>(targets.size());out.n_categories=static_cast<int>(nc);
    if(nc>512||targets.size()>static_cast<size_t>(std::numeric_limits<int>::max())||static_cast<long double>(targets.size())*nc>32000000.L)throw std::invalid_argument("Categorical output memory limit exceeded");
    for(const auto& p:targets){
        finite_point(p);
        for(double value:p)if(std::abs(value)>8e15)throw std::invalid_argument("Categorical target outside exact-index coordinate range");
        Point3 offset{};for(int j=0;j<3;++j)offset[j]=p[j]-xyz.front()[j];
        finite_point(transform(offset,search.axes,search.radii));
    }
    out.probabilities.resize(targets.size()*nc);out.support_holes.resize(targets.size());out.supported.resize(targets.size());out.exact_conditioned.resize(targets.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(std::int64_t target_index=0;target_index<static_cast<std::int64_t>(targets.size());++target_index){size_t t=static_cast<size_t>(target_index);auto near=index->query(targets[t],0);out.support_holes[t]=near.n_holes;out.supported[t]=near.supported;
        std::vector<double> p(nc),raw(nc);std::map<int,double> gates;
        if(near.supported){for(int i:near.indices){Point3 d{};for(int j=0;j<3;++j)d[j]=targets[t][j]-xyz[i][j];auto q=transform(d,axes,ranges);double h=0,hg=0;for(int j=0;j<3;++j){h+=q[j]*q[j];hg+=q[j]*q[j]/(gate_scale[j]*gate_scale[j]);}raw[encoded[i]]+=std::exp(-.5*h)*support[i];gates[category_domains[encoded[i]]]+=std::exp(-.5*hg)*support[i];}
            for(size_t j=0;j<nc;++j)raw[j]+=1e-12*prior[j];
            if(hierarchical){std::map<int,double> mass,conditional;for(size_t j=0;j<nc;++j){mass[category_domains[j]]+=prior[j];conditional[category_domains[j]]+=raw[j];}
                double gate_total=0;for(int group:groups){gates[group]+=1e-12*mass[group];gate_total+=gates[group];}
                for(size_t j=0;j<nc;++j)p[j]=gates[category_domains[j]]/gate_total*raw[j]/conditional[category_domains[j]];
            }else{double sum=std::accumulate(raw.begin(),raw.end(),0.);for(size_t j=0;j<nc;++j)p[j]=raw[j]/sum;}
        }else p=prior;
        for(size_t j=0;j<nc;++j)p[j]=(1-mix)*p[j]+mix*prior[j];
        auto coincident=exact.query(targets[t],1e-7,static_cast<int>(xyz.size()));
        if(!coincident.empty()){std::fill(p.begin(),p.end(),0.);double sum=0;for(const auto& n:coincident){p[encoded[n.index]]+=support[n.index];sum+=support[n.index];}for(auto& v:p)v/=sum;out.exact_conditioned[t]=true;out.supported[t]=true;}
        std::copy(p.begin(),p.end(),out.probabilities.begin()+t*nc);
    }return out;
    }
};
NativeCategoricalModel::NativeCategoricalModel(const std::vector<Point3>& x,const std::vector<int>& labels,const std::vector<int>& holes,const std::vector<double>& support,const std::vector<int>& categories,const std::vector<int>& category_domains,const Point3& ranges,const Axes3& axes,const Point3& gate_scale,const NeighborhoodSpec& search):impl_(std::make_unique<Impl>(x,labels,holes,support,categories,category_domains,ranges,axes,gate_scale,search)){}
NativeCategoricalModel::~NativeCategoricalModel()=default;
NativeCategoricalResult NativeCategoricalModel::predict(const std::vector<Point3>& targets,bool hierarchical,double mix)const{return impl_->predict(targets,hierarchical,mix);}
NativeCategoricalResult categorical_rbf_native(const std::vector<Point3>& x,const std::vector<int>& labels,const std::vector<int>& holes,const std::vector<double>& support,const std::vector<int>& categories,const std::vector<int>& category_domains,const std::vector<Point3>& targets,const Point3& ranges,const Axes3& axes,const Point3& gate_scale,const NeighborhoodSpec& search,bool hierarchical,double mix){
    return NativeCategoricalModel(x,labels,holes,support,categories,category_domains,ranges,axes,gate_scale,search).predict(targets,hierarchical,mix);
}
NativeGaussianResult simulate_gaussian_native(const std::vector<Point3>& xyz,const std::vector<double>& values,const std::vector<Point3>& targets,const CovarianceSpec& cov,int realizations,unsigned long long seed,double mean,int dense_limit){
    cov.validate();if(xyz.size()!=values.size()||realizations<1||!std::isfinite(mean)||dense_limit<1||dense_limit>4096)throw std::invalid_argument("Invalid Gaussian simulation controls");
    std::map<Point3,double> observed;for(size_t i=0;i<xyz.size();++i){finite_point(xyz[i]);if(!std::isfinite(values[i]))throw std::invalid_argument("Nonfinite observed value");auto [it,inserted]=observed.emplace(xyz[i],values[i]);if(!inserted&&it->second!=values[i])throw std::invalid_argument("Conflicting coincident Gaussian values");}
    std::map<Point3,int> unique;std::vector<Point3> q,x;std::vector<int> mapping;std::vector<double> y;
    for(const auto& item:observed){x.push_back(item.first);y.push_back(item.second);}
    for(const auto& p:targets){finite_point(p);auto [it,inserted]=unique.emplace(p,static_cast<int>(q.size()));if(inserted)q.push_back(p);mapping.push_back(it->second);}
    const int n=static_cast<int>(x.size()),m=static_cast<int>(q.size());
    if(n>dense_limit||m>dense_limit||targets.size()>static_cast<size_t>(std::numeric_limits<int>::max())||static_cast<long double>(targets.size())*realizations>16000000.L||static_cast<long double>(targets.size())*targets.size()>8000000.L)throw std::invalid_argument("Gaussian simulation exceeds explicit dense/output memory limits");
    NativeGaussianResult out;out.n_targets=static_cast<int>(targets.size());out.n_realizations=realizations;
    if(m==0)return out;
    Eigen::VectorXd mu=Eigen::VectorXd::Constant(m,mean);Eigen::MatrixXd conditional(m,m);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)conditional(i,j)=cov.covariance(q[i],q[j],i==j)/cov.sill;
    if(n){Eigen::MatrixXd dd(n,n),qd(m,n);for(int i=0;i<n;++i)for(int j=0;j<n;++j)dd(i,j)=cov.covariance(x[i],x[j],i==j)/cov.sill;
        for(int i=0;i<m;++i)for(int j=0;j<n;++j)qd(i,j)=cov.covariance(q[i],x[j],q[i]==x[j])/cov.sill;
        Eigen::LLT<Eigen::MatrixXd> factor(dd);if(factor.info()!=Eigen::Success)throw std::runtime_error("Gaussian conditioning covariance is not positive definite");
        Eigen::VectorXd centered(n);for(int i=0;i<n;++i)centered[i]=y[i]-mean;
        Eigen::VectorXd weights=factor.solve(centered);Eigen::MatrixXd correction=factor.solve(qd.transpose());
        out.conditioning_residual=std::max((dd*weights-centered).norm()/std::max(1.,centered.norm()),(dd*correction-qd.transpose()).norm()/std::max(1.,qd.norm()));
        if(!weights.allFinite()||!correction.allFinite()||out.conditioning_residual>1e-9)throw std::runtime_error("Gaussian conditioning residual failed");
        mu+=qd*weights;conditional-=qd*correction;
    }
    conditional=(.5*(conditional+conditional.transpose())).eval();Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(conditional);
    if(eig.info()!=Eigen::Success)throw std::runtime_error("Conditional covariance eigensolve failed");
    out.minimum_conditional_eigenvalue=eig.eigenvalues().minCoeff()*cov.sill;
    if(eig.eigenvalues().minCoeff() < -1e-9)throw std::runtime_error("Conditional Gaussian covariance is indefinite");
    Eigen::MatrixXd root=eig.eigenvectors()*eig.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal()*std::sqrt(cov.sill);
    out.simulations.resize(targets.size()*realizations);out.conditional_mean.resize(targets.size());out.conditional_covariance.resize(targets.size()*targets.size());
    for(size_t i=0;i<targets.size();++i){out.conditional_mean[i]=mu[mapping[i]];for(size_t j=0;j<targets.size();++j)out.conditional_covariance[i*targets.size()+j]=conditional(mapping[i],mapping[j])*cov.sill;}
    for(int r=0;r<realizations;++r){std::mt19937_64 rng(seed+104729ULL*r);std::normal_distribution<double> normal;Eigen::VectorXd z(m);for(int j=0;j<m;++j)z[j]=normal(rng);Eigen::VectorXd draw=mu+root*z;
        for(size_t i=0;i<targets.size();++i){auto it=observed.find(targets[i]);out.simulations[i*realizations+r]=it==observed.end()?draw[mapping[i]]:it->second;}
    }return out;
}
} // namespace deeprbf
