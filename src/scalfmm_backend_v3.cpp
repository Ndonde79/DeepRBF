
#include <scalfmm/algorithms/fmm.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/interpolation/interpolation.hpp>
#include <scalfmm/operators/fmm_operators.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/for_each.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/tree/leaf_view.hpp>


#include "deeprbf/detail/scalfmm_backend.hpp"
#include <cmath>
#include <limits>
#include <chrono>
#include <stdexcept>
struct GaussianKernel {
    static constexpr auto homogeneity_tag = scalfmm::matrix_kernels::homogeneity::non_homogenous;
    static constexpr auto symmetry_tag = scalfmm::matrix_kernels::symmetry::symmetric;
    static constexpr size_t km=1, kn=1;
    static constexpr int separation_criterion=1;
    template<class T> using matrix_type=std::array<T,1>;
    template<class T> using vector_type=std::array<T,1>;
    std::string name() const { return "DeepRBF Gaussian exp(-r^2)"; }
    template<class T> auto mutual_coefficient() const { return vector_type<T>{T(1)}; }
    template<class P, class Q> auto evaluate(const P& x, const Q& y) const {
        using T=std::decay_t<decltype(x[0]-y[0])>;
        T r2(0); for(size_t k=0;k<3;++k) { auto d=x[k]-y[k]; r2+=d*d; }
        using std::exp;
        return matrix_type<T>{exp(-r2)};
    }
    template<class P, class Q> auto operator()(const P& x, const Q& y) const { return evaluate(x,y); }
};


namespace deeprbf::detail {
class RecoveredScalFMM final : public IScalFMMBackend {
    std::vector<std::array<double,3>> centers;
    double epsilon=1.;int order=6,tree_height=4;
    ScalFMMBackendReport info;
public:
    void setup(const std::vector<std::array<double,3>>& x,const std::string& kernel,double e,const ScalFMMBackendOptions& o)override {
        if(kernel!="gaussian")throw std::invalid_argument("Recovered ScalFMM backend supports Gaussian only; use direct for other kernels");
        if(x.empty()||!std::isfinite(e)||e<=0)throw std::invalid_argument("invalid ScalFMM source/epsilon");
        for(const auto& p:x)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite sources");
        if(o.interpolation_order<3||o.interpolation_order>8)throw std::invalid_argument("ScalFMM order must be 3..8");
        tree_height=o.tree_height?o.tree_height:4;
        if(tree_height<3||tree_height>6)throw std::invalid_argument("ScalFMM tree height must be 3..6");
        order=o.interpolation_order;centers=x;epsilon=e;
        info.compiled=true;info.backend="scalfmm3_chebyshev_dense_sequential";info.message="Gaussian with legacy distance/epsilon convention; sequential FMM";
        info.tree_height=tree_height;info.interpolation_order=order;info.source_count=x.size();
    }
    void multiply(const std::vector<double>& w,std::vector<double>& y)override {evaluate(centers,w,y);}
    void evaluate(const std::vector<std::array<double,3>>& targets,const std::vector<double>& weights,std::vector<double>& result)override {
        if(weights.size()!=centers.size())throw std::invalid_argument("ScalFMM weight count mismatch");
        for(double w:weights)if(!std::isfinite(w))throw std::invalid_argument("non-finite weights");
        result.assign(targets.size(),0.);if(targets.empty())return;
        auto started=std::chrono::steady_clock::now();
    using namespace scalfmm;
    using Particle=container::particle<double,3,double,1,double,1,size_t>;
    using Point=Particle::position_type;
    using Box=component::box<Point>;
    using Interp=interpolation::interpolator<double,3,GaussianKernel,options::chebyshev_<options::dense_>>;
    using Leaf=component::leaf_view<Particle>;
    using Cell=component::cell<typename Interp::storage_type>;
    using Tree=component::group_tree_view<Cell,Leaf,Box>;
    std::vector<Particle> sources(centers.size()), targets_vec(targets.size());
    Point lo{}, hi{}, middle{};
    for(size_t k=0;k<3;++k) { lo[k]=std::numeric_limits<double>::infinity(); hi[k]=-lo[k]; }
    auto fill=[&](const std::vector<std::array<double,3>>& m, auto& particles, bool source) {
        for(size_t i=0;i<m.size();++i) {
            auto& p=particles[i];
            for(size_t k=0;k<3;++k) {
                const double v=m[i][k]/epsilon;
                if(!std::isfinite(v)) throw std::runtime_error("transformed coordinates overflow");
                p.position()[k]=v; lo[k]=std::min(lo[k],v); hi[k]=std::max(hi[k],v);
            }
            p.inputs()[0]=source?weights[i]:0.; p.outputs()[0]=0.; p.variables(i);
        }
    };
    fill(centers,sources,true); fill(targets,targets_vec,false);
    double width=0.;
    for(size_t k=0;k<3;++k) { width=std::max(width,hi[k]-lo[k]); middle[k]=lo[k]/2.+hi[k]/2.; }
    width=std::max(width*1.00001,1e-8);
    if(!std::isfinite(width)) throw std::runtime_error("bounding box overflow");
    Box box(width,middle);
    Tree source_tree(tree_height,order,box,32,32,sources,false);
    Tree target_tree(tree_height,order,box,32,32,targets_vec,false);
    Interp interp(order,tree_height,width);
    operators::near_field_operator<GaussianKernel> near_field(false);
    operators::far_field_operator<Interp> far_field(interp);
    operators::fmm_operators<decltype(near_field),decltype(far_field)> ops(near_field,far_field);
    list::sequential::build_interaction_lists(source_tree,target_tree,ops);
    algorithms::sequential::sequential(source_tree,target_tree,ops,algorithms::all);
    component::for_each_leaf(std::cbegin(target_tree),std::cend(target_tree),[&](const auto& leaf) {
        for(auto ref:leaf) {
            const auto p=typename Leaf::const_proxy_type(ref);
            size_t i=std::get<0>(p.variables());
            double value=p.outputs()[0];
            result[i]=value;
        }
    });

        info.executed=true;info.target_count=targets.size();
        info.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    }
    ScalFMMBackendReport report()const override{return info;}
};
std::unique_ptr<IScalFMMBackend> make_scalfmm_backend(){return std::make_unique<RecoveredScalFMM>();}
bool scalfmm_backend_compiled()noexcept{return true;}
std::string scalfmm_backend_build_status(){return "Recovered sequential ScalFMM Gaussian backend compiled";}
}
