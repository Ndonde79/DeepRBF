#include "deeprbf/implicit_model.hpp"
#include "deeprbf/kernels.hpp"
#include "deeprbf/mkl_solver.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
namespace deeprbf {
NativeTrendImplicitModel::NativeTrendImplicitModel():NativeTrendImplicitModel("gaussian",1.,1e-6){}
NativeTrendImplicitModel::NativeTrendImplicitModel(const std::string& k,double e,double r):kernel_(k),epsilon_(e),regularization_(r),use_trend_transform_(false),trend_rotation_{1,0,0,0,1,0,0,0,1},trend_scale_{1,1,1},trend_center_{0,0,0}{
    evaluate_kernel(k,0.,e);if(!std::isfinite(r)||r<0)throw std::invalid_argument("invalid regularization");
}
void NativeTrendImplicitModel::set_trend_transform(const std::array<double,9>& r,const Point3& s,const Point3& c){
    Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> rot(r.data());
    if(!rot.allFinite()||!(rot*rot.transpose()).isApprox(Eigen::Matrix3d::Identity(),1e-6))throw std::invalid_argument("rotation must be orthonormal");
    for(int k=0;k<3;++k)if(!std::isfinite(s[k])||s[k]<=0||!std::isfinite(c[k]))throw std::invalid_argument("invalid transform");
    trend_rotation_=r;trend_scale_=s;trend_center_=c;use_trend_transform_=true;fitted_=false;
}
void NativeTrendImplicitModel::clear_trend_transform(){use_trend_transform_=false;fitted_=false;}
bool NativeTrendImplicitModel::has_trend_transform()const{return use_trend_transform_;}
Point3 NativeTrendImplicitModel::apply_transform(const Point3& p)const{
    for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinate");
    if(!use_trend_transform_)return p;Point3 out{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i]+=trend_rotation_[i*3+j]*(p[j]-trend_center_[j])/trend_scale_[i];return out;
}
std::vector<Point3> NativeTrendImplicitModel::transform_points(const std::vector<Point3>& p)const{auto out=p;for(auto& v:out)v=apply_transform(v);return out;}
void NativeTrendImplicitModel::validate_weights(const std::vector<double>& w)const{for(double v:w)if(!std::isfinite(v)||v<=0)throw std::invalid_argument("weights must be finite and positive");}
void NativeTrendImplicitModel::fit_signed_constraints(const std::vector<Point3>& p,const std::vector<double>& v){fit_internal(p,v,nullptr);}
void NativeTrendImplicitModel::fit_signed_constraints(const std::vector<Point3>& p,const std::vector<double>& v,const std::vector<double>& w){fit_internal(p,v,&w);}
void NativeTrendImplicitModel::fit_internal(const std::vector<Point3>& pts,const std::vector<double>& vals,const std::vector<double>* w){
    fitted_=false;if(pts.empty()||pts.size()!=vals.size()||(w&&w->size()!=pts.size()))throw std::invalid_argument("constraint length mismatch");
    if(w)validate_weights(*w);for(double v:vals)if(!std::isfinite(v))throw std::invalid_argument("invalid value");
    centers_=transform_points(pts);size_t n=pts.size(),m=n+4;
    if(3.L*m*m*sizeof(double)>256.L*1024*1024)throw std::length_error("Weighted RBF exceeds 256 MiB dense workspace limit");
    std::vector<double>A(m*m,0.),b(m,0.);
    for(size_t i=0;i<n;++i){b[i]=vals[i];for(size_t j=0;j<n;++j)A[i*m+j]=evaluate_kernel(kernel_,point_distance(centers_[i],centers_[j]),epsilon_);
        double wi=w?(*w)[i]:1.;A[i*m+i]+=regularization_/(wi*wi);A[i*m+n]=A[n*m+i]=1.;
        for(size_t k=0;k<3;++k)A[i*m+n+1+k]=A[(n+1+k)*m+i]=centers_[i][k];}
    if(!solve_dense_linear_system(A,b,static_cast<int>(m),solver_backend_))throw std::runtime_error("singular weighted RBF constraints");coefficients_=std::move(b);fitted_=true;
}
std::vector<double> NativeTrendImplicitModel::predict(const std::vector<Point3>& pts)const{
    if(!fitted_)throw std::runtime_error("model is not fitted");auto transformed=transform_points(pts);size_t n=centers_.size();std::vector<double>out(pts.size());
    for(size_t i=0;i<pts.size();++i){double v=coefficients_[n];for(size_t k=0;k<3;++k)v+=coefficients_[n+1+k]*transformed[i][k];for(size_t j=0;j<n;++j)v+=coefficients_[j]*evaluate_kernel(kernel_,point_distance(transformed[i],centers_[j]),epsilon_);out[i]=v;}return out;
}
bool NativeTrendImplicitModel::is_fitted()const{return fitted_;}
int NativeTrendImplicitModel::n_centers()const{return static_cast<int>(centers_.size());}
std::string NativeTrendImplicitModel::solver_backend()const{return "weighted_augmented_"+solver_backend_;}
std::string NativeTrendImplicitModel::scalfmm_status()const{return "weighted trend model uses direct evaluation";}
std::string NativeTrendImplicitModel::kernel()const{return kernel_;}
double NativeTrendImplicitModel::epsilon()const{return epsilon_;}
double NativeTrendImplicitModel::regularization()const{return regularization_;}
}
