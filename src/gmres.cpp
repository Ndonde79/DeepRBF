#include "deeprbf/gmres.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace deeprbf {
bool gmres_solve(int n,const ApplyOperatorFn& A,const std::vector<double>& b,std::vector<double>& x,
                 const GMRESConfig& cfg,std::string& status,const ApplyPreconditionerFn& M) {
    if(n<=0||b.size()!=static_cast<size_t>(n)||cfg.restart<=0||cfg.max_iter<=0||!std::isfinite(cfg.tol)||cfg.tol<=0||!A)
        throw std::invalid_argument("invalid GMRES parameters");
    using Vec=Eigen::VectorXd;using Mat=Eigen::MatrixXd;
    Vec rhs=Eigen::Map<const Vec>(b.data(),n);
    if(!rhs.allFinite())throw std::invalid_argument("non-finite GMRES rhs");
    if(x.empty())x.assign(n,0.);
    if(x.size()!=static_cast<size_t>(n))throw std::invalid_argument("GMRES initial vector size mismatch");
    Vec current=Eigen::Map<const Vec>(x.data(),n);
    if(!current.allFinite())throw std::invalid_argument("non-finite GMRES initial vector");
    auto apply=[&](const ApplyOperatorFn& fn,const Vec& in)->Vec{
        std::vector<double> input(in.data(),in.data()+n),out;
        fn(input,out); if(out.size()!=static_cast<size_t>(n))throw std::runtime_error("GMRES callback output size mismatch");
        Vec v=Eigen::Map<const Vec>(out.data(),n);if(!v.allFinite())throw std::runtime_error("non-finite GMRES callback output");return v;
    };
    double tolerance=cfg.tol*(rhs.norm()>0?rhs.norm():1.);
    int iterations=0;
    while(iterations<cfg.max_iter){
        Vec r=rhs-apply(A,current); double beta=r.norm();
        if(beta<=tolerance){x.assign(current.data(),current.data()+n);status="converged; true residual="+std::to_string(beta);return true;}
        int kmax=std::min({cfg.restart,n,cfg.max_iter-iterations});
        Mat V=Mat::Zero(n,kmax+1),Z=Mat::Zero(n,kmax),H=Mat::Zero(kmax+1,kmax);
        V.col(0)=r/beta;Vec trial=current;
        for(int j=0;j<kmax;++j){
            Z.col(j)=M?apply(M,V.col(j)):Vec(V.col(j));
            Vec w=apply(A,Z.col(j));
            // Two-pass modified Gram-Schmidt avoids loss of orthogonality.
            for(int pass=0;pass<2;++pass)for(int i=0;i<=j;++i){double h=V.col(i).dot(w);H(i,j)+=h;w-=h*V.col(i);}
            H(j+1,j)=w.norm();bool breakdown=H(j+1,j)<=1e-14;
            if(!breakdown)V.col(j+1)=w/H(j+1,j);
            Vec g=Vec::Zero(j+2);g[0]=beta;
            Vec y=H.topLeftCorner(j+2,j+1).colPivHouseholderQr().solve(g);
            ++iterations;
            // 3.3.2: the Arnoldi residual estimate ||g - H y|| is checked every iteration; the true residual (one
            // extra operator product) only when the estimate passes, at a breakdown or at the end of a cycle.
            const double estimate=(g-H.topLeftCorner(j+2,j+1)*y).norm();
            const bool check=estimate<=tolerance||breakdown||j+1==kmax;
            if(!check)continue;
            trial=current+Z.leftCols(j+1)*y;
            double residual=(rhs-apply(A,trial)).norm();
            if(residual<=tolerance){x.assign(trial.data(),trial.data()+n);status="converged; true residual="+std::to_string(residual);return true;}
            if(breakdown){x.assign(trial.data(),trial.data()+n);status="Arnoldi breakdown without convergence";return false;}
        }
        current=trial;
    }
    x.assign(current.data(),current.data()+n);status="iteration limit; true residual="+std::to_string((rhs-apply(A,current)).norm());return false;
}
}
