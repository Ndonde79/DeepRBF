#include "deeprbf/mkl_solver.hpp"
#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(DEEPRBF_HAS_MKL) && DEEPRBF_HAS_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#endif

namespace deeprbf {

namespace {

bool solve_dense_fallback(std::vector<double>& A,
                          std::vector<double>& b,
                          int n) {
    if (n <= 0) return false;
    if (static_cast<int>(A.size()) != n * n) return false;
    if (static_cast<int>(b.size()) != n) return false;

    std::vector<double> work_A = A;
    std::vector<double> work_b = b;

    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double max_abs = std::abs(work_A[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                                         static_cast<std::size_t>(k)]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::abs(
                work_A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(k)]);
            if (v > max_abs) { max_abs = v; pivot = i; }
        }
        if (max_abs < 1.0e-14) return false;

        if (pivot != k) {
            for (int j = k; j < n; ++j) {
                std::swap(
                    work_A[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(j)],
                    work_A[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(j)]);
            }
            std::swap(work_b[static_cast<std::size_t>(k)],
                      work_b[static_cast<std::size_t>(pivot)]);
        }

        const double diag =
            work_A[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                   static_cast<std::size_t>(k)];

        for (int i = k + 1; i < n; ++i) {
            const double factor =
                work_A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(k)] / diag;
            work_A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                   static_cast<std::size_t>(k)] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                work_A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(j)] -=
                    factor *
                    work_A[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(j)];
            }
            work_b[static_cast<std::size_t>(i)] -=
                factor * work_b[static_cast<std::size_t>(k)];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        double sum = work_b[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= work_A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                          static_cast<std::size_t>(j)] *
                   work_b[static_cast<std::size_t>(j)];
        }
        const double diag =
            work_A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                   static_cast<std::size_t>(i)];
        if (std::abs(diag) < 1.0e-14) return false;
        work_b[static_cast<std::size_t>(i)] = sum / diag;
    }

    b = std::move(work_b);
    return true;
}

} // namespace

bool solve_dense_linear_system(std::vector<double>& A,
                               std::vector<double>& b,
                               int n,
                               std::string& backend_used) {
    if (n <= 0) { backend_used = "invalid_dimension"; return false; }
    if (A.size() != static_cast<size_t>(n) * static_cast<size_t>(n)) {
        backend_used = "matrix_size_mismatch"; return false;
    }
    if (static_cast<int>(b.size()) != n) {
        backend_used = "rhs_size_mismatch"; return false;
    }
    for(double v:A)if(!std::isfinite(v)){backend_used="nonfinite_matrix";return false;}
    for(double v:b)if(!std::isfinite(v)){backend_used="nonfinite_rhs";return false;}
    const std::vector<double> original_rhs=b;
    auto valid_solution=[&](const std::vector<double>& solution){
        long double sum=0,den=0;
        for(int i=0;i<n;++i){long double r=-original_rhs[i];if(!std::isfinite(solution[i]))return false;
            for(int j=0;j<n;++j)r+=static_cast<long double>(A[static_cast<size_t>(i)*n+j])*solution[j];sum+=r*r;den+=static_cast<long double>(original_rhs[i])*original_rhs[i];}
        return std::sqrt(sum)<=1e-8L*std::max(1.L,std::sqrt(den));
    };

#if defined(DEEPRBF_HAS_MKL) && DEEPRBF_HAS_MKL
    try {
        std::vector<MKL_INT> ipiv(static_cast<std::size_t>(n), 0);
        const MKL_INT N = static_cast<MKL_INT>(n);
        const MKL_INT NRHS = 1;

        std::vector<double> A_col_major(
            static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A_col_major[static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
                            static_cast<std::size_t>(i)] =
                    A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                      static_cast<std::size_t>(j)];
            }
        }

        std::vector<double> b_copy = b;
        const MKL_INT info = LAPACKE_dgesv(
            LAPACK_COL_MAJOR, N, NRHS,
            A_col_major.data(), N,
            ipiv.data(), b_copy.data(), N);

        if (info == 0) {
            if(!valid_solution(b_copy)){backend_used="mkl_residual_failure";return false;}
            b = std::move(b_copy);
            backend_used = "mkl_lapacke_dgesv";
            return true;
        }
        backend_used = "mkl_lapacke_dgesv_failed_info=" + std::to_string(info);
    } catch (const std::exception& e) {
        backend_used = "mkl_exception_" + std::string(e.what());
    } catch (...) {
        backend_used = "mkl_unknown_exception";
    }
#endif

    using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    Eigen::Map<const RowMatrix> matrix(A.data(),n,n);
    // 3.3.4: partial-pivot LU first (O(n^3/3) flops, blocked and OpenMP-parallel in Eigen, the same
    // factorisation as LAPACK dgesv); full-pivot LU only if the partial-pivot residual check fails.
    {
        Eigen::PartialPivLU<Eigen::MatrixXd> factor(matrix);
        Eigen::VectorXd solution=factor.solve(Eigen::Map<const Eigen::VectorXd>(b.data(),n));
        std::vector<double> candidate(solution.data(),solution.data()+n);
        if(solution.allFinite()&&valid_solution(candidate)){b=std::move(candidate);backend_used="eigen_partial_pivot_lu";return true;}
    }
    Eigen::FullPivLU<Eigen::MatrixXd> factor(matrix);
    if(factor.rank()!=n){backend_used="eigen_rank_deficient";return false;}
    Eigen::VectorXd solution=factor.solve(Eigen::Map<const Eigen::VectorXd>(b.data(),n));
    std::vector<double> candidate(solution.data(),solution.data()+n);
    const bool ok = valid_solution(candidate);
    if(ok)b=std::move(candidate);
    backend_used = ok ? "eigen_full_pivot_lu" : "eigen_residual_failure";
    return ok;
}

bool MKLSolver::solve(std::vector<double>& A,
                      std::vector<double>& b,
                      int n,
                      std::string& backend_used) const {
    return solve_dense_linear_system(A, b, n, backend_used);
}

} // namespace deeprbf
