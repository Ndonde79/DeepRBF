#ifndef DEEPRBF_RBF_MODEL_HPP
#define DEEPRBF_RBF_MODEL_HPP

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace deeprbf {

class CompactNeighbourGrid;

class NativeExactRBFImplicitModel {
public:
    static void set_dense_direct_limit(int limit);
    static void force_iterative_solver(bool force);
    // 3.3.2: iterative solver controls (GMRES restart, total iteration budget, relative tolerance)
    static void set_iterative_solver(int restart, int max_iter, double tol);
    // 3.3.2: dense path memory budget in MB (the dense path holds about three (n+4)^2 double matrices)
    static void set_dense_memory_limit_mb(double megabytes);
    // 3.3.2: prediction backend "auto" | "direct" | "scalfmm"; "scalfmm" never falls back silently
    static void set_predict_backend(const std::string& backend);
    static std::string predict_backend();
    // 3.3.3: sparse direct path for compactly supported kernels (spherical). By default it is used when the kernel is
    // compact and n exceeds the dense direct limit (or the dense memory budget); force=true uses it for every fit of
    // a compact kernel. force_iterative_solver(true) still takes precedence.
    static void set_sparse_solver(bool force);
    static bool sparse_solver_forced();

    NativeExactRBFImplicitModel();
    NativeExactRBFImplicitModel(const std::string& kernel, double epsilon, double regularization);

    NativeExactRBFImplicitModel(const NativeExactRBFImplicitModel&) = delete;
    NativeExactRBFImplicitModel& operator=(const NativeExactRBFImplicitModel&) = delete;

    NativeExactRBFImplicitModel(NativeExactRBFImplicitModel&&) = default;
    NativeExactRBFImplicitModel& operator=(NativeExactRBFImplicitModel&&) = default;

    void fit(const std::vector<std::array<double, 3>>& centers,
             const std::vector<double>& values);

    void fit_with_reduction(const std::vector<std::array<double, 3>>& centers,
                            const std::vector<double>& values,
                            double fitting_accuracy = 1e-4,
                            double initial_ratio = 0.1,
                            double append_ratio = 0.05);

    std::vector<double> predict(const std::vector<std::array<double, 3>>& points) const;

    bool is_fitted() const;
    int n_centers() const;
    std::string solver_backend() const;
    std::string scalfmm_status() const;
    std::string last_predict_backend() const;
    // 3.3.3: last sparse fit: {nnz_lower, assembly_s, factor_s, solve_s, relative_residual}; empty otherwise
    std::vector<double> sparse_solve_info() const;
    std::string kernel() const;
    double epsilon() const;
    double regularization() const;

    // Polynomial drift appended to the RBF system: "linear" [1,x,y,z] (default, the 3.3.2 behaviour),
    // "constant" [1], or "none". "none" is accepted only for strictly positive definite kernels (gaussian,
    // inverse_multiquadric, spherical, exponential, gen_cauchy); "constant" is refused for cubic and thin_plate,
    // which need the linear drift. Changing the drift clears any fit.
    void set_drift(const std::string& drift);
    std::string drift() const;
    // Generalised Cauchy shape: only for gen_cauchy models; rewrites kernel() to "gen_cauchy:<alpha>:<power>".
    void set_kernel_shape(double alpha, double power);

private:
    double distance3(const std::array<double, 3>& a, const std::array<double, 3>& b) const;
    double kernel_eval(double r) const;

    void fit_dense(const std::vector<std::array<double, 3>>& centers,
                   const std::vector<double>& values);

    void fit_iterative(const std::vector<std::array<double, 3>>& centers,
                       const std::vector<double>& values);

    void fit_sparse_compact(const std::vector<std::array<double, 3>>& centers,
                            const std::vector<double>& values);

    std::string kernel_;
    int poly_dim_ = 4;   // 0 none, 1 constant, 4 linear
    double epsilon_;
    double regularization_;
    bool fitted_;
    std::vector<std::array<double, 3>> centers_;
    std::vector<double> weights_;
    std::string solver_backend_;
    std::string scalfmm_status_;
    mutable std::string last_predict_backend_;
    std::vector<double> sparse_info_;
    int drift_columns_ = 4;                              // 0, 1 or 4; the sparse path supports all three
    std::shared_ptr<const CompactNeighbourGrid> compact_grid_;   // set at fit for compact kernels

    static int dense_direct_limit_;
    static bool force_iterative_;
    static int gmres_restart_;
    static int gmres_max_iter_;
    static double gmres_tol_;
    static double dense_memory_mb_;
    static std::string predict_backend_;
    static bool force_sparse_;
};

} // namespace deeprbf

#endif // DEEPRBF_RBF_MODEL_HPP