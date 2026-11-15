#include "deeprbf/kernels.hpp"
#include "deeprbf/rbf_model.hpp"

#include "deeprbf/domain_decomposition.hpp"
#include "deeprbf/gmres.hpp"
#include "deeprbf/mkl_solver.hpp"
#include "deeprbf/operator.hpp"
#include "deeprbf/scalfmm_adapter.hpp"
#include "deeprbf/sparse_compact.hpp"
#include <Eigen/Dense>

#include <algorithm>
#include <cstdio>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace deeprbf {

namespace {

constexpr int DENSE_DIRECT_LIMIT = 8000;
constexpr int LOCAL_SUBDOMAIN_DIRECT_LIMIT = 600;
constexpr double LOCAL_DIAGONAL_BOOST_FACTOR = 1.0;

constexpr int DEFAULT_FINE_SUBDOMAIN_SIZE = 300;
constexpr double DEFAULT_OVERLAP_RATIO = 0.75;
constexpr double DEFAULT_COARSE_RATIO = 0.15;
constexpr int MIN_COARSE_SIZE = 20;

struct LocalPreconditionerBlock {
    std::vector<int> overlap_indices;
    std::vector<char> owns;              // 3.3.2: 1 where the overlap point belongs to this block's core

    std::vector<int> global_to_local;
    std::vector<double> A_factorized;
    std::shared_ptr<Eigen::FullPivLU<Eigen::MatrixXd>> factor;
    std::string backend;
    int local_n = 0;
    int local_dim = 0;
    bool valid = false;
};

struct CoarseLevelBlock {
    std::vector<int> indices;
    std::vector<double> A_factorized;
    std::vector<double> factorized_matrix;
    std::shared_ptr<Eigen::FullPivLU<Eigen::MatrixXd>> factor;
    int local_dim = 0;
    bool valid = false;
    bool is_recursive = false;
    std::unique_ptr<CoarseLevelBlock> next_level;
};

struct TwoLevelDDMConfig {
    int fine_subdomain_size = DEFAULT_FINE_SUBDOMAIN_SIZE;
    double overlap_ratio = DEFAULT_OVERLAP_RATIO;
    double coarse_ratio = DEFAULT_COARSE_RATIO;
    int max_levels = 2;
    bool use_direct_coarse = true;
};

// Polynomial drift basis [1, x, y, z] truncated to poly_dim (0 none, 1 constant, 4 linear).
inline void poly_basis(int poly_dim, const std::array<double, 3>& c, double* out) {
    if (poly_dim >= 1) out[0] = 1.0;
    if (poly_dim >= 4) { out[1] = c[0]; out[2] = c[1]; out[3] = c[2]; }
}

void split_kd_tree_local(
    std::vector<int>& current_indices,
    const std::vector<std::array<double, 3>>& centers,
    std::vector<std::vector<int>>& leaves,
    std::size_t target_size
) {
    if (current_indices.size() <= target_size || current_indices.size() <= 1U) {
        leaves.push_back(current_indices);
        return;
    }

    double min_pt[3] = {
        centers[static_cast<std::size_t>(current_indices[0])][0],
        centers[static_cast<std::size_t>(current_indices[0])][1],
        centers[static_cast<std::size_t>(current_indices[0])][2]
    };
    double max_pt[3] = {min_pt[0], min_pt[1], min_pt[2]};

    for (int idx : current_indices) {
        const auto& c = centers[static_cast<std::size_t>(idx)];
        for (int d = 0; d < 3; ++d) {
            min_pt[d] = std::min(min_pt[d], c[d]);
            max_pt[d] = std::max(max_pt[d], c[d]);
        }
    }

    double max_len = -1.0;
    int best_axis = 0;
    for (int d = 0; d < 3; ++d) {
        const double len = max_pt[d] - min_pt[d];
        if (len > max_len) { max_len = len; best_axis = d; }
    }

    std::sort(current_indices.begin(), current_indices.end(),
              [&centers, best_axis](int a, int b) {
                  return centers[static_cast<std::size_t>(a)][best_axis] <
                         centers[static_cast<std::size_t>(b)][best_axis];
              });

    const std::size_t mid = current_indices.size() / 2U;
    std::vector<int> left(current_indices.begin(),
                          current_indices.begin() + static_cast<std::ptrdiff_t>(mid));
    std::vector<int> right(current_indices.begin() + static_cast<std::ptrdiff_t>(mid),
                           current_indices.end());

    split_kd_tree_local(left, centers, leaves, target_size);
    split_kd_tree_local(right, centers, leaves, target_size);
}

std::vector<Subdomain> build_paper_subdomains(
    const std::vector<std::array<double, 3>>& centers,
    int target_size,
    double overlap_ratio
) {
    std::vector<Subdomain> subs;
    if (centers.empty()) return subs;

    std::vector<int> initial_indices(centers.size());
    for (std::size_t i = 0; i < centers.size(); ++i)
        initial_indices[i] = static_cast<int>(i);

    std::vector<std::vector<int>> leaves;
    split_kd_tree_local(initial_indices, centers, leaves,
                        static_cast<std::size_t>(target_size));

    // 3.3.2: overlap = the core plus its nearest neighbours to the core centroid, about (1 + overlap_ratio) x the
    // core size and never above the local direct limit. The 3.3.1 bounding-box expansion made most overlaps larger
    // than that limit on drillhole data, so those blocks were skipped and GMRES ran almost unpreconditioned.
    for (const auto& leaf_indices : leaves) {
        if (leaf_indices.empty()) continue;
        std::array<double, 3> cen{0.0, 0.0, 0.0};
        for (int idx : leaf_indices)
            for (int d = 0; d < 3; ++d) cen[d] += centers[static_cast<std::size_t>(idx)][d];
        for (int d = 0; d < 3; ++d) cen[d] /= static_cast<double>(leaf_indices.size());
        const std::size_t want = std::min<std::size_t>(
            centers.size(),
            std::min<std::size_t>(static_cast<std::size_t>(LOCAL_SUBDOMAIN_DIRECT_LIMIT) - 8U,
                                  static_cast<std::size_t>(std::ceil((1.0 + overlap_ratio) * leaf_indices.size()))));
        std::vector<std::pair<double, int>> dist;
        dist.reserve(centers.size());
        for (std::size_t i = 0; i < centers.size(); ++i) {
            const auto& p = centers[i];
            const double dx = p[0] - cen[0], dy = p[1] - cen[1], dz = p[2] - cen[2];
            dist.emplace_back(dx * dx + dy * dy + dz * dz, static_cast<int>(i));
        }
        const std::size_t k = std::max(want, leaf_indices.size());
        if (k < dist.size())
            std::nth_element(dist.begin(), dist.begin() + static_cast<std::ptrdiff_t>(k), dist.end());
        std::vector<int> ov;
        ov.reserve(k + leaf_indices.size());
        for (std::size_t i = 0; i < std::min(k, dist.size()); ++i) ov.push_back(dist[i].second);
        ov.insert(ov.end(), leaf_indices.begin(), leaf_indices.end());
        std::sort(ov.begin(), ov.end());
        ov.erase(std::unique(ov.begin(), ov.end()), ov.end());

        Subdomain sd;
        sd.indices = leaf_indices;
        sd.overlap_indices = std::move(ov);
        subs.push_back(std::move(sd));
    }

    return subs;
}

CoarseLevelBlock build_coarse_level(
    const std::vector<std::array<double, 3>>& centers,
    const std::string& kernel,
    double epsilon,
    double regularization,
    double coarse_ratio,
    int poly_dim = 4
) {
    CoarseLevelBlock coarse;

    const int n = static_cast<int>(centers.size());
    // 3.3.2: coarse level capped at 2,000 points (dense full-pivot LU cost grows with its cube).
    const int n_coarse = std::min(2000, std::max(MIN_COARSE_SIZE,
                                  static_cast<int>(n * coarse_ratio)));

    if (n_coarse >= n) { coarse.valid = false; return coarse; }

    coarse.indices.reserve(n_coarse);
    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    for (int i = 0; i < n_coarse; ++i) {
        int idx = static_cast<int>(static_cast<double>(i) * n / n_coarse);
        coarse.indices.push_back(all_indices[idx]);
    }

    const int dim = n_coarse + poly_dim;
    coarse.local_dim = dim;

    const std::size_t mat_size = static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim);
    coarse.A_factorized.assign(mat_size, 0.0);

    const ResolvedKernel kernel_eval = resolve_kernel(kernel, epsilon);

    auto distance3 = [](const std::array<double, 3>& a,
                        const std::array<double, 3>& b) -> double {
        const double dx = a[0] - b[0];
        const double dy = a[1] - b[1];
        const double dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    for (int i = 0; i < n_coarse; ++i) {
        const int gi = coarse.indices[static_cast<std::size_t>(i)];
        const auto& ci = centers[static_cast<std::size_t>(gi)];
        const std::size_t row_off = static_cast<std::size_t>(i) * static_cast<std::size_t>(dim);

        for (int j = 0; j < n_coarse; ++j) {
            const int gj = coarse.indices[static_cast<std::size_t>(j)];
            const auto& cj = centers[static_cast<std::size_t>(gj)];
            const double r = distance3(ci, cj);
            coarse.A_factorized[row_off + static_cast<std::size_t>(j)] = kernel_eval(r);
        }
        coarse.A_factorized[row_off + static_cast<std::size_t>(i)] += regularization;

        double pb[4] = {0.0, 0.0, 0.0, 0.0};
        poly_basis(poly_dim, ci, pb);
        for (int k = 0; k < poly_dim; ++k) {
            coarse.A_factorized[row_off + static_cast<std::size_t>(n_coarse + k)] = pb[k];
            coarse.A_factorized[static_cast<std::size_t>(n_coarse + k) * static_cast<std::size_t>(dim) +
                                static_cast<std::size_t>(i)] = pb[k];
        }
    }

    using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    Eigen::Map<const RowMatrix> matrix(coarse.A_factorized.data(),dim,dim);
    coarse.factor=std::make_shared<Eigen::FullPivLU<Eigen::MatrixXd>>(matrix);
    coarse.valid = coarse.factor->rank()==dim;
    return coarse;
}

bool solve_coarse_level(const CoarseLevelBlock& coarse,
                        std::vector<double>& rhs,
                        std::string& backend_used) {
    if (!coarse.valid) return false;
    const int dim = coarse.local_dim;
    Eigen::VectorXd b=Eigen::Map<const Eigen::VectorXd>(rhs.data(),dim);
    Eigen::VectorXd x=coarse.factor->solve(b);
    using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    Eigen::Map<const RowMatrix> matrix(coarse.A_factorized.data(),dim,dim);
    if(!x.allFinite()||(matrix*x-b).norm()>1e-8*std::max(1.,b.norm()))return false;
    rhs.assign(x.data(),x.data()+dim);backend_used="cached_eigen_full_pivot_lu";return true;
}

class TwoLevelDDMPreconditioner {
public:
    TwoLevelDDMPreconditioner(
        const std::vector<std::array<double, 3>>& centers,
        const std::string& kernel,
        double epsilon,
        double regularization,
        const TwoLevelDDMConfig& cfg,
        int poly_dim)
        : centers_(centers),
          kernel_(kernel),
          epsilon_(epsilon),
          regularization_(regularization),
          cfg_(cfg),
          poly_dim_(poly_dim) {

        const int n = static_cast<int>(centers.size());
        std::vector<Subdomain> subdomains = build_paper_subdomains(
            centers, cfg_.fine_subdomain_size, cfg_.overlap_ratio);

        local_blocks_.reserve(subdomains.size());
        for (const auto& sub : subdomains) {
            LocalPreconditionerBlock block;
            block.overlap_indices = sub.overlap_indices;
            block.local_n = static_cast<int>(block.overlap_indices.size());
            block.local_dim = block.local_n + poly_dim_;

            if (block.local_n <= 0 || block.local_n > LOCAL_SUBDOMAIN_DIRECT_LIMIT) {
                local_blocks_.push_back(std::move(block));
                continue;
            }

            {
                std::vector<int> core = sub.indices;
                std::sort(core.begin(), core.end());
                block.owns.assign(block.overlap_indices.size(), 0);
                for (std::size_t i = 0; i < block.overlap_indices.size(); ++i)
                    block.owns[i] = std::binary_search(core.begin(), core.end(), block.overlap_indices[i]) ? 1 : 0;
            }
            block.global_to_local.assign(static_cast<std::size_t>(n), -1);
            for (int i = 0; i < block.local_n; ++i) {
                const int gi = block.overlap_indices[static_cast<std::size_t>(i)];
                if (gi >= 0 && gi < n) block.global_to_local[static_cast<std::size_t>(gi)] = i;
            }

            const int dim = block.local_dim;
            const std::size_t mat_size = static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim);
            std::vector<double> A_loc(mat_size, 0.0);

            const ResolvedKernel kernel_eval = resolve_kernel(kernel_, epsilon_);

            auto distance3 = [](const std::array<double, 3>& a,
                                const std::array<double, 3>& b) -> double {
                const double dx = a[0] - b[0];
                const double dy = a[1] - b[1];
                const double dz = a[2] - b[2];
                return std::sqrt(dx * dx + dy * dy + dz * dz);
            };

            for (int i = 0; i < block.local_n; ++i) {
                const int gi = block.overlap_indices[static_cast<std::size_t>(i)];
                const auto& ci = centers_[static_cast<std::size_t>(gi)];
                const std::size_t row_off = static_cast<std::size_t>(i) * static_cast<std::size_t>(dim);

                for (int j = 0; j < block.local_n; ++j) {
                    const int gj = block.overlap_indices[static_cast<std::size_t>(j)];
                    const auto& cj = centers_[static_cast<std::size_t>(gj)];
                    const double r = distance3(ci, cj);
                    A_loc[row_off + static_cast<std::size_t>(j)] = kernel_eval(r);
                }
                A_loc[row_off + static_cast<std::size_t>(i)] +=
                    LOCAL_DIAGONAL_BOOST_FACTOR * regularization_;

                double pb[4] = {0.0, 0.0, 0.0, 0.0};
                poly_basis(poly_dim_, ci, pb);
                for (int k = 0; k < poly_dim_; ++k) {
                    A_loc[row_off + static_cast<std::size_t>(block.local_n + k)] = pb[k];
                    A_loc[static_cast<std::size_t>(block.local_n + k) * static_cast<std::size_t>(dim) +
                          static_cast<std::size_t>(i)] = pb[k];
                }
            }

            block.A_factorized = std::move(A_loc);
            using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
            Eigen::Map<const RowMatrix> matrix(block.A_factorized.data(),dim,dim);
            block.factor=std::make_shared<Eigen::FullPivLU<Eigen::MatrixXd>>(matrix);
            block.valid = block.factor->rank()==dim;
            local_blocks_.push_back(std::move(block));
        }

        coarse_level_ = build_coarse_level(
            centers_, kernel_, epsilon_, regularization_,
            cfg_.coarse_ratio, poly_dim_);

        if (cfg_.max_levels > 1 && coarse_level_.valid &&
            static_cast<int>(coarse_level_.indices.size()) > MIN_COARSE_SIZE * 2) {
            TwoLevelDDMConfig rec_cfg = cfg_;
            rec_cfg.max_levels = cfg_.max_levels - 1;

            std::vector<std::array<double, 3>> coarse_centers;
            coarse_centers.reserve(coarse_level_.indices.size());
            for (int idx : coarse_level_.indices)
                coarse_centers.push_back(centers_[static_cast<std::size_t>(idx)]);

            auto next_coarse = std::make_unique<CoarseLevelBlock>(
                build_coarse_level(coarse_centers, kernel_, epsilon_, regularization_,
                                   cfg_.coarse_ratio, poly_dim_));
            coarse_level_.next_level = std::move(next_coarse);
            coarse_level_.is_recursive = true;
        }
    }

    void apply(const std::vector<double>& r_aug, std::vector<double>& z_aug) const {
        const int n = static_cast<int>(centers_.size());
        const int poly_dim = poly_dim_;
        const int aug_dim = n + poly_dim;

        if (static_cast<int>(r_aug.size()) != aug_dim) {
            throw std::runtime_error("TwoLevelDDMPreconditioner::apply: input size mismatch.");
        }

        z_aug.assign(static_cast<std::size_t>(aug_dim), 0.0);

        if (coarse_level_.valid) {
            const int n_coarse = static_cast<int>(coarse_level_.indices.size());
            const int coarse_dim = coarse_level_.local_dim;

            std::vector<double> r_coarse(static_cast<std::size_t>(coarse_dim), 0.0);
            for (int i = 0; i < n_coarse; ++i) {
                const int gi = coarse_level_.indices[static_cast<std::size_t>(i)];
                r_coarse[static_cast<std::size_t>(i)] = r_aug[static_cast<std::size_t>(gi)];
            }
            for (int k = 0; k < poly_dim; ++k)
                r_coarse[static_cast<std::size_t>(n_coarse + k)] = r_aug[static_cast<std::size_t>(n + k)];

            std::string backend_used;
            std::vector<double> z_coarse = r_coarse;
            const bool ok = solve_coarse_level(coarse_level_, z_coarse, backend_used);
            if (ok) {
                for (int i = 0; i < n_coarse; ++i) {
                    const int gi = coarse_level_.indices[static_cast<std::size_t>(i)];
                    z_aug[static_cast<std::size_t>(gi)] += z_coarse[static_cast<std::size_t>(i)];
                }
                for (int k = 0; k < poly_dim; ++k)
                    z_aug[static_cast<std::size_t>(n + k)] += z_coarse[static_cast<std::size_t>(n_coarse + k)];
            }
        }

        std::vector<int> hit_count(static_cast<std::size_t>(aug_dim), 0);

        for (const auto& block : local_blocks_) {
            if (!block.valid) continue;

            const int dim = block.local_dim;
            std::vector<double> rhs_loc(static_cast<std::size_t>(dim), 0.0);

            for (int i = 0; i < block.local_n; ++i) {
                const int gi = block.overlap_indices[static_cast<std::size_t>(i)];
                rhs_loc[static_cast<std::size_t>(i)] = r_aug[static_cast<std::size_t>(gi)];
            }
            for (int k = 0; k < poly_dim; ++k)
                rhs_loc[static_cast<std::size_t>(block.local_n + k)] = r_aug[static_cast<std::size_t>(n + k)];

            Eigen::VectorXd b=Eigen::Map<const Eigen::VectorXd>(rhs_loc.data(),dim);
            Eigen::VectorXd x=block.factor->solve(b);
            using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
            Eigen::Map<const RowMatrix> matrix(block.A_factorized.data(),dim,dim);
            if(!x.allFinite()||(matrix*x-b).norm()>1e-8*std::max(1.,b.norm()))continue;
            rhs_loc.assign(x.data(),x.data()+dim);

            // 3.3.2 restricted additive Schwarz: each point takes the local solution of the block that owns it.
            // The polynomial coefficients come from the coarse level only. 3.3.1 averaged every block's full
            // solution (and divided the coarse correction by the same counts), which weakened the preconditioner.
            for (int i = 0; i < block.local_n; ++i) {
                if (!block.owns[static_cast<std::size_t>(i)]) continue;
                const int gi = block.overlap_indices[static_cast<std::size_t>(i)];
                z_aug[static_cast<std::size_t>(gi)] += rhs_loc[static_cast<std::size_t>(i)];
                hit_count[static_cast<std::size_t>(gi)] += 1;
            }
        }

        for (int i = 0; i < n; ++i)
            if (hit_count[static_cast<std::size_t>(i)] == 0 && !coarse_level_.valid) z_aug[static_cast<std::size_t>(i)] = r_aug[static_cast<std::size_t>(i)];
        if (!coarse_level_.valid)
            for (int i = n; i < aug_dim; ++i) z_aug[static_cast<std::size_t>(i)] = r_aug[static_cast<std::size_t>(i)];
    }

    bool has_coarse_level() const { return coarse_level_.valid; }
    int n_local_blocks() const { return static_cast<int>(local_blocks_.size()); }

private:
    const std::vector<std::array<double, 3>>& centers_;
    std::string kernel_;
    double epsilon_;
    double regularization_;
    TwoLevelDDMConfig cfg_;
    int poly_dim_;   // 0 none, 1 constant, 4 linear

    std::vector<LocalPreconditionerBlock> local_blocks_;
    CoarseLevelBlock coarse_level_;
};

} // namespace

int NativeExactRBFImplicitModel::dense_direct_limit_ = DENSE_DIRECT_LIMIT;
bool NativeExactRBFImplicitModel::force_iterative_ = false;

void NativeExactRBFImplicitModel::set_dense_direct_limit(int limit) {
    if(limit<1)throw std::invalid_argument("Positive dense solver limit required");
    dense_direct_limit_ = limit;
}

void NativeExactRBFImplicitModel::force_iterative_solver(bool force) {
    force_iterative_ = force;
}

// 3.3.2 defaults. The 3.3.1 values (restart 40, 200 iterations, 256 MB) failed to converge on real drillhole
// data above about 3,300 points (19,000-point drillhole contact field, Nov 2026); a prototype on the same data converged in 100-600 iterations
// with restart 150.
int NativeExactRBFImplicitModel::gmres_restart_ = 150;
int NativeExactRBFImplicitModel::gmres_max_iter_ = 3000;
double NativeExactRBFImplicitModel::gmres_tol_ = 1.0e-6;
double NativeExactRBFImplicitModel::dense_memory_mb_ = 1024.0;
std::string NativeExactRBFImplicitModel::predict_backend_ = "auto";

void NativeExactRBFImplicitModel::set_iterative_solver(int restart, int max_iter, double tol) {
    if (restart < 1 || max_iter < 1 || !std::isfinite(tol) || !(tol > 0.0) || tol >= 1.0)
        throw std::invalid_argument("set_iterative_solver: restart and max_iter must be positive and 0 < tol < 1");
    gmres_restart_ = restart; gmres_max_iter_ = max_iter; gmres_tol_ = tol;
}

void NativeExactRBFImplicitModel::set_dense_memory_limit_mb(double megabytes) {
    if (!std::isfinite(megabytes) || !(megabytes > 0.0))
        throw std::invalid_argument("set_dense_memory_limit_mb: positive finite value required");
    dense_memory_mb_ = megabytes;
}

void NativeExactRBFImplicitModel::set_predict_backend(const std::string& backend) {
    if (backend != "auto" && backend != "direct" && backend != "scalfmm")
        throw std::invalid_argument("set_predict_backend: use auto, direct or scalfmm");
    predict_backend_ = backend;
}

std::string NativeExactRBFImplicitModel::predict_backend() { return predict_backend_; }

bool NativeExactRBFImplicitModel::force_sparse_ = false;
void NativeExactRBFImplicitModel::set_sparse_solver(bool force) { force_sparse_ = force; }
bool NativeExactRBFImplicitModel::sparse_solver_forced() { return force_sparse_; }

NativeExactRBFImplicitModel::NativeExactRBFImplicitModel()
    : kernel_("gaussian"),
      epsilon_(1.0),
      regularization_(1.0e-6),
      fitted_(false),
      solver_backend_("uninitialized"),
      scalfmm_status_(deeprbf_scalfmm_runtime_status()) {
}

NativeExactRBFImplicitModel::NativeExactRBFImplicitModel(
    const std::string& kernel, double epsilon, double regularization)
    : kernel_(kernel),
      epsilon_(epsilon),
      regularization_(regularization),
      fitted_(false),
      solver_backend_("uninitialized"),
      scalfmm_status_(deeprbf_scalfmm_runtime_status()) {
    evaluate_kernel(kernel,0.,epsilon);
    if (!std::isfinite(epsilon) || !(epsilon > 0.0))
        throw std::runtime_error("NativeExactRBFImplicitModel: epsilon must be > 0.");
    if (!std::isfinite(regularization) || regularization < 0.0)
        throw std::runtime_error("NativeExactRBFImplicitModel: regularization must be >= 0.");
}

double NativeExactRBFImplicitModel::distance3(
    const std::array<double, 3>& a, const std::array<double, 3>& b) const {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::hypot(dx,dy,dz);
}

double NativeExactRBFImplicitModel::kernel_eval(double r) const { return evaluate_kernel(kernel_, r, epsilon_); }

namespace {
int drift_poly_dim(const std::string& drift) {
    if (drift == "linear") return 4;
    if (drift == "constant") return 1;
    if (drift == "none") return 0;
    throw std::invalid_argument("set_drift: drift must be \"none\", \"constant\" or \"linear\", got \"" + drift + "\"");
}
// Throws when the kernel's (conditional) positive definiteness needs a richer drift than poly_dim provides.
void check_drift_for_kernel(const std::string& kernel, int poly_dim) {
    const KernelId id = kernel_id(kernel);
    if (poly_dim == 0 && !kernel_is_positive_definite(id))
        throw std::invalid_argument("set_drift(\"none\"): kernel \"" + kernel +
            "\" is not positive definite; use gaussian, inverse_multiquadric, spherical, exponential or gen_cauchy, "
            "or keep a polynomial drift");
    if (poly_dim < 4 && kernel_needs_linear_drift(id))
        throw std::invalid_argument("set_drift: kernel \"" + kernel +
            "\" is conditionally positive definite of order 2 and needs the \"linear\" drift");
}
}  // namespace

void NativeExactRBFImplicitModel::set_drift(const std::string& drift) {
    const int pd = drift_poly_dim(drift);
    check_drift_for_kernel(kernel_, pd);
    if (pd != poly_dim_) { fitted_ = false; weights_.clear(); }
    poly_dim_ = pd;
}

std::string NativeExactRBFImplicitModel::drift() const {
    return poly_dim_ == 0 ? "none" : (poly_dim_ == 1 ? "constant" : "linear");
}

void NativeExactRBFImplicitModel::set_kernel_shape(double alpha, double power) {
    if (!is_gen_cauchy_name(kernel_))
        throw std::invalid_argument("set_kernel_shape: only the gen_cauchy kernel has a shape (kernel is \"" + kernel_ + "\")");
    char buf[96];
    std::snprintf(buf, sizeof(buf), "gen_cauchy:%.17g:%.17g", alpha, power);
    const std::string k(buf);
    evaluate_kernel(k, 0.0, epsilon_);   // validates alpha > 0, 0 < power <= 2
    if (k != kernel_) { fitted_ = false; weights_.clear(); }
    kernel_ = k;
}

void NativeExactRBFImplicitModel::fit_dense(
    const std::vector<std::array<double, 3>>& centers,
    const std::vector<double>& values) {
    const int n = static_cast<int>(centers.size());
    const int poly_dim = poly_dim_;
    const int dim = n + poly_dim;

    std::vector<double> A(static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim), 0.0);
    std::vector<double> b(static_cast<std::size_t>(dim), 0.0);

    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = values[static_cast<std::size_t>(i)];
    const ResolvedKernel kern = resolve_kernel(kernel_, epsilon_);
    const double sys_reg = regularization_sign(kernel_id(kernel_)) * regularization_;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        const auto& ci = centers[static_cast<std::size_t>(i)];
        const std::size_t row_off = static_cast<std::size_t>(i) * static_cast<std::size_t>(dim);

        for (int j = 0; j < n; ++j) {
            const double r = distance3(ci, centers[static_cast<std::size_t>(j)]);
            A[row_off + static_cast<std::size_t>(j)] = kern(r);
        }
        A[row_off + static_cast<std::size_t>(i)] += sys_reg;

        double pb[4] = {0.0, 0.0, 0.0, 0.0};
        poly_basis(poly_dim, ci, pb);
        for (int k = 0; k < poly_dim; ++k) {
            A[row_off + static_cast<std::size_t>(n + k)] = pb[k];
            A[static_cast<std::size_t>(n + k) * static_cast<std::size_t>(dim) + static_cast<std::size_t>(i)] = pb[k];
        }
    }

    std::string backend_used;
    const bool ok = solve_dense_linear_system(A, b, dim, backend_used);
    if (!ok) throw std::runtime_error("fit_dense(): dense augmented solve failed.");

    weights_ = std::move(b);
    fitted_ = true;
    solver_backend_ = "dense_augmented_" + backend_used;
    scalfmm_status_ = deeprbf_scalfmm_runtime_status();
}

void NativeExactRBFImplicitModel::fit_iterative(
    const std::vector<std::array<double, 3>>& centers,
    const std::vector<double>& values) {
    const int n = static_cast<int>(centers.size());
    const int poly_dim = poly_dim_;
    const int aug_dim = n + poly_dim;

    RBFOperatorConfig op_cfg;
    op_cfg.kernel = kernel_;
    op_cfg.epsilon = epsilon_;
    op_cfg.regularization = regularization_sign(kernel_id(kernel_)) * regularization_;
    // 3.3.2: ScalFMM supports the Gaussian kernel only; other kernels use the exact OpenMP product directly
    // (3.3.1 routed them through the adapter's slower fallback path on every product).
    op_cfg.prefer_scalfmm = (kernel_ == "gaussian");

    RBFMatrixFreeOperator Aop(centers, op_cfg);

    std::vector<double> rhs(static_cast<std::size_t>(aug_dim), 0.0);
    for (int i = 0; i < n; ++i) rhs[static_cast<std::size_t>(i)] = values[static_cast<std::size_t>(i)];

    ApplyOperatorFn A = [this, &Aop, n, aug_dim, poly_dim](const std::vector<double>& x_aug,
                                                 std::vector<double>& y_aug) {
        if (static_cast<int>(x_aug.size()) != aug_dim)
            throw std::runtime_error("fit_iterative operator: input size mismatch.");

        y_aug.assign(static_cast<std::size_t>(aug_dim), 0.0);

        std::vector<double> lambda(x_aug.begin(),
                                   x_aug.begin() + static_cast<std::ptrdiff_t>(n));
        std::vector<double> y_lambda(static_cast<std::size_t>(n), 0.0);
        Aop.apply(lambda, y_lambda);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const auto& c = centers_[static_cast<std::size_t>(i)];
            double pb[4] = {0.0, 0.0, 0.0, 0.0};
            poly_basis(poly_dim, c, pb);
            double poly = 0.0;
            for (int k = 0; k < poly_dim; ++k) poly += x_aug[static_cast<std::size_t>(n + k)] * pb[k];
            y_aug[static_cast<std::size_t>(i)] = y_lambda[static_cast<std::size_t>(i)] + poly;
        }

        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : s0, s1, s2, s3) schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const double li = lambda[static_cast<std::size_t>(i)];
            const auto& c = centers_[static_cast<std::size_t>(i)];
            s0 += li;
            s1 += li * c[0];
            s2 += li * c[1];
            s3 += li * c[2];
        }

        const double s[4] = {s0, s1, s2, s3};
        for (int k = 0; k < poly_dim; ++k) y_aug[static_cast<std::size_t>(n + k)] = s[k];
    };

    TwoLevelDDMConfig ddm_cfg;
    ddm_cfg.fine_subdomain_size = DEFAULT_FINE_SUBDOMAIN_SIZE;
    ddm_cfg.overlap_ratio = DEFAULT_OVERLAP_RATIO;
    ddm_cfg.coarse_ratio = DEFAULT_COARSE_RATIO;
    ddm_cfg.max_levels = 2;
    ddm_cfg.use_direct_coarse = true;

    TwoLevelDDMPreconditioner preconditioner(centers_, kernel_, epsilon_, regularization_sign(kernel_id(kernel_)) * regularization_, ddm_cfg, poly_dim);

    ApplyPreconditionerFn M = [&preconditioner](const std::vector<double>& r_aug,
                                                std::vector<double>& z_aug) {
        preconditioner.apply(r_aug, z_aug);
    };

    GMRESConfig gcfg;
    gcfg.restart = gmres_restart_;
    gcfg.max_iter = gmres_max_iter_;
    gcfg.tol = gmres_tol_;

    std::vector<double> x(aug_dim, 0.0);
    std::string gmres_status;

    const bool ok = gmres_solve(aug_dim, A, rhs, x, gcfg, gmres_status, M);
    if (!ok)
        throw std::runtime_error("fit_iterative(): GMRES failed to converge: " + gmres_status);

    weights_ = std::move(x);
    fitted_ = true;

    solver_backend_ = "iterative_augmented_" + Aop.backend_name() +
                      "_gmres_ddm_levels=" + std::to_string(preconditioner.has_coarse_level() ? 2 : 1) +
                      "_blocks=" + std::to_string(preconditioner.n_local_blocks());

    scalfmm_status_ = deeprbf_scalfmm_runtime_status();
}

// 3.3.3: sparse direct solve for compact kernels (Schur complement on the drift columns, see sparse_compact.hpp).
void NativeExactRBFImplicitModel::fit_sparse_compact(
    const std::vector<std::array<double, 3>>& centers,
    const std::vector<double>& values) {
    if (!compact_grid_) throw std::runtime_error("fit_sparse_compact(): kernel is not compactly supported.");
    SparseCompactSolveInfo info;
    weights_ = solve_compact_sparse_schur(centers, values, kernel_id(kernel_), epsilon_, regularization_,
                                          poly_dim_, *compact_grid_, &info);
    weights_.resize(centers.size() + static_cast<std::size_t>(poly_dim_));   // solver pads to n + 4; keep the model's n + poly_dim layout
    sparse_info_ = {static_cast<double>(info.nnz_lower), info.assembly_seconds, info.factor_seconds,
                    info.solve_seconds, info.residual};
    fitted_ = true;
    solver_backend_ = "sparse_cholesky_schur";
    scalfmm_status_ = deeprbf_scalfmm_runtime_status();
}

void NativeExactRBFImplicitModel::fit(
    const std::vector<std::array<double, 3>>& centers,
    const std::vector<double>& values) {
    fitted_=false;
    evaluate_kernel(kernel_,0.,epsilon_);
    if(!std::isfinite(regularization_)||regularization_<0)throw std::invalid_argument("invalid regularization");
    for(const auto& p:centers)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite centers");
    for(double v:values)if(!std::isfinite(v))throw std::invalid_argument("non-finite values");

    if (centers.empty()) throw std::runtime_error("fit(): centers is empty.");
    if(centers.size()>static_cast<size_t>(std::numeric_limits<int>::max()-4))throw std::length_error("Too many RBF centers");
    if (centers.size() != values.size())
        throw std::runtime_error("fit(): centers and values size mismatch.");
    if (!(epsilon_ > 0.0)) throw std::runtime_error("fit(): epsilon must be > 0.");
    if (regularization_ < 0.0) throw std::runtime_error("fit(): regularization must be >= 0.");
    check_drift_for_kernel(kernel_, poly_dim_);

    centers_ = centers;
    weights_.clear();
    fitted_ = false;
    solver_backend_ = "initializing";
    scalfmm_status_ = deeprbf_scalfmm_runtime_status();

    const int n = static_cast<int>(centers_.size());
    const KernelId kid = kernel_id(kernel_);
    compact_grid_.reset();
    sparse_info_.clear();
    if (kernel_is_compact(kid))
        compact_grid_ = std::make_shared<const CompactNeighbourGrid>(centers_, compact_support_radius(kid, epsilon_));
    // Budget the three dense working matrices used by the fallback solver.
    const long double dense_bytes=3.L*(n+poly_dim_)*(n+poly_dim_)*sizeof(double);
    const bool too_big_for_dense = n > dense_direct_limit_ ||
        dense_bytes > static_cast<long double>(dense_memory_mb_) * 1024.L * 1024.L;
    if (!force_iterative_ && compact_grid_ && (force_sparse_ || too_big_for_dense)) {
        fit_sparse_compact(centers_, values);
    } else if (force_iterative_ || too_big_for_dense) {
        fit_iterative(centers, values);
    } else {
        fit_dense(centers, values);
    }
}

void NativeExactRBFImplicitModel::fit_with_reduction(
    const std::vector<std::array<double, 3>>& centers,
    const std::vector<double>& values,
    double fitting_accuracy,
    double initial_ratio,
    double append_ratio) {
    fitted_=false;
    if(!std::isfinite(fitting_accuracy)||fitting_accuracy<0||!std::isfinite(initial_ratio)||initial_ratio<=0||initial_ratio>1||
       !std::isfinite(append_ratio)||append_ratio<=0||append_ratio>1)
        throw std::invalid_argument("Invalid reduction tolerance or sampling ratios");
    if (centers.empty() || centers.size() != values.size())
        throw std::runtime_error("fit_with_reduction(): Invalid input data.");

    const int N = static_cast<int>(centers.size());

    int n_initial = std::max(10, static_cast<int>(N * initial_ratio));
    n_initial = std::min(n_initial, N);
    const int n_append = std::max(1, static_cast<int>(N * append_ratio));

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(42);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<int> active_indices(indices.begin(), indices.begin() + n_initial);
    std::vector<bool> is_active(N, false);
    for (int idx : active_indices) is_active[idx] = true;

    bool converged = false;
    int iteration = 0;
    const int MAX_REDUCTION_ITERS = 50;

    std::vector<std::array<double, 3>> current_centers;
    std::vector<double> current_values;

    while (!converged && iteration < MAX_REDUCTION_ITERS &&
           active_indices.size() < static_cast<size_t>(N)) {
        current_centers.clear();
        current_values.clear();
        current_centers.reserve(active_indices.size());
        current_values.reserve(active_indices.size());

        for (int idx : active_indices) {
            current_centers.push_back(centers[idx]);
            current_values.push_back(values[idx]);
        }

        fit(current_centers, current_values);

        std::vector<int> inactive_indices;
        std::vector<std::array<double, 3>> eval_points;
        inactive_indices.reserve(N - active_indices.size());
        eval_points.reserve(N - active_indices.size());

        for (int i = 0; i < N; ++i) {
            if (!is_active[i]) {
                inactive_indices.push_back(i);
                eval_points.push_back(centers[i]);
            }
        }
        if (inactive_indices.empty()) { converged = true; break; }

        std::vector<double> preds = predict(eval_points);

        struct ResidualInfo { int original_idx; double residual; };
        std::vector<ResidualInfo> residuals;
        residuals.reserve(inactive_indices.size());

        double max_res = 0.0;
        for (size_t i = 0; i < inactive_indices.size(); ++i) {
            double res = std::abs(preds[i] - values[inactive_indices[i]]);
            max_res = std::max(max_res, res);
            residuals.push_back({inactive_indices[i], res});
        }

        if (max_res <= fitting_accuracy) { converged = true; break; }

        std::sort(residuals.begin(), residuals.end(),
                  [](const ResidualInfo& a, const ResidualInfo& b) {
                      return a.residual > b.residual;
                  });

        int to_add = std::min(n_append, static_cast<int>(residuals.size()));
        for (int i = 0; i < to_add; ++i) {
            int add_idx = residuals[i].original_idx;
            active_indices.push_back(add_idx);
            is_active[add_idx] = true;
        }
        iteration++;
    }

    if (!converged && active_indices.size() == static_cast<size_t>(N)) fit(centers, values);
    else if(!converged){fitted_=false;throw std::runtime_error("RBF reduction iteration limit reached before requested accuracy");}
}

std::vector<double> NativeExactRBFImplicitModel::predict(
    const std::vector<std::array<double, 3>>& points) const {
    for(const auto& p:points)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite targets");

    if (!fitted_) throw std::runtime_error("predict(): model is not fitted.");

    const int n = static_cast<int>(centers_.size());
    const int poly_dim = poly_dim_;
    if (static_cast<int>(weights_.size()) != n + poly_dim)
        throw std::runtime_error("predict(): invalid weights size.");

    // 3.3.3: with "auto", compact kernels sum only the centers within the support radius (exact up to rounding;
    // the terms skipped are zero). "direct" keeps the full O(n) sum per query.
    if (compact_grid_ && predict_backend_ == "auto") {
        last_predict_backend_ = "compact_grid_openmp";
        { std::vector<double> w4(weights_); w4.resize(centers_.size() + 4U, 0.0);   // predictor expects n + 4 drift slots (zeros beyond poly_dim)
          return predict_compact_sparse(points, w4, kernel_id(kernel_), epsilon_, *compact_grid_); }
    }

    std::vector<double> out(points.size(), 0.0);
    std::vector<double> lambda(weights_.begin(),
                               weights_.begin() + static_cast<std::ptrdiff_t>(n));

    // 3.3.2: explicit backend. "direct" is exact. "scalfmm" is approximate and throws if unavailable. "auto" uses
    // ScalFMM only for the Gaussian kernel on large evaluations and reports a fallback in last_predict_backend().
    bool want_fmm = predict_backend_ == "scalfmm";
    std::string used = "direct_openmp";
#if defined(DEEPRBF_HAS_SCALFMM) && DEEPRBF_HAS_SCALFMM
    if (predict_backend_ == "auto")
        want_fmm = kernel_ == "gaussian" &&
                   static_cast<long double>(points.size()) * static_cast<long double>(n) >= 2.0e9L;
    if (want_fmm) {
        try {
            ScalFMMAdapter fmm;
            fmm.setup(centers_, kernel_, epsilon_);
            fmm.evaluate(points, lambda, out);
            for (std::size_t i = 0; i < points.size(); ++i) {
                double pb[4] = {0.0, 0.0, 0.0, 0.0};
                poly_basis(poly_dim, points[i], pb);
                for (int k = 0; k < poly_dim; ++k) out[i] += weights_[static_cast<std::size_t>(n + k)] * pb[k];
            }
            last_predict_backend_ = "scalfmm";
            return out;
        } catch (const std::exception& e) {
            if (predict_backend_ == "scalfmm")
                throw std::runtime_error(std::string("predict(): ScalFMM backend requested but failed: ") + e.what());
            used = std::string("direct_openmp_after_scalfmm_error: ") + e.what();
        }
    }
#else
    if (want_fmm) throw std::runtime_error("predict(): ScalFMM backend requested but this build has no ScalFMM");
#endif
    last_predict_backend_ = used;

    const ResolvedKernel kern = resolve_kernel(kernel_, epsilon_);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        double v = 0.0;
        const auto& p = points[static_cast<std::size_t>(i)];
        for (int j = 0; j < n; ++j) {
            const double r = distance3(p, centers_[static_cast<std::size_t>(j)]);
            v += lambda[static_cast<std::size_t>(j)] * kern(r);
        }
        double pb[4] = {0.0, 0.0, 0.0, 0.0};
        poly_basis(poly_dim, p, pb);
        for (int k = 0; k < poly_dim; ++k) v += weights_[static_cast<std::size_t>(n + k)] * pb[k];
        out[static_cast<std::size_t>(i)] = v;
    }
    return out;
}

bool NativeExactRBFImplicitModel::is_fitted() const { return fitted_; }
int NativeExactRBFImplicitModel::n_centers() const { return static_cast<int>(centers_.size()); }
std::string NativeExactRBFImplicitModel::solver_backend() const { return solver_backend_; }
std::string NativeExactRBFImplicitModel::last_predict_backend() const { return last_predict_backend_; }
std::vector<double> NativeExactRBFImplicitModel::sparse_solve_info() const { return sparse_info_; }
std::string NativeExactRBFImplicitModel::scalfmm_status() const {
    return deeprbf_scalfmm_runtime_status();
}
std::string NativeExactRBFImplicitModel::kernel() const { return kernel_; }
double NativeExactRBFImplicitModel::epsilon() const { return epsilon_; }
double NativeExactRBFImplicitModel::regularization() const { return regularization_; }

} // namespace deeprbf
