#include "deeprbf/kernels.hpp"
#include "deeprbf/operator.hpp"
#include "deeprbf/scalfmm_adapter.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace deeprbf {

namespace {
bool allow_fallback(const ScalFMMOptions& options) {
#if defined(DEEPRBF_STRICT_SCALFMM_RUNTIME) && DEEPRBF_STRICT_SCALFMM_RUNTIME
    return false;
#else
    return options.fallback_on_validation_failure;
#endif
}
}

RBFMatrixFreeOperator::RBFMatrixFreeOperator(
    const std::vector<std::array<double, 3>>& centers,
    const RBFOperatorConfig& cfg)
    : centers_(centers),
      cfg_(cfg),
      scalfmm_initialized_(false),
      scalfmm_enabled_(false),
      backend_name_("direct_openmp"),
      backend_message_("ScalFMM initialization pending"),
      scalfmm_(nullptr) {
    if (centers_.empty()) {
        throw std::invalid_argument("RBFMatrixFreeOperator: centers must not be empty");
    }
    (void)evaluate_kernel(cfg_.kernel, 0.0, cfg_.epsilon);
    if (!std::isfinite(cfg_.regularization))   // signed: negative for order-1 CPD kernels (3.4.2)
        throw std::invalid_argument("invalid RBF regularization");
    for (const auto& p : centers_) for (double v : p)
        if (!std::isfinite(v) || std::abs(v)>1e100)
            throw std::invalid_argument("invalid RBF center coordinate");
    kernel_fast_ = FastKernel(cfg_.kernel, cfg_.epsilon, bbox_diagonal(centers_));
    if (cfg_.use_hmatrix) {
        hmat_ = std::make_unique<HMatrix>(centers_, cfg_.kernel, cfg_.epsilon, cfg_.hmatrix_options);
        backend_name_ = "hmatrix_aca";
        backend_message_ = "H-matrix/ACA product";
        scalfmm_initialized_ = true;   // no ScalFMM with the H-matrix
    }
}

int RBFMatrixFreeOperator::size() const { return static_cast<int>(centers_.size()); }
bool RBFMatrixFreeOperator::has_scalfmm() const { return scalfmm_enabled_; }
std::string RBFMatrixFreeOperator::backend_name() const { return backend_name_; }
std::string RBFMatrixFreeOperator::backend_message() const { return backend_message_; }

double RBFMatrixFreeOperator::distance3(const std::array<double, 3>& a,
                                        const std::array<double, 3>& b) const {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::hypot(dx, dy, dz);
}

double RBFMatrixFreeOperator::kernel_eval(double r) const { return evaluate_kernel(cfg_.kernel, r, cfg_.epsilon); }

void RBFMatrixFreeOperator::initialize_scalfmm_if_needed() const {
    if (scalfmm_initialized_) return;
    scalfmm_initialized_ = true;
    scalfmm_enabled_ = false;
    backend_name_ = "direct_openmp";

    if (!cfg_.prefer_scalfmm) {
        backend_message_ = "ScalFMM disabled by operator configuration";
        return;
    }

    try {
        scalfmm_ = std::make_unique<ScalFMMAdapter>();
        scalfmm_->setup(centers_, cfg_.kernel, cfg_.epsilon, cfg_.scalfmm_options);
        backend_name_ = scalfmm_->backend_name();
        backend_message_ = scalfmm_->last_error();
    } catch (const std::exception& exception) {
        scalfmm_.reset();
        backend_name_ = "direct_openmp";
        backend_message_ = exception.what();
        if (!allow_fallback(cfg_.scalfmm_options)) { scalfmm_initialized_=false; throw; }
    }
}

void RBFMatrixFreeOperator::apply_direct(const std::vector<double>& x,
                                         std::vector<double>& y) const {
    const int n = static_cast<int>(centers_.size());
    y.assign(static_cast<std::size_t>(n), 0.0);
    const FastKernel& kern = kernel_fast_;   // 3.5.0: tabulated kernel, validated in the constructor

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        double compensation = 0.0;
        const auto& center_i = centers_[static_cast<std::size_t>(i)];
        for (int j = 0; j < n; ++j) {
            const double term = x[static_cast<std::size_t>(j)] *
                kern(fast_distance(center_i, centers_[static_cast<std::size_t>(j)]));
            const double t = sum + term;
            if (std::abs(sum) >= std::abs(term)) {
                compensation += (sum - t) + term;
            } else {
                compensation += (term - t) + sum;
            }
            sum = t;
        }
        y[static_cast<std::size_t>(i)] = sum + compensation +
            cfg_.regularization * x[static_cast<std::size_t>(i)];
    }
}

void RBFMatrixFreeOperator::apply(const std::vector<double>& x,
                                  std::vector<double>& y) const {
    const int n = static_cast<int>(centers_.size());
    if (static_cast<int>(x.size()) != n) {
        throw std::invalid_argument("RBFMatrixFreeOperator::apply: vector size mismatch");
    }
    for (double v : x) if (!std::isfinite(v))
        throw std::invalid_argument("nonfinite RBF operator weights");

    if (hmat_) {
        hmat_->apply(x, y);
        for (int i = 0; i < n; ++i) y[static_cast<std::size_t>(i)] += cfg_.regularization * x[static_cast<std::size_t>(i)];
        return;
    }
    initialize_scalfmm_if_needed();
    if (scalfmm_) {
        try {
            scalfmm_->multiply(x, y);
            if (static_cast<int>(y.size()) != n) {
                throw std::runtime_error("ScalFMM adapter returned the wrong output size");
            }
            for (int i = 0; i < n; ++i) {
                y[static_cast<std::size_t>(i)] +=
                    cfg_.regularization * x[static_cast<std::size_t>(i)];
            }
            const auto info = scalfmm_->execution_info();
            scalfmm_enabled_ = info.active;
            backend_name_ = info.backend;
            backend_message_ = info.message;
            return;
        } catch (const std::exception& exception) {
            scalfmm_enabled_ = false;
            backend_name_ = "direct_openmp_after_adapter_error";
            backend_message_ = exception.what();
            scalfmm_.reset();
            if (!allow_fallback(cfg_.scalfmm_options)) { scalfmm_initialized_=false; throw; }
        }
    }

    apply_direct(x, y);
}

}  // namespace deeprbf
