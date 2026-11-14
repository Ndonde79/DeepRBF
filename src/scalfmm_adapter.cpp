#include "deeprbf/kernels.hpp"
#include "deeprbf/scalfmm_adapter.hpp"
#include "deeprbf/detail/scalfmm_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace deeprbf {
namespace {

#if defined(DEEPRBF_STRICT_SCALFMM_RUNTIME) && DEEPRBF_STRICT_SCALFMM_RUNTIME
constexpr bool kStrictScalFMMRuntime = true;
#else
constexpr bool kStrictScalFMMRuntime = false;
#endif

std::mutex g_runtime_mutex;
ScalFMMExecutionInfo g_runtime_info = [] {
    ScalFMMExecutionInfo info;
    info.compiled = detail::scalfmm_backend_compiled();
    info.message = "No evaluation has run yet";
    return info;
}();

std::string bool_text(bool value) { return value ? "true" : "false"; }

std::string number_text(double value) {
    if (!std::isfinite(value)) return "nan";
    std::ostringstream stream;
    stream << std::setprecision(8) << value;
    return stream.str();
}

void publish_runtime_info(const ScalFMMExecutionInfo& info) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    g_runtime_info = info;
}

ScalFMMExecutionInfo read_runtime_info() {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    return g_runtime_info;
}

std::vector<std::size_t> sample_indices(std::size_t count, std::size_t requested) {
    if (count == 0 || requested == 0) return {};
    requested = std::min(count, requested);
    std::vector<std::size_t> indices;
    indices.reserve(requested);
    if (requested == 1) { indices.push_back(count / 2); return indices; }
    for (std::size_t k = 0; k < requested; ++k) {
        const long double fraction = static_cast<long double>(k) /
                                     static_cast<long double>(requested - 1);
        const auto index = static_cast<std::size_t>(
            std::llround(fraction * static_cast<long double>(count - 1)));
        if (indices.empty() || indices.back() != index) indices.push_back(index);
    }
    return indices;
}

}  // namespace

struct ScalFMMAdapter::Impl {
    std::vector<std::array<double, 3>> centers;
    std::string kernel = "gaussian";
    double epsilon = 1.0;
    ScalFMMOptions options{};
    bool ready = false;

    mutable bool backend_disabled = false;
    mutable bool validation_completed = false;
    mutable std::unique_ptr<detail::IScalFMMBackend> backend;
    mutable ScalFMMExecutionInfo info{};

    [[nodiscard]] double distance(const std::array<double, 3>& a,
                                  const std::array<double, 3>& b) const {
        const double dx = a[0] - b[0];
        const double dy = a[1] - b[1];
        const double dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    [[nodiscard]] double phi(double r) const { return evaluate_kernel_id(kernel_id(kernel),r,epsilon); }

    void direct_evaluate(const std::vector<std::array<double, 3>>& targets,
                         const std::vector<double>& weights,
                         std::vector<double>& values) const {
        values.assign(targets.size(), 0.0);
        const auto n_targets = static_cast<std::int64_t>(targets.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t ii = 0; ii < n_targets; ++ii) {
            const auto i = static_cast<std::size_t>(ii);
            double sum = 0.0;
            double compensation = 0.0;
            for (std::size_t j = 0; j < centers.size(); ++j) {
                const double term = weights[j] * phi(distance(targets[i], centers[j]));
                const double t = sum + term;
                if (std::abs(sum) >= std::abs(term)) {
                    compensation += (sum - t) + term;
                } else {
                    compensation += (term - t) + sum;
                }
                sum = t;
            }
            values[i] = sum + compensation;
        }
    }

    [[nodiscard]] double validation_error(
        const std::vector<std::array<double, 3>>& targets,
        const std::vector<double>& weights,
        const std::vector<double>& approximate) const {
        const auto indices = sample_indices(targets.size(), options.validation_target_samples);
        if (indices.empty()) return 0.0;

        long double numerator = 0.0L;
        long double denominator = 0.0L;
        for (const auto i : indices) {
            long double exact = 0.0L;
            long double compensation = 0.0L;
            for (std::size_t j = 0; j < centers.size(); ++j) {
                const long double term = static_cast<long double>(weights[j]) *
                    static_cast<long double>(phi(distance(targets[i], centers[j])));
                const long double t = exact + term;
                if (std::abs(exact) >= std::abs(term)) {
                    compensation += (exact - t) + term;
                } else {
                    compensation += (term - t) + exact;
                }
                exact = t;
            }
            exact += compensation;
            const long double delta = static_cast<long double>(approximate[i]) - exact;
            numerator += delta * delta;
            denominator += exact * exact;
        }
        const long double floor = std::numeric_limits<double>::epsilon();
        return static_cast<double>(std::sqrt(numerator / std::max(denominator, floor)));
    }

    void set_direct_status(const std::string& message,
                           std::size_t target_count,
                           bool preserve_error = false) const {
        info.compiled = detail::scalfmm_backend_compiled();
        info.active = false;
        info.fallback_used = true;
        if (!preserve_error || info.backend.empty() || info.active) {
            info.backend = "direct_openmp";
        }
        info.message = message;
        info.kernel = kernel;
        info.source_count = centers.size();
        info.target_count = target_count;
        if (!preserve_error) info.validation_relative_error = -1.0;
        publish_runtime_info(info);
    }

    bool attempt_backend(const std::vector<std::array<double, 3>>& targets,
                         const std::vector<double>& weights,
                         std::vector<double>& values) const {
        if (backend_disabled && (kStrictScalFMMRuntime || !options.fallback_on_validation_failure))
            throw std::runtime_error(info.message);
        if (!backend || backend_disabled) return false;
        if (centers.size() < options.minimum_fmm_sources) {
            set_direct_status("problem_below_fmm_threshold", targets.size());
            return false;
        }

        try {
            if (targets.data() == centers.data() && targets.size() == centers.size()) {
                backend->multiply(weights, values);
            } else {
                backend->evaluate(targets, weights, values);
            }

            const auto report = backend->report();
            info.compiled = report.compiled;
            info.active = report.executed;
            info.fallback_used = !report.executed;
            info.backend = report.backend;
            info.message = report.message;
            info.kernel = kernel;
            info.tree_height = report.tree_height;
            info.interpolation_order = report.interpolation_order;
            info.source_count = report.source_count;
            info.target_count = report.target_count;
            info.elapsed_seconds = report.elapsed_seconds;
            ++info.execution_count;

            if (!report.executed || values.size() != targets.size()) {
                throw std::runtime_error("ScalFMM backend did not return a complete result");
            }

            if (options.validate_first_execution && !validation_completed) {
                const double error = validation_error(targets, weights, values);
                info.validation_relative_error = error;
                info.validation_tolerance = options.validation_relative_tolerance;
                validation_completed = true;
                if (!std::isfinite(error) || error > options.validation_relative_tolerance) {
                    info.validated = false;
                    std::ostringstream message;
                    message << "ScalFMM validation failed: relative L2 error=" << error
                            << ", tolerance=" << options.validation_relative_tolerance;
                    info.message = message.str();
                    if (kStrictScalFMMRuntime || !options.fallback_on_validation_failure) {
                        publish_runtime_info(info);
                        throw std::runtime_error(info.message);
                    }
                    backend_disabled = true;
                    info.active = false;
                    info.fallback_used = true;
                    info.backend = "direct_openmp_after_fmm_validation_failure";
                    publish_runtime_info(info);
                    return false;
                }
                info.validated = true;
                std::ostringstream message;
                message << "ScalFMM validation passed: relative L2 error=" << error
                        << ", tolerance=" << options.validation_relative_tolerance;
                info.message = message.str();
            }

            publish_runtime_info(info);
            return true;
        } catch (const std::exception& exception) {
            backend_disabled = true;
            info.active = false;
            info.fallback_used = !kStrictScalFMMRuntime && options.fallback_on_validation_failure;
            info.backend = kStrictScalFMMRuntime
                ? "scalfmm_runtime_error_strict"
                : "direct_openmp_after_scalfmm_error";
            info.message = exception.what();
            info.source_count = centers.size();
            info.target_count = targets.size();
            publish_runtime_info(info);
            if (kStrictScalFMMRuntime || !options.fallback_on_validation_failure) throw;
            return false;
        } catch (...) {
            backend_disabled = true;
            info.active = false;
            info.fallback_used = !kStrictScalFMMRuntime && options.fallback_on_validation_failure;
            info.backend = kStrictScalFMMRuntime
                ? "scalfmm_unknown_error_strict"
                : "direct_openmp_after_unknown_scalfmm_error";
            info.message = "unknown ScalFMM backend exception";
            info.source_count = centers.size();
            info.target_count = targets.size();
            publish_runtime_info(info);
            if (kStrictScalFMMRuntime || !options.fallback_on_validation_failure) throw;
            return false;
        }
    }
};

ScalFMMAdapter::ScalFMMAdapter() : impl_(std::make_unique<Impl>()) {}
ScalFMMAdapter::~ScalFMMAdapter() = default;
ScalFMMAdapter::ScalFMMAdapter(ScalFMMAdapter&&) noexcept = default;
ScalFMMAdapter& ScalFMMAdapter::operator=(ScalFMMAdapter&&) noexcept = default;

void ScalFMMAdapter::setup(const std::vector<std::array<double, 3>>& centers,
                           const std::string& kernel,
                           double epsilon,
                           const ScalFMMOptions& options) {
    if (centers.empty()) {
        throw std::invalid_argument("ScalFMMAdapter::setup: centers must not be empty");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("ScalFMMAdapter::setup: epsilon must be positive and finite");
    }
    if (options.interpolation_order < 2 || options.interpolation_order > 16) {
        throw std::invalid_argument("ScalFMMAdapter::setup: interpolation order must be in [2, 16]");
    }
    if (options.group_size <= 0) {
        throw std::invalid_argument("ScalFMMAdapter::setup: group size must be positive");
    }
    if(!std::isfinite(options.validation_relative_tolerance)||options.validation_relative_tolerance<=0||
       (options.validate_first_execution&&options.validation_target_samples==0))
        throw std::invalid_argument("Invalid ScalFMM validation controls");

    evaluate_kernel(kernel,0.0,epsilon);
    for(const auto& p:centers)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite center");
    impl_->centers = centers;
    impl_->kernel = kernel;
    impl_->epsilon = epsilon;
    impl_->options = options;
    impl_->ready = true;
    impl_->backend_disabled = false;
    impl_->validation_completed = false;
    impl_->info = {};
    impl_->info.compiled = detail::scalfmm_backend_compiled();
    impl_->info.kernel = kernel;
    impl_->info.source_count = centers.size();
    impl_->info.validation_tolerance = options.validation_relative_tolerance;

    impl_->backend = detail::make_scalfmm_backend();
    if (impl_->backend) {
        detail::ScalFMMBackendOptions backend_options;
        backend_options.tree_height = options.tree_height;
        backend_options.interpolation_order = options.interpolation_order;
        backend_options.group_size = options.group_size;
        backend_options.use_openmp = options.use_openmp;
        try {
            impl_->backend->setup(centers, kernel, epsilon, backend_options);
            const auto report = impl_->backend->report();
            impl_->info.compiled = report.compiled;
            impl_->info.backend = report.backend;
            impl_->info.message = report.message;
            impl_->info.tree_height = report.tree_height;
            impl_->info.interpolation_order = report.interpolation_order;
            impl_->info.fallback_used = true;
        } catch (const std::exception& exception) {
            impl_->backend.reset();
            impl_->backend_disabled = true;
            impl_->info.backend = kStrictScalFMMRuntime
                ? "scalfmm_setup_error_strict"
                : "direct_openmp_after_scalfmm_setup_error";
            impl_->info.message = exception.what();
            if (kStrictScalFMMRuntime || !options.fallback_on_validation_failure) {
                publish_runtime_info(impl_->info);
                throw;
            }
        }
    } else {
        impl_->info.backend = "direct_openmp";
        impl_->info.message = detail::scalfmm_backend_build_status();
    }
    publish_runtime_info(impl_->info);
}

bool ScalFMMAdapter::is_ready() const { return impl_->ready; }
bool ScalFMMAdapter::using_scalfmm() const { return impl_->info.active; }
int ScalFMMAdapter::n_centers() const { return static_cast<int>(impl_->centers.size()); }
std::string ScalFMMAdapter::kernel_name() const { return impl_->kernel; }
double ScalFMMAdapter::epsilon() const { return impl_->epsilon; }
std::string ScalFMMAdapter::backend_name() const { return impl_->info.backend; }
std::string ScalFMMAdapter::last_error() const { return impl_->info.message; }
ScalFMMExecutionInfo ScalFMMAdapter::execution_info() const { return impl_->info; }

double ScalFMMAdapter::distance3(const std::array<double, 3>& a,
                                 const std::array<double, 3>& b) const {
    return impl_->distance(a, b);
}

double ScalFMMAdapter::kernel_eval(double r) const { return impl_->phi(r); }

void ScalFMMAdapter::multiply(const std::vector<double>& source_weights,
                              std::vector<double>& target_values) const {
    if (!impl_->ready) throw std::runtime_error("ScalFMMAdapter::multiply: adapter is not initialized");
    for(double v:source_weights)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite source weight");
    if (source_weights.size() != impl_->centers.size()) {
        throw std::invalid_argument("ScalFMMAdapter::multiply: source weight size mismatch");
    }
    if (!impl_->attempt_backend(impl_->centers, source_weights, target_values)) {
        impl_->direct_evaluate(impl_->centers, source_weights, target_values);
        impl_->set_direct_status(
            impl_->info.message.empty() ? "exact_direct_fallback" : impl_->info.message,
            impl_->centers.size(),
            true);
    }
}

void ScalFMMAdapter::evaluate(const std::vector<std::array<double, 3>>& target_points,
                              const std::vector<double>& source_weights,
                              std::vector<double>& target_values) const {
    if (!impl_->ready) throw std::runtime_error("ScalFMMAdapter::evaluate: adapter is not initialized");
    for(double v:source_weights)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite source weight");
    for(const auto& p:target_points)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite target coordinate");
    if (source_weights.size() != impl_->centers.size()) {
        throw std::invalid_argument("ScalFMMAdapter::evaluate: source weight size mismatch");
    }
    if (target_points.empty()) { target_values.clear(); return; }
    if (!impl_->attempt_backend(target_points, source_weights, target_values)) {
        impl_->direct_evaluate(target_points, source_weights, target_values);
        impl_->set_direct_status(
            impl_->info.message.empty() ? "exact_direct_fallback" : impl_->info.message,
            target_points.size(),
            true);
    }
}

std::string deeprbf_scalfmm_runtime_status() {
    const auto info = read_runtime_info();
    if (info.active) return info.backend + ":active";
    if (info.compiled) return info.backend + ":compiled_not_active";
    return "direct_openmp:scalfmm_not_compiled";
}

std::map<std::string, std::string> deeprbf_scalfmm_diagnostics() {
    const auto info = read_runtime_info();
    return {
        {"compiled", bool_text(info.compiled)},
        {"active", bool_text(info.active)},
        {"validated", bool_text(info.validated)},
        {"fallback_used", bool_text(info.fallback_used)},
        {"backend", info.backend},
        {"message", info.message},
        {"kernel", info.kernel},
        {"tree_height", std::to_string(info.tree_height)},
        {"interpolation_order", std::to_string(info.interpolation_order)},
        {"source_count", std::to_string(info.source_count)},
        {"target_count", std::to_string(info.target_count)},
        {"execution_count", std::to_string(info.execution_count)},
        {"validation_relative_error", number_text(info.validation_relative_error)},
        {"validation_tolerance", number_text(info.validation_tolerance)},
        {"elapsed_seconds", number_text(info.elapsed_seconds)},
        {"build_status", detail::scalfmm_backend_build_status()}
    };
}

}  // namespace deeprbf
