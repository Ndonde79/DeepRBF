#ifndef DEEPRBF_SCALFMM_ADAPTER_HPP
#define DEEPRBF_SCALFMM_ADAPTER_HPP

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace deeprbf {

struct ScalFMMOptions {
    int tree_height = 0;                    // 0 = automatic
    int interpolation_order = 8;
    int group_size = 256;
    std::size_t minimum_fmm_sources = 512;
    std::size_t validation_target_samples = 48;
    double validation_relative_tolerance = 2.0e-3;
    bool validate_first_execution = true;
    bool fallback_on_validation_failure = true;
    bool use_openmp = true;
};

struct ScalFMMExecutionInfo {
    bool compiled = false;
    bool active = false;
    bool validated = false;
    bool fallback_used = true;
    std::string backend = "direct_openmp";
    std::string message;
    std::string kernel;
    int tree_height = 0;
    int interpolation_order = 0;
    std::size_t source_count = 0;
    std::size_t target_count = 0;
    std::size_t execution_count = 0;
    double validation_relative_error = -1.0;
    double validation_tolerance = 2.0e-3;
    double elapsed_seconds = 0.0;
};

class ScalFMMAdapter {
public:
    ScalFMMAdapter();
    ~ScalFMMAdapter();

    ScalFMMAdapter(ScalFMMAdapter&& other) noexcept;
    ScalFMMAdapter& operator=(ScalFMMAdapter&& other) noexcept;

    ScalFMMAdapter(const ScalFMMAdapter&) = delete;
    ScalFMMAdapter& operator=(const ScalFMMAdapter&) = delete;

    void setup(const std::vector<std::array<double, 3>>& centers,
               const std::string& kernel,
               double epsilon,
               const ScalFMMOptions& options = {});

    [[nodiscard]] bool is_ready() const;
    [[nodiscard]] bool using_scalfmm() const;
    [[nodiscard]] int n_centers() const;
    [[nodiscard]] std::string kernel_name() const;
    [[nodiscard]] double epsilon() const;
    [[nodiscard]] std::string backend_name() const;
    [[nodiscard]] std::string last_error() const;
    [[nodiscard]] ScalFMMExecutionInfo execution_info() const;

    [[nodiscard]] double distance3(const std::array<double, 3>& a,
                                   const std::array<double, 3>& b) const;
    [[nodiscard]] double kernel_eval(double r) const;

    void multiply(const std::vector<double>& source_weights,
                  std::vector<double>& target_values) const;

    void evaluate(const std::vector<std::array<double, 3>>& target_points,
                  const std::vector<double>& source_weights,
                  std::vector<double>& target_values) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string deeprbf_scalfmm_runtime_status();
[[nodiscard]] std::map<std::string, std::string> deeprbf_scalfmm_diagnostics();

}  // namespace deeprbf

#endif  // DEEPRBF_SCALFMM_ADAPTER_HPP