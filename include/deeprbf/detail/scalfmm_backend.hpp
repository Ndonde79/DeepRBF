#ifndef DEEPRBF_DETAIL_SCALFMM_BACKEND_HPP
#define DEEPRBF_DETAIL_SCALFMM_BACKEND_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace deeprbf::detail {

struct ScalFMMBackendOptions {
    int tree_height = 0;
    int interpolation_order = 8;
    int group_size = 256;
    bool use_openmp = true;
};

struct ScalFMMBackendReport {
    bool compiled = false;
    bool executed = false;
    std::string backend = "direct_openmp";
    std::string message;
    int tree_height = 0;
    int interpolation_order = 0;
    std::size_t source_count = 0;
    std::size_t target_count = 0;
    double elapsed_seconds = 0.0;
};

class IScalFMMBackend {
public:
    virtual ~IScalFMMBackend() = default;

    virtual void setup(const std::vector<std::array<double, 3>>& centers,
                       const std::string& kernel,
                       double epsilon,
                       const ScalFMMBackendOptions& options) = 0;

    virtual void multiply(const std::vector<double>& source_weights,
                          std::vector<double>& target_values) = 0;

    virtual void evaluate(const std::vector<std::array<double, 3>>& target_points,
                          const std::vector<double>& source_weights,
                          std::vector<double>& target_values) = 0;

    [[nodiscard]] virtual ScalFMMBackendReport report() const = 0;
};

std::unique_ptr<IScalFMMBackend> make_scalfmm_backend();
bool scalfmm_backend_compiled() noexcept;
std::string scalfmm_backend_build_status();

}  // namespace deeprbf::detail

#endif  // DEEPRBF_DETAIL_SCALFMM_BACKEND_HPP