#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "deeprbf/hmatrix.hpp"
#include "deeprbf/kernel_table.hpp"
#include "deeprbf/scalfmm_adapter.hpp"

namespace deeprbf {

class ScalFMMAdapter;

struct RBFOperatorConfig {
    std::string kernel = "gaussian";
    double epsilon = 1.0;
    double regularization = 1.0e-6;
    bool prefer_scalfmm = true;
    bool use_hmatrix = false;          // 3.5.0: H-matrix/ACA product instead of the exact O(n^2) sum
    HMatrixOptions hmatrix_options{};
    ScalFMMOptions scalfmm_options{};
};

class RBFMatrixFreeOperator {
public:
    RBFMatrixFreeOperator(
        const std::vector<std::array<double, 3>>& centers,
        const RBFOperatorConfig& cfg
    );

    [[nodiscard]] int size() const;
    void apply(const std::vector<double>& x, std::vector<double>& y) const;
    [[nodiscard]] double kernel_eval(double r) const;
    [[nodiscard]] bool has_scalfmm() const;
    [[nodiscard]] std::string backend_name() const;
    [[nodiscard]] std::string backend_message() const;
    [[nodiscard]] const HMatrix* hmatrix() const { return hmat_.get(); }

private:
    [[nodiscard]] double distance3(const std::array<double, 3>& a,
                                   const std::array<double, 3>& b) const;
    void apply_direct(const std::vector<double>& x, std::vector<double>& y) const;
    void initialize_scalfmm_if_needed() const;

private:
    std::vector<std::array<double, 3>> centers_;
    RBFOperatorConfig cfg_;
    FastKernel kernel_fast_;   // 3.5.0: tabulated kernel for the direct product

    mutable bool scalfmm_initialized_;
    mutable bool scalfmm_enabled_;
    mutable std::string backend_name_;
    mutable std::string backend_message_;
    mutable std::unique_ptr<ScalFMMAdapter> scalfmm_;
    std::unique_ptr<HMatrix> hmat_;
};

}  // namespace deeprbf