#pragma once

#include <array>
#include <string>
#include <vector>

#include "deeprbf/rbf_model.hpp"

namespace deeprbf {

class NativeTrendImplicitModel {
public:
    NativeTrendImplicitModel();
    NativeTrendImplicitModel(
        const std::string& kernel,
        double epsilon,
        double regularization
    );

    void set_trend_transform(
        const std::array<double, 9>& rotation,
        const std::array<double, 3>& scale,
        const std::array<double, 3>& center
    );

    void clear_trend_transform();
    bool has_trend_transform() const;

    void fit_signed_constraints(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& signed_values
    );

    void fit_signed_constraints(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& signed_values,
        const std::vector<double>& weights
    );

    std::vector<double> predict(
        const std::vector<std::array<double, 3>>& points
    ) const;

    bool is_fitted() const;
    int n_centers() const;
    std::string solver_backend() const;
    std::string scalfmm_status() const;

    std::string kernel() const;
    double epsilon() const;
    double regularization() const;

private:
    std::array<double, 3> apply_transform(
        const std::array<double, 3>& p
    ) const;

    std::vector<std::array<double, 3>> transform_points(
        const std::vector<std::array<double, 3>>& points
    ) const;

    void validate_weights(
        const std::vector<double>& weights
    ) const;

    void fit_internal(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& signed_values,
        const std::vector<double>* weights
    );

    void build_weighted_training_set(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& signed_values,
        const std::vector<double>& weights,
        std::vector<std::array<double, 3>>& out_points,
        std::vector<double>& out_values
    ) const;

private:
    std::string kernel_;
    double epsilon_;
    double regularization_;

    bool use_trend_transform_;
    std::array<double, 9> trend_rotation_;
    std::array<double, 3> trend_scale_;
    std::array<double, 3> trend_center_;

    NativeExactRBFImplicitModel backend_; // retained for source compatibility
    std::vector<std::array<double,3>> centers_;
    std::vector<double> coefficients_;
    bool fitted_=false;
    std::string solver_backend_="uninitialized";
};

} // namespace deeprbf