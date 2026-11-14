#pragma once
#include "deeprbf/core.hpp"
namespace deeprbf {
enum class GeologicalConstraintType { Interface, Inside, Outside };
struct GeologicalConstraint { Point3 xyz; double target_value; double weight; GeologicalConstraintType type; };
// Reconstructed from the archived implementation; helper methods remain available for validation.
class NativeGeologicalImplicitModel {
public:
NativeGeologicalImplicitModel();
NativeGeologicalImplicitModel(
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
void set_use_polynomial_drift(bool use_polynomial_drift);
bool use_polynomial_drift() const;
void validate_transform() const;
std::array<double, 3> apply_transform(
    const std::array<double, 3>& p
) const;
std::array<double, 3> apply_normal_transform(
    const std::array<double, 3>& n
) const;
std::vector<std::array<double, 3>> transform_points(
    const std::vector<std::array<double, 3>>& points
) const;
double distance3(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b
) const;
double kernel_eval(double r) const;
std::array<double, 3> kernel_gradient(
    const std::array<double, 3>& x,
    const std::array<double, 3>& c
) const;
void build_constraints(
    const std::vector<std::array<double, 3>>& interface_points,
    const std::vector<std::array<double, 3>>& inside_points,
    const std::vector<std::array<double, 3>>& outside_points,
    double inside_value,
    double outside_value,
    double interface_weight,
    double inside_weight,
    double outside_weight,
    std::vector<GeologicalConstraint>& constraints
) const;
void fit_constraints(
    const std::vector<std::array<double, 3>>& interface_points,
    const std::vector<std::array<double, 3>>& inside_points,
    const std::vector<std::array<double, 3>>& outside_points,
    double inside_value,
    double outside_value,
    double interface_weight,
    double inside_weight,
    double outside_weight
);
void fit_constraints_with_gradients(
    const std::vector<std::array<double, 3>>& interface_points,
    const std::vector<std::array<double, 3>>& inside_points,
    const std::vector<std::array<double, 3>>& outside_points,
    const std::vector<std::array<double, 3>>& gradient_points,
    const std::vector<std::array<double, 3>>& gradient_normals,
    const std::vector<double>& gradient_targets,
    double inside_value,
    double outside_value,
    double interface_weight,
    double inside_weight,
    double outside_weight,
    double gradient_weight
);
std::vector<double> predict(
    const std::vector<std::array<double, 3>>& points
) const;
bool is_fitted() const;
int n_centers() const;
std::string solver_backend() const;
std::string kernel() const;
double epsilon() const;
double regularization() const;
private:
std::string kernel_; double epsilon_,regularization_;
bool use_trend_transform_; std::array<double,9> trend_rotation_;
Point3 trend_scale_,trend_center_; bool use_polynomial_drift_,fitted_;
std::string solver_backend_; std::vector<Point3> centers_; std::vector<double> weights_;
};
}
