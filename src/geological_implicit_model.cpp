#include "deeprbf/kernels.hpp"
#include "deeprbf/geological_implicit_model.hpp"

#include "deeprbf/mkl_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace deeprbf {

namespace {

inline bool is_finite_number(const double v) {
    return std::isfinite(v) != 0;
}

inline std::size_t idx(const int row, const int col, const int ncols) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(ncols) +
           static_cast<std::size_t>(col);
}

inline double norm3(const std::array<double, 3>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline std::array<double, 3> normalize3(const std::array<double, 3>& v) {
    const double n = norm3(v);
    if (n <= 1.0e-12) return {0.0, 0.0, 0.0};
    return {v[0] / n, v[1] / n, v[2] / n};
}

} // namespace

NativeGeologicalImplicitModel::NativeGeologicalImplicitModel()
    : kernel_("gaussian"),
      epsilon_(1.0),
      regularization_(1.0e-6),
      use_trend_transform_(false),
      trend_rotation_{1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0},
      trend_scale_{1.0, 1.0, 1.0},
      trend_center_{0.0, 0.0, 0.0},
      use_polynomial_drift_(false),
      fitted_(false),
      solver_backend_("uninitialized"),
      centers_(),
      weights_() {
}

NativeGeologicalImplicitModel::NativeGeologicalImplicitModel(
    const std::string& kernel,
    double epsilon,
    double regularization
)
    : kernel_(kernel),
      epsilon_(epsilon),
      regularization_(regularization),
      use_trend_transform_(false),
      trend_rotation_{1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0},
      trend_scale_{1.0, 1.0, 1.0},
      trend_center_{0.0, 0.0, 0.0},
      use_polynomial_drift_(false),
      fitted_(false),
      solver_backend_("uninitialized"),
      centers_(),
      weights_() {
    if (!(epsilon_ > 0.0))
        throw std::runtime_error("NativeGeologicalImplicitModel: epsilon must be > 0.");
    if (regularization_ < 0.0)
        throw std::runtime_error("NativeGeologicalImplicitModel: regularization must be >= 0.");
}

void NativeGeologicalImplicitModel::set_trend_transform(
    const std::array<double, 9>& rotation,
    const std::array<double, 3>& scale,
    const std::array<double, 3>& center
) {
    for (double v : rotation) {
        if (!is_finite_number(v))
            throw std::runtime_error("set_trend_transform: rotation contains non-finite value.");
    }
    for (double v : scale) {
        if (!is_finite_number(v) || v <= 0.0)
            throw std::runtime_error("set_trend_transform: scale values must be finite and > 0.");
    }
    for (double v : center) {
        if (!is_finite_number(v))
            throw std::runtime_error("set_trend_transform: center contains non-finite value.");
    }

    trend_rotation_ = rotation;
    trend_scale_ = scale;
    trend_center_ = center;
    use_trend_transform_ = true;
    fitted_=false;

    validate_transform();
}

void NativeGeologicalImplicitModel::clear_trend_transform() {
    use_trend_transform_ = false;
    fitted_=false;
    trend_rotation_ = {1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0};
    trend_scale_ = {1.0, 1.0, 1.0};
    trend_center_ = {0.0, 0.0, 0.0};
}

bool NativeGeologicalImplicitModel::has_trend_transform() const {
    return use_trend_transform_;
}

void NativeGeologicalImplicitModel::set_use_polynomial_drift(bool use_polynomial_drift) {
    if(use_polynomial_drift_!=use_polynomial_drift) fitted_=false;
    use_polynomial_drift_ = use_polynomial_drift;
}

bool NativeGeologicalImplicitModel::use_polynomial_drift() const {
    return use_polynomial_drift_;
}

void NativeGeologicalImplicitModel::validate_transform() const {
    if (!use_trend_transform_) return;

    const double r00 = trend_rotation_[0], r01 = trend_rotation_[1], r02 = trend_rotation_[2];
    const double r10 = trend_rotation_[3], r11 = trend_rotation_[4], r12 = trend_rotation_[5];
    const double r20 = trend_rotation_[6], r21 = trend_rotation_[7], r22 = trend_rotation_[8];

    const double m00 = r00 * r00 + r01 * r01 + r02 * r02;
    const double m11 = r10 * r10 + r11 * r11 + r12 * r12;
    const double m22 = r20 * r20 + r21 * r21 + r22 * r22;

    const double m01 = r00 * r10 + r01 * r11 + r02 * r12;
    const double m02 = r00 * r20 + r01 * r21 + r02 * r22;
    const double m12 = r10 * r20 + r11 * r21 + r12 * r22;

    const double tol = 1.0e-3;
    if (std::abs(m00 - 1.0) > tol ||
        std::abs(m11 - 1.0) > tol ||
        std::abs(m22 - 1.0) > tol ||
        std::abs(m01) > tol ||
        std::abs(m02) > tol ||
        std::abs(m12) > tol) {
        throw std::runtime_error(
            "set_trend_transform: rotation matrix is not sufficiently orthonormal.");
    }
}

std::array<double, 3> NativeGeologicalImplicitModel::apply_transform(
    const std::array<double, 3>& p
) const {
    if (!use_trend_transform_) return p;

    const double qx = p[0] - trend_center_[0];
    const double qy = p[1] - trend_center_[1];
    const double qz = p[2] - trend_center_[2];

    const double tx =
        trend_rotation_[0] * qx +
        trend_rotation_[1] * qy +
        trend_rotation_[2] * qz;

    const double ty =
        trend_rotation_[3] * qx +
        trend_rotation_[4] * qy +
        trend_rotation_[5] * qz;

    const double tz =
        trend_rotation_[6] * qx +
        trend_rotation_[7] * qy +
        trend_rotation_[8] * qz;

    return {
        tx / trend_scale_[0],
        ty / trend_scale_[1],
        tz / trend_scale_[2]
    };
}

std::array<double,3> NativeGeologicalImplicitModel::apply_normal_transform(const std::array<double,3>& n) const {
    auto u=normalize3(n);
    if(!use_trend_transform_)return u;
    std::array<double,3> out{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i]+=trend_rotation_[i*3+j]*u[j]/trend_scale_[i];
    return out;
}

std::vector<std::array<double, 3>> NativeGeologicalImplicitModel::transform_points(
    const std::vector<std::array<double, 3>>& points
) const {
    std::vector<std::array<double, 3>> out;
    out.reserve(points.size());
    for (const auto& p : points) out.push_back(apply_transform(p));
    return out;
}

double NativeGeologicalImplicitModel::distance3(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b
) const {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double NativeGeologicalImplicitModel::kernel_eval(double r) const { return evaluate_kernel(kernel_, r, epsilon_); }

std::array<double, 3> NativeGeologicalImplicitModel::kernel_gradient(
    const std::array<double, 3>& x,
    const std::array<double, 3>& c
) const {
    const double dx = x[0] - c[0];
    const double dy = x[1] - c[1];
    const double dz = x[2] - c[2];

    const double r2 = dx * dx + dy * dy + dz * dz;
    const double r = std::sqrt(r2);
    const double e = (epsilon_ > 0.0) ? epsilon_ : 1.0;

    if (r <= 1.0e-14) return {0.0, 0.0, 0.0};

    if (kernel_ == "gaussian") {
        const double phi = std::exp(-(r / e) * (r / e));
        const double coeff = -2.0 * phi / (e * e);
        return {coeff * dx, coeff * dy, coeff * dz};
    }
    if (kernel_ == "multiquadric") {
        const double denom = std::sqrt(r2 + e * e);
        const double coeff = 1.0 / denom;
        return {coeff * dx, coeff * dy, coeff * dz};
    }
    if (kernel_ == "inverse_multiquadric") {
        const double denom = std::pow(r2 + e * e, 1.5);
        const double coeff = -1.0 / denom;
        return {coeff * dx, coeff * dy, coeff * dz};
    }
    if (kernel_ == "thin_plate" || kernel_ == "thin_plate_spline") {
        const double re = r / e;
        const double dphi_dr = (2*re*std::log(re+1e-12)+re*re/(re+1e-12))/e;
        const double coeff = dphi_dr / r;
        return {coeff * dx, coeff * dy, coeff * dz};
    }
    if (kernel_ == "biharmonic" || kernel_ == "cubic") {
        const double coeff = 3.0 * r;
        return {coeff * dx, coeff * dy, coeff * dz};
    }

    const double coeff = 1.0 / r;
    return {coeff * dx, coeff * dy, coeff * dz};
}

void NativeGeologicalImplicitModel::build_constraints(
    const std::vector<std::array<double, 3>>& interface_points,
    const std::vector<std::array<double, 3>>& inside_points,
    const std::vector<std::array<double, 3>>& outside_points,
    double inside_value,
    double outside_value,
    double interface_weight,
    double inside_weight,
    double outside_weight,
    std::vector<GeologicalConstraint>& constraints
) const {
    constraints.clear();
    constraints.reserve(interface_points.size() + inside_points.size() + outside_points.size());

    for (const auto& p : interface_points)
        constraints.push_back({p, 0.0, interface_weight, GeologicalConstraintType::Interface});
    for (const auto& p : inside_points)
        constraints.push_back({p, inside_value, inside_weight, GeologicalConstraintType::Inside});
    for (const auto& p : outside_points)
        constraints.push_back({p, outside_value, outside_weight, GeologicalConstraintType::Outside});
}

void NativeGeologicalImplicitModel::fit_constraints(
    const std::vector<std::array<double, 3>>& interface_points,
    const std::vector<std::array<double, 3>>& inside_points,
    const std::vector<std::array<double, 3>>& outside_points,
    double inside_value,
    double outside_value,
    double interface_weight,
    double inside_weight,
    double outside_weight
) {
    fit_constraints_with_gradients(
        interface_points,
        inside_points,
        outside_points,
        {}, {}, {},
        inside_value,
        outside_value,
        interface_weight,
        inside_weight,
        outside_weight,
        1.0
    );
}

void NativeGeologicalImplicitModel::fit_constraints_with_gradients(
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
) {
    fitted_=false;
    evaluate_kernel(kernel_,0.,epsilon_);
    for(double v:{regularization_,inside_value,outside_value,interface_weight,inside_weight,outside_weight,gradient_weight})
        if(!std::isfinite(v))throw std::invalid_argument("Nonfinite geological fit control");
    if(regularization_<0)throw std::invalid_argument("Negative geological regularization");
    for(const auto* group:{&interface_points,&inside_points,&outside_points,&gradient_points,&gradient_normals})
        for(const auto& p:*group)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite geological constraint");
    for(double v:gradient_targets)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite gradient target");
    const long double count=static_cast<long double>(interface_points.size())+inside_points.size()+outside_points.size()+gradient_points.size()+4;
    if(5.L*count*count*sizeof(double)>256.L*1024*1024)throw std::length_error("Geological fit exceeds 256 MiB dense workspace limit");
    if (interface_points.empty() && inside_points.empty() &&
        outside_points.empty() && gradient_points.empty()) {
        throw std::runtime_error("fit_constraints_with_gradients: no constraints provided.");
    }
    if (!(inside_value > 0.0))
        throw std::runtime_error("fit_constraints_with_gradients: inside_value must be > 0.");
    if (!(outside_value < 0.0))
        throw std::runtime_error("fit_constraints_with_gradients: outside_value must be < 0.");
    if (!(interface_weight > 0.0) || !(inside_weight > 0.0) ||
        !(outside_weight > 0.0) || !(gradient_weight > 0.0)) {
        throw std::runtime_error("fit_constraints_with_gradients: all weights must be > 0.");
    }
    if (gradient_points.size() != gradient_normals.size() ||
        gradient_points.size() != gradient_targets.size()) {
        throw std::runtime_error("fit_constraints_with_gradients: gradient arrays size mismatch.");
    }

    std::vector<GeologicalConstraint> constraints;
    build_constraints(
        interface_points, inside_points, outside_points,
        inside_value, outside_value,
        interface_weight, inside_weight, outside_weight,
        constraints);

    const std::vector<std::array<double, 3>> transformed_value_points = [&]() {
        std::vector<std::array<double, 3>> pts;
        pts.reserve(constraints.size());
        for (const auto& c : constraints) pts.push_back(apply_transform(c.xyz));
        return pts;
    }();

    const std::vector<std::array<double, 3>> transformed_grad_points =
        transform_points(gradient_points);

    std::vector<std::array<double, 3>> transformed_grad_normals;
    transformed_grad_normals.reserve(gradient_normals.size());
    for (const auto& n : gradient_normals)
        transformed_grad_normals.push_back(apply_normal_transform(n));

    centers_.clear();
    centers_.reserve(transformed_value_points.size() + transformed_grad_points.size());
    centers_.insert(centers_.end(),
                    transformed_value_points.begin(), transformed_value_points.end());
    centers_.insert(centers_.end(),
                    transformed_grad_points.begin(), transformed_grad_points.end());

    const int n = static_cast<int>(centers_.size());
    const int m_value = static_cast<int>(constraints.size());
    const int m_grad = static_cast<int>(gradient_points.size());
    const int m = m_value + m_grad;
    const int poly_dim = use_polynomial_drift_ ? 4 : 0;
    const int p = n + poly_dim;

    std::vector<double> A(static_cast<std::size_t>(m) * static_cast<std::size_t>(p), 0.0);
    std::vector<double> b(static_cast<std::size_t>(m), 0.0);
    std::vector<double> w(static_cast<std::size_t>(m), 1.0);

    for (int i = 0; i < m_value; ++i) {
        const auto& xi = transformed_value_points[static_cast<std::size_t>(i)];

        for (int j = 0; j < n; ++j) {
            const double r = distance3(xi, centers_[static_cast<std::size_t>(j)]);
            A[idx(i, j, p)] = kernel_eval(r);
        }

        if (use_polynomial_drift_) {
            A[idx(i, n, p)]     = 1.0;
            A[idx(i, n + 1, p)] = xi[0];
            A[idx(i, n + 2, p)] = xi[1];
            A[idx(i, n + 3, p)] = xi[2];
        }

        b[static_cast<std::size_t>(i)] = constraints[static_cast<std::size_t>(i)].target_value;
        w[static_cast<std::size_t>(i)] = constraints[static_cast<std::size_t>(i)].weight;
    }

    for (int ig = 0; ig < m_grad; ++ig) {
        const int row = m_value + ig;
        const auto& xg = transformed_grad_points[static_cast<std::size_t>(ig)];
        const auto& ng = transformed_grad_normals[static_cast<std::size_t>(ig)];

        for (int j = 0; j < n; ++j) {
            const auto grad_phi = kernel_gradient(xg, centers_[static_cast<std::size_t>(j)]);
            A[idx(row, j, p)] =
                ng[0] * grad_phi[0] +
                ng[1] * grad_phi[1] +
                ng[2] * grad_phi[2];
        }

        if (use_polynomial_drift_) {
            A[idx(row, n, p)]     = 0.0;
            A[idx(row, n + 1, p)] = ng[0];
            A[idx(row, n + 2, p)] = ng[1];
            A[idx(row, n + 3, p)] = ng[2];
        }

        b[static_cast<std::size_t>(row)] = gradient_targets[static_cast<std::size_t>(ig)];
        w[static_cast<std::size_t>(row)] = gradient_weight;
    }

    // Normal equations
    std::vector<double> ATA(static_cast<std::size_t>(p) * static_cast<std::size_t>(p), 0.0);
    std::vector<double> ATb(static_cast<std::size_t>(p), 0.0);

    for (int i = 0; i < m; ++i) {
        const double wi = w[static_cast<std::size_t>(i)];
        const double wi2 = wi * wi;

        for (int c1 = 0; c1 < p; ++c1) {
            const double a1 = A[idx(i, c1, p)];
            ATb[static_cast<std::size_t>(c1)] += wi2 * a1 * b[static_cast<std::size_t>(i)];

            for (int c2 = 0; c2 < p; ++c2) {
                ATA[idx(c1, c2, p)] += wi2 * a1 * A[idx(i, c2, p)];
            }
        }
    }

    for (int i = 0; i < n; ++i) ATA[idx(i, i, p)] += regularization_;

    std::string backend_used;
    const bool ok = solve_dense_linear_system(ATA, ATb, p, backend_used);
    if (!ok)
        throw std::runtime_error(
            "fit_constraints_with_gradients: dense normal-equation solve failed.");

    weights_ = std::move(ATb);
    fitted_ = true;
    solver_backend_ = "geological_weighted_ls_" + backend_used;
}

std::vector<double> NativeGeologicalImplicitModel::predict(
    const std::vector<std::array<double, 3>>& points
) const {
    if (!fitted_)
        throw std::runtime_error("predict: model is not fitted.");

    const int n = static_cast<int>(centers_.size());
    const int poly_dim = use_polynomial_drift_ ? 4 : 0;
    if (static_cast<int>(weights_.size()) != n + poly_dim)
        throw std::runtime_error("predict: invalid coefficient vector size.");

    std::vector<double> out(points.size(), 0.0);

    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto x = apply_transform(points[i]);

        double v = 0.0;
        for (int j = 0; j < n; ++j) {
            const double r = distance3(x, centers_[static_cast<std::size_t>(j)]);
            v += weights_[static_cast<std::size_t>(j)] * kernel_eval(r);
        }

        if (use_polynomial_drift_) {
            v += weights_[static_cast<std::size_t>(n)];
            v += weights_[static_cast<std::size_t>(n + 1)] * x[0];
            v += weights_[static_cast<std::size_t>(n + 2)] * x[1];
            v += weights_[static_cast<std::size_t>(n + 3)] * x[2];
        }

        out[i] = v;
    }

    return out;
}

bool NativeGeologicalImplicitModel::is_fitted() const { return fitted_; }
int NativeGeologicalImplicitModel::n_centers() const { return static_cast<int>(centers_.size()); }
std::string NativeGeologicalImplicitModel::solver_backend() const { return solver_backend_; }
std::string NativeGeologicalImplicitModel::kernel() const { return kernel_; }
double NativeGeologicalImplicitModel::epsilon() const { return epsilon_; }
double NativeGeologicalImplicitModel::regularization() const { return regularization_; }

} // namespace deeprbf
