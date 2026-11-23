#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// ---- DeepRBF headers ----
#include "deeprbf/core.hpp"
#include "deeprbf/kernels.hpp"
#include "deeprbf/kernel_table.hpp"
#include "deeprbf/hmatrix.hpp"
#include <random>
#include "deeprbf/spatial_hash.hpp"
#include "deeprbf/nested_variogram.hpp"
#include "deeprbf/local_variogram.hpp"
#include "deeprbf/gmres.hpp"
#include "deeprbf/mkl_solver.hpp"
#include "deeprbf/rbf_model.hpp"
#include "deeprbf/operator.hpp"
#include "deeprbf/scalfmm_adapter.hpp"
#include "deeprbf/domain_decomposition.hpp"
#include "deeprbf/gc_geostatistics.hpp"
#include "deeprbf/gc_lmc_fit.hpp"
#include "deeprbf/gc_block_kriging.hpp"
#include "deeprbf/gc_change_of_support.hpp"
#include "deeprbf/implicit_model.hpp"
#include "deeprbf/geological_implicit_model.hpp"
#include "deeprbf/mps_snesim.hpp"
#include "deeprbf/gc_simulation.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;
void bind_audited_engine(py::module_& m);
using namespace deeprbf;

// =============================================================================
// Helpers
// =============================================================================
static std::vector<Point3> points_to_vector(
    const std::vector<std::array<double, 3>>& pts
) {
    std::vector<Point3> out;
    out.reserve(pts.size());
    for (const auto& p : pts) out.push_back({p[0], p[1], p[2]});
    return out;
}

// =============================================================================
// Module
// =============================================================================
PYBIND11_MODULE(deeprbf_recovered, m) {
    bind_audited_engine(m);
    m.doc() = "DeepRBF Core Engine: RBF, implicit geology, OK, SGS, IK, SIS, "
              "cokriging, block kriging, change of support, SNESIM.";

    // -------------------------------------------------------------------------
    // Version / build flags
    // -------------------------------------------------------------------------
#if defined(DEEPRBF_VERSION)
    m.attr("__version__") = DEEPRBF_VERSION;
#else
    m.attr("__version__") = "2.2.5-recovered";
#endif

#if defined(DEEPRBF_HAS_MKL) && DEEPRBF_HAS_MKL
    m.attr("__has_mkl__") = true;
#else
    m.attr("__has_mkl__") = false;
#endif

#if defined(DEEPRBF_HAS_SCALFMM) && DEEPRBF_HAS_SCALFMM
    m.attr("__has_scalfmm__") = true;
#else
    m.attr("__has_scalfmm__") = false;
#endif

#ifdef _OPENMP
    m.attr("__has_openmp__") = true;
    m.attr("__openmp_version__") = _OPENMP;
    m.attr("__openmp_max_threads__") = omp_get_max_threads();
#else
    m.attr("__has_openmp__") = false;
    m.attr("__openmp_version__") = 0;
    m.attr("__openmp_max_threads__") = 1;
#endif

    m.def("get_openmp_threads", []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Returns available OpenMP thread count");

    m.def("set_openmp_threads", [](int n) {
#ifdef _OPENMP
        omp_set_num_threads(std::max(n, 1));
        return true;
#else
        (void)n;
        return false;
#endif
    }, py::arg("n"), "Sets OpenMP thread count if OpenMP is enabled");

    m.def("deeprbf_scalfmm_runtime_status", &deeprbf_scalfmm_runtime_status);
    m.def("deeprbf_scalfmm_diagnostics",   &deeprbf_scalfmm_diagnostics);

    // -------------------------------------------------------------------------
    // Nested variogram
    // -------------------------------------------------------------------------
    py::enum_<VariogramModelType>(m, "VariogramModelType")
        .value("Spherical",   VariogramModelType::Spherical)
        .value("Exponential", VariogramModelType::Exponential)
        .value("Gaussian",    VariogramModelType::Gaussian);

    py::class_<VariogramStructure>(m, "VariogramStructure")
        .def(py::init<>())
        .def_readwrite("type", &VariogramStructure::type)
        .def_readwrite("range", &VariogramStructure::range)
        .def_readwrite("contribution", &VariogramStructure::contribution);

    py::class_<NestedVariogram>(m, "NestedVariogram")
        .def(py::init<double>(), py::arg("nugget") = 0.0)
        .def("add_structure", &NestedVariogram::add_structure)
        .def("set_nugget", &NestedVariogram::set_nugget)
        .def("nugget", &NestedVariogram::nugget)
        .def("sill", &NestedVariogram::sill)
        .def("n_structures", &NestedVariogram::n_structures)
        .def("correlation", &NestedVariogram::correlation)
        .def("covariance", &NestedVariogram::covariance);

    // -------------------------------------------------------------------------
    // Local variogram
    // -------------------------------------------------------------------------
    py::class_<LocalVariogramParams>(m, "LocalVariogramParams")
        .def_readonly("sill",      &LocalVariogramParams::sill)
        .def_readonly("range",     &LocalVariogramParams::range)
        .def_readonly("nugget",    &LocalVariogramParams::nugget)
        .def_readonly("valid",     &LocalVariogramParams::valid)
        .def_readonly("rss",       &LocalVariogramParams::rss)
        .def_readonly("r_squared", &LocalVariogramParams::r_squared);

    m.def("fit_local_variogram_spherical",
        [](const std::vector<std::array<double, 3>>& samples,
           const std::vector<double>& values,
           const std::array<double, 3>& target,
           double search_radius, int min_points, int n_lags,
           double sill_guess, double range_guess, double nugget_guess) {
            Point3 tgt = {target[0], target[1], target[2]};
            return fit_local_variogram_spherical(
                points_to_vector(samples), values, tgt,
                search_radius, min_points, n_lags,
                sill_guess, range_guess, nugget_guess);
        },
        py::arg("samples"), py::arg("values"), py::arg("target"),
        py::arg("search_radius"),
        py::arg("min_points") = 10, py::arg("n_lags") = 5,
        py::arg("sill_guess") = -1.0,
        py::arg("range_guess") = -1.0,
        py::arg("nugget_guess") = -1.0);

    m.def("fit_local_variograms_batch",
        [](const std::vector<std::array<double, 3>>& samples,
           const std::vector<double>& values,
           const std::vector<std::array<double, 3>>& targets,
           double search_radius, int min_points, int n_lags) {
            return fit_local_variograms_batch(
                points_to_vector(samples), values,
                points_to_vector(targets),
                search_radius, min_points, n_lags);
        },
        py::arg("samples"), py::arg("values"), py::arg("targets"),
        py::arg("search_radius"),
        py::arg("min_points") = 10, py::arg("n_lags") = 5);

    // -------------------------------------------------------------------------
    // Linear solvers
    // -------------------------------------------------------------------------
    py::class_<GMRESConfig>(m, "GMRESConfig")
        .def(py::init<>())
        .def_readwrite("restart",  &GMRESConfig::restart)
        .def_readwrite("max_iter", &GMRESConfig::max_iter)
        .def_readwrite("tol",      &GMRESConfig::tol);

    m.def("solve_dense_linear_system",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> A_in,
           py::array_t<double, py::array::c_style | py::array::forcecast> b_in) {
            if (A_in.ndim() != 2 || A_in.shape(0) != A_in.shape(1))
                throw std::runtime_error("A must be square 2D");
            const int n = static_cast<int>(A_in.shape(0));
            if (b_in.ndim() != 1 || static_cast<int>(b_in.shape(0)) != n)
                throw std::runtime_error("b length must equal A rows");

            auto A_buf = A_in.request();
            auto b_buf = b_in.request();
            const double* A_ptr = static_cast<const double*>(A_buf.ptr);
            const double* b_ptr = static_cast<const double*>(b_buf.ptr);

            std::vector<double> A(A_ptr, A_ptr + static_cast<std::size_t>(n) * n);
            std::vector<double> b(b_ptr, b_ptr + n);
            std::string backend;
            const bool ok = solve_dense_linear_system(A, b, n, backend);

            py::array_t<double> x(n);
            auto x_buf = x.mutable_unchecked<1>();
            for (int i = 0; i < n; ++i) x_buf(i) = b[static_cast<std::size_t>(i)];

            py::dict out;
            out["success"] = ok;
            out["x"] = x;
            out["backend"] = backend;
            return out;
        },
        py::arg("A"), py::arg("b"));

    // -------------------------------------------------------------------------
    // RBF model
    // -------------------------------------------------------------------------
    py::class_<NativeExactRBFImplicitModel>(m, "NativeExactRBFImplicitModel")
        .def(py::init<>())
        .def(py::init<const std::string&, double, double>(),
             py::arg("kernel"), py::arg("epsilon"), py::arg("regularization"))
        .def_static("set_dense_direct_limit",
                    &NativeExactRBFImplicitModel::set_dense_direct_limit,
                    py::arg("limit"))
        .def_static("force_iterative_solver",
                    &NativeExactRBFImplicitModel::force_iterative_solver,
                    py::arg("force"))
        .def_static("set_iterative_solver", &NativeExactRBFImplicitModel::set_iterative_solver,
                    py::arg("restart") = 150, py::arg("max_iter") = 3000, py::arg("tol") = 1.0e-6)
        .def_static("set_dense_memory_limit_mb", &NativeExactRBFImplicitModel::set_dense_memory_limit_mb,
                    py::arg("megabytes"))
        .def_static("set_predict_backend", &NativeExactRBFImplicitModel::set_predict_backend, py::arg("backend"))
        .def_static("predict_backend", &NativeExactRBFImplicitModel::predict_backend)
        .def_static("set_sparse_solver", &NativeExactRBFImplicitModel::set_sparse_solver, py::arg("force"),
                    "Force the sparse Cholesky/Schur solver for compactly supported kernels (default: automatic above "
                    "the dense direct limit)")
        .def_static("sparse_solver_forced", &NativeExactRBFImplicitModel::sparse_solver_forced)
        .def("fit",
            [](NativeExactRBFImplicitModel& self,
               const std::vector<std::array<double, 3>>& centers,
               const std::vector<double>& values) {
                self.fit(points_to_vector(centers), values);
            },
            py::arg("centers"), py::arg("values"))
        .def("fit_with_reduction",
            [](NativeExactRBFImplicitModel& self,
               const std::vector<std::array<double, 3>>& centers,
               const std::vector<double>& values,
               double fitting_accuracy, double initial_ratio, double append_ratio) {
                self.fit_with_reduction(points_to_vector(centers), values,
                                        fitting_accuracy, initial_ratio, append_ratio);
            },
            py::arg("centers"), py::arg("values"),
            py::arg("fitting_accuracy") = 1e-4,
            py::arg("initial_ratio") = 0.1,
            py::arg("append_ratio") = 0.05)
        .def("predict",
            [](const NativeExactRBFImplicitModel& self,
               const std::vector<std::array<double, 3>>& points) {
                return self.predict(points_to_vector(points));
            },
            py::arg("points"))
        .def("is_fitted",       &NativeExactRBFImplicitModel::is_fitted)
        .def("n_centers",       &NativeExactRBFImplicitModel::n_centers)
        .def("iterative_info",  &NativeExactRBFImplicitModel::iterative_info,
             "3.5.0: last iterative fit: products, timings, H-matrix statistics, exact residual on sampled rows.")
        .def_static("set_hmatrix", &NativeExactRBFImplicitModel::set_hmatrix, py::arg("mode"), py::arg("tol") = 1.0e-9,
             py::arg("eta") = 1.5, py::arg("leaf_size") = 64, py::arg("near_memory_mb") = 2048.0,
             "3.5.0: H-matrix/ACA product in the iterative solver: mode \"auto\" (n >= 4000), \"on\" or \"off\".")
        .def_static("hmatrix_mode", &NativeExactRBFImplicitModel::hmatrix_mode)
        .def_static("set_preconditioner_form", &NativeExactRBFImplicitModel::set_preconditioner_form, py::arg("form"),
                    "Two-level preconditioner form: \"auto\" (default, 3.5.1), \"symmetric\", \"hybrid\" or \"additive\" (3.5.0).")
        .def_static("preconditioner_form", &NativeExactRBFImplicitModel::preconditioner_form)
        .def("weights",         &NativeExactRBFImplicitModel::weights,
             "Fitted coefficients: n kernel weights, then the drift coefficients (3.5.0).")
        .def("solver_backend",  &NativeExactRBFImplicitModel::solver_backend)
        .def("scalfmm_status",  &NativeExactRBFImplicitModel::scalfmm_status)
        .def("last_predict_backend", &NativeExactRBFImplicitModel::last_predict_backend)
        .def("sparse_solve_info", [](const NativeExactRBFImplicitModel& self) {
                py::dict out; const auto v = self.sparse_solve_info();
                if (v.size() == 5) {
                    out["nnz_lower"] = static_cast<long long>(v[0]); out["assembly_seconds"] = v[1];
                    out["factor_seconds"] = v[2]; out["solve_seconds"] = v[3]; out["relative_residual"] = v[4];
                }
                return out;
            })
        .def("kernel",          &NativeExactRBFImplicitModel::kernel)
        .def("epsilon",         &NativeExactRBFImplicitModel::epsilon)
        .def("regularization",  &NativeExactRBFImplicitModel::regularization)
        .def("set_drift", &NativeExactRBFImplicitModel::set_drift, py::arg("drift"),
             "Polynomial drift: \"linear\" [1,x,y,z] (default), \"constant\" [1] or \"none\" (positive definite kernels only). Call before fit.")
        .def("drift", &NativeExactRBFImplicitModel::drift)
        .def("set_kernel_shape", &NativeExactRBFImplicitModel::set_kernel_shape, py::arg("alpha"), py::arg("power"),
             "gen_cauchy only: C(h) = (1 + (h/b)^power)^(-alpha/power), epsilon = range where C = 0.04. Call before fit.");

    m.def("kernel_value", [](const std::string& kernel, double r, double epsilon) {
              return evaluate_kernel(kernel, r, epsilon);
          }, py::arg("kernel"), py::arg("r"), py::arg("epsilon"),
          "Value of an RBF / covariance kernel at distance r (kernel string as for NativeExactRBFImplicitModel).");

    py::class_<HMatrix>(m, "HMatrix",
        "3.5.0: hierarchical (H-matrix/ACA) approximation of the kernel matrix K_ij = phi(|x_i - x_j|).")
        .def(py::init([](const std::vector<Point3>& points, const std::string& kernel, double epsilon,
                         double tol, double eta, int leaf_size, double near_memory_mb) {
                 HMatrixOptions o; o.tol = tol; o.eta = eta; o.leaf_size = leaf_size; o.near_memory_mb = near_memory_mb;
                 py::gil_scoped_release release;
                 return new HMatrix(points, kernel, epsilon, o);
             }),
             py::arg("points"), py::arg("kernel"), py::arg("epsilon"), py::arg("tol") = 1.0e-9, py::arg("eta") = 1.5,
             py::arg("leaf_size") = 64, py::arg("near_memory_mb") = 2048.0)
        .def("apply", [](const HMatrix& h, const std::vector<double>& x) {
                 std::vector<double> y; { py::gil_scoped_release release; h.apply(x, y); } return y; },
             py::arg("x"), "y = K x (kernel matrix only, no regularisation).")
        .def("size", &HMatrix::size)
        .def("locate", &HMatrix::locate, py::arg("i"), py::arg("j"))
        .def("stats", [](const HMatrix& h) {
                 const auto& s = h.stats(); py::dict d;
                 d["n"] = s.n; d["n_lowrank"] = s.n_lowrank; d["n_near"] = s.n_near; d["max_rank"] = s.max_rank;
                 d["mean_rank"] = s.mean_rank; d["lowrank_mb"] = s.lowrank_mb; d["near_mb"] = s.near_mb;
                 d["near_stored"] = s.near_stored; d["build_seconds"] = s.build_seconds; d["compression"] = s.compression;
                 return d; });

    m.def("kernel_table_check", [](const std::string& kernel, double epsilon, double d_max, int samples) {
              const FastKernel fk(kernel, epsilon, d_max);
              std::mt19937_64 rng(12345);
              std::uniform_real_distribution<double> U(0.0, 1.0);
              double max_abs = 0.0, max_rel = 0.0;
              for (int i = 0; i < samples; ++i) {
                  // half log-uniform over [1e-9, 1] x d_max, half uniform over [0, d_max]
                  const double d = (i % 2) ? d_max * std::pow(10.0, -9.0 * U(rng)) : d_max * U(rng);
                  const double ex = fk.exact(d), tb = fk(d), e = std::fabs(tb - ex);
                  max_abs = std::max(max_abs, e);
                  max_rel = std::max(max_rel, e / std::max(1.0, std::fabs(ex)));
              }
              py::dict out;
              out["tabulated"] = fk.tabulated();
              out["max_abs_err"] = max_abs;
              out["max_rel_err"] = max_rel;
              return out;
          }, py::arg("kernel"), py::arg("epsilon"), py::arg("d_max"), py::arg("samples") = 200000,
          "3.5.0: compare the tabulated kernel used in the O(n^2) loops with the closed form on random distances.");

    // -------------------------------------------------------------------------
    // Implicit geology — trend + weighted model
    // -------------------------------------------------------------------------
    py::class_<NativeTrendImplicitModel>(m, "NativeTrendImplicitModel")
        .def(py::init<>())
        .def(py::init<const std::string&, double, double>(),
             py::arg("kernel"), py::arg("epsilon"), py::arg("regularization"))
        .def("set_trend_transform",
            [](NativeTrendImplicitModel& self,
               const std::vector<double>& rotation,
               const std::vector<double>& scale,
               const std::vector<double>& center) {
                if (rotation.size() != 9 || scale.size() != 3 || center.size() != 3)
                    throw std::runtime_error(
                        "set_trend_transform requires rotation size 9, scale size 3, center size 3");
                std::array<double, 9> r{};
                std::array<double, 3> s{}, c{};
                std::copy(rotation.begin(), rotation.end(), r.begin());
                std::copy(scale.begin(), scale.end(), s.begin());
                std::copy(center.begin(), center.end(), c.begin());
                self.set_trend_transform(r, s, c);
            },
            py::arg("rotation"), py::arg("scale"), py::arg("center"))
        .def("clear_trend_transform", &NativeTrendImplicitModel::clear_trend_transform)
        .def("has_trend_transform",   &NativeTrendImplicitModel::has_trend_transform)
        .def("fit_signed_constraints",
            [](NativeTrendImplicitModel& self,
               const std::vector<std::array<double, 3>>& points,
               const std::vector<double>& values) {
                self.fit_signed_constraints(points, values);
            },
            py::arg("points"), py::arg("signed_values"))
        .def("fit_signed_constraints",
            [](NativeTrendImplicitModel& self,
               const std::vector<std::array<double, 3>>& points,
               const std::vector<double>& values,
               const std::vector<double>& weights) {
                self.fit_signed_constraints(points, values, weights);
            },
            py::arg("points"), py::arg("signed_values"), py::arg("weights"))
        .def("predict",
            [](const NativeTrendImplicitModel& self,
               const std::vector<std::array<double, 3>>& points) {
                return self.predict(points);
            },
            py::arg("points"))
        .def("is_fitted",      &NativeTrendImplicitModel::is_fitted)
        .def("n_centers",      &NativeTrendImplicitModel::n_centers)
        .def("solver_backend", &NativeTrendImplicitModel::solver_backend)
        .def("scalfmm_status", &NativeTrendImplicitModel::scalfmm_status)
        .def("kernel",         &NativeTrendImplicitModel::kernel)
        .def("epsilon",        &NativeTrendImplicitModel::epsilon)
        .def("regularization", &NativeTrendImplicitModel::regularization);

    // -------------------------------------------------------------------------
    // Implicit geology — constraint + gradient model
    // -------------------------------------------------------------------------
    py::class_<NativeGeologicalImplicitModel>(m, "NativeGeologicalImplicitModel")
        .def(py::init<>())
        .def(py::init<const std::string&, double, double>(),
             py::arg("kernel"), py::arg("epsilon"), py::arg("regularization"))
        .def("set_trend_transform",
            [](NativeGeologicalImplicitModel& self,
               const std::vector<double>& rotation,
               const std::vector<double>& scale,
               const std::vector<double>& center) {
                if (rotation.size() != 9 || scale.size() != 3 || center.size() != 3)
                    throw std::runtime_error(
                        "set_trend_transform requires rotation size 9, scale size 3, center size 3");
                std::array<double, 9> r{};
                std::array<double, 3> s{}, c{};
                std::copy(rotation.begin(), rotation.end(), r.begin());
                std::copy(scale.begin(), scale.end(), s.begin());
                std::copy(center.begin(), center.end(), c.begin());
                self.set_trend_transform(r, s, c);
            },
            py::arg("rotation"), py::arg("scale"), py::arg("center"))
        .def("clear_trend_transform", &NativeGeologicalImplicitModel::clear_trend_transform)
        .def("has_trend_transform",   &NativeGeologicalImplicitModel::has_trend_transform)
        .def("set_use_polynomial_drift",
             &NativeGeologicalImplicitModel::set_use_polynomial_drift, py::arg("use"))
        .def("use_polynomial_drift",
             &NativeGeologicalImplicitModel::use_polynomial_drift)
        .def("fit_constraints",
            [](NativeGeologicalImplicitModel& self,
               const std::vector<std::array<double, 3>>& interface_pts,
               const std::vector<std::array<double, 3>>& inside_pts,
               const std::vector<std::array<double, 3>>& outside_pts,
               double inside_value, double outside_value,
               double interface_weight, double inside_weight, double outside_weight) {
                self.fit_constraints(interface_pts, inside_pts, outside_pts,
                                     inside_value, outside_value,
                                     interface_weight, inside_weight, outside_weight);
            },
            py::arg("interface_points"), py::arg("inside_points"), py::arg("outside_points"),
            py::arg("inside_value") = 1.0, py::arg("outside_value") = -1.0,
            py::arg("interface_weight") = 2.0,
            py::arg("inside_weight") = 1.0,
            py::arg("outside_weight") = 1.0)
        .def("fit_constraints_with_gradients",
            [](NativeGeologicalImplicitModel& self,
               const std::vector<std::array<double, 3>>& interface_pts,
               const std::vector<std::array<double, 3>>& inside_pts,
               const std::vector<std::array<double, 3>>& outside_pts,
               const std::vector<std::array<double, 3>>& gradient_pts,
               const std::vector<std::array<double, 3>>& gradient_normals,
               const std::vector<double>& gradient_targets,
               double inside_value, double outside_value,
               double interface_weight, double inside_weight, double outside_weight,
               double gradient_weight) {
                self.fit_constraints_with_gradients(
                    interface_pts, inside_pts, outside_pts,
                    gradient_pts, gradient_normals, gradient_targets,
                    inside_value, outside_value,
                    interface_weight, inside_weight, outside_weight,
                    gradient_weight);
            },
            py::arg("interface_points"), py::arg("inside_points"), py::arg("outside_points"),
            py::arg("gradient_points"), py::arg("gradient_normals"), py::arg("gradient_targets"),
            py::arg("inside_value") = 1.0, py::arg("outside_value") = -1.5,
            py::arg("interface_weight") = 3.0,
            py::arg("inside_weight") = 1.0,
            py::arg("outside_weight") = 1.25,
            py::arg("gradient_weight") = 0.35)
        .def("predict",
            [](const NativeGeologicalImplicitModel& self,
               const std::vector<std::array<double, 3>>& points) {
                return self.predict(points);
            },
            py::arg("points"))
        .def("is_fitted",      &NativeGeologicalImplicitModel::is_fitted)
        .def("n_centers",      &NativeGeologicalImplicitModel::n_centers)
        .def("solver_backend", &NativeGeologicalImplicitModel::solver_backend)
        .def("kernel",         &NativeGeologicalImplicitModel::kernel)
        .def("epsilon",        &NativeGeologicalImplicitModel::epsilon)
        .def("regularization", &NativeGeologicalImplicitModel::regularization);

    // -------------------------------------------------------------------------
    // Geostatistics structures
    // -------------------------------------------------------------------------
    py::class_<IndicatorVariogram>(m, "IndicatorVariogram")
        .def(py::init<>())
        .def_readwrite("category",         &IndicatorVariogram::category)
        .def_readwrite("range_major",      &IndicatorVariogram::range_major)
        .def_readwrite("range_minor",      &IndicatorVariogram::range_minor)
        .def_readwrite("range_vertical",   &IndicatorVariogram::range_vertical)
        .def_readwrite("azimuth_deg",      &IndicatorVariogram::azimuth_deg)
        .def_readwrite("dip_deg",          &IndicatorVariogram::dip_deg)
        .def_readwrite("plunge_deg",       &IndicatorVariogram::plunge_deg)
        .def_readwrite("sill",             &IndicatorVariogram::sill)
        .def_readwrite("nugget",           &IndicatorVariogram::nugget);

    py::class_<CrossVariogram>(m, "CrossVariogram")
        .def(py::init<>())
        .def_readwrite("category_i", &CrossVariogram::category_i)
        .def_readwrite("category_j", &CrossVariogram::category_j)
        .def_readwrite("sill",       &CrossVariogram::sill)
        .def_readwrite("nugget",     &CrossVariogram::nugget)
        .def_readwrite("range",      &CrossVariogram::range);

    py::class_<LMCStructure>(m, "LMCStructure")
        .def(py::init<>())
        .def_readwrite("family", &LMCStructure::family)
        .def_readwrite("structure_id",      &LMCStructure::structure_id)
        .def_readwrite("range_major",       &LMCStructure::range_major)
        .def_readwrite("range_minor",       &LMCStructure::range_minor)
        .def_readwrite("range_vertical",    &LMCStructure::range_vertical)
        .def_readwrite("azimuth_deg",       &LMCStructure::azimuth_deg)
        .def_readwrite("dip_deg",           &LMCStructure::dip_deg)
        .def_readwrite("plunge_deg",        &LMCStructure::plunge_deg)
        .def_readwrite("covariance_matrix", &LMCStructure::covariance_matrix);

    py::class_<LinearModelOfCoregionalization>(m, "LinearModelOfCoregionalization")
        .def(py::init<>())
        .def(py::pickle(
            [](const LinearModelOfCoregionalization& lmc) {
                py::list struct_list;
                for (const auto& s : lmc.structures()) {
                    py::dict d;
                    d["family"] = s.family;
                    d["structure_id"] = s.structure_id;
                    d["range_major"]    = s.range_major;
                    d["range_minor"]    = s.range_minor;
                    d["range_vertical"] = s.range_vertical;
                    d["azimuth_deg"]    = s.azimuth_deg;
                    d["dip_deg"]        = s.dip_deg;
                    d["plunge_deg"]     = s.plunge_deg;
                    py::list cov;
                    for (const auto& row : s.covariance_matrix) {
                        py::list row_list;
                        for (double v : row) row_list.append(v);
                        cov.append(row_list);
                    }
                    d["covariance_matrix"] = cov;
                    struct_list.append(d);
                }
                return py::make_tuple(struct_list);
            },
            [](py::tuple t) {
                if (t.size() != 1)
                    throw std::runtime_error("Invalid pickle state for LMC");
                py::list struct_list = t[0].cast<py::list>();
                LinearModelOfCoregionalization lmc;
                for (py::handle item : struct_list) {
                    py::dict d = item.cast<py::dict>();
                    LMCStructure s;
                    if(d.contains("family"))s.family=d["family"].cast<std::string>();
                    if(d.contains("structure_id"))s.structure_id=d["structure_id"].cast<int>();
                    s.range_major    = d["range_major"].cast<double>();
                    s.range_minor    = d["range_minor"].cast<double>();
                    s.range_vertical = d["range_vertical"].cast<double>();
                    s.azimuth_deg    = d["azimuth_deg"].cast<double>();
                    s.dip_deg        = d["dip_deg"].cast<double>();
                    s.plunge_deg     = d["plunge_deg"].cast<double>();
                    py::list cov_list = d["covariance_matrix"].cast<py::list>();
                    size_t n_cat = cov_list.size();
                    s.covariance_matrix.resize(n_cat);
                    for (size_t i = 0; i < n_cat; ++i) {
                        py::list row = cov_list[i].cast<py::list>();
                        if(row.size()!=n_cat)throw std::invalid_argument("LMC pickle matrix must be square");
                        s.covariance_matrix[i].resize(n_cat);
                        for (size_t j = 0; j < n_cat; ++j)
                            s.covariance_matrix[i][j] = row[j].cast<double>();
                    }
                    lmc.add_structure(s);
                }
                return lmc;
            }))
        .def("add_structure", &LinearModelOfCoregionalization::add_structure)
        .def("covariance", &LinearModelOfCoregionalization::covariance,
             py::arg("cat_i"), py::arg("cat_j"), py::arg("h_scaled"))
        .def("structures", &LinearModelOfCoregionalization::structures)
        .def("is_positive_definite",
             [](const LinearModelOfCoregionalization& lmc) {
                 return is_positive_definite(lmc);
             });

    py::class_<CoKrigingResult>(m, "CoKrigingResult")
        .def(py::init<>())
        .def_readwrite("estimates",            &CoKrigingResult::estimates)
        .def_readwrite("variances",            &CoKrigingResult::variances)
        .def_readwrite("n_samples",            &CoKrigingResult::n_samples)
        .def_readwrite("samples_per_category", &CoKrigingResult::samples_per_category);

    py::class_<SICoKResult>(m, "SICoKResult")
        .def(py::init<>())
        .def_readwrite("realizations",   &SICoKResult::realizations)
        .def_readwrite("n_targets",      &SICoKResult::n_targets)
        .def_readwrite("n_realizations", &SICoKResult::n_realizations);

    py::class_<Direction>(m, "Direction")
        .def(py::init<>())
        .def_readwrite("azimuth_deg",   &Direction::azimuth_deg)
        .def_readwrite("dip_deg",       &Direction::dip_deg)
        .def_readwrite("tolerance_deg", &Direction::tolerance_deg);

    py::class_<VariogramLag>(m, "VariogramLag")
        .def(py::init<>())
        .def_readwrite("lag_center",  &VariogramLag::lag_center)
        .def_readwrite("gamma_value", &VariogramLag::gamma_value)
        .def_readwrite("n_pairs",     &VariogramLag::n_pairs);

    py::class_<VariogramResult>(m, "VariogramResult")
        .def(py::init<>())
        .def_readwrite("categories", &VariogramResult::categories)
        .def_readwrite("lags",       &VariogramResult::lags);

    py::class_<VariogramFitResult>(m, "VariogramFitResult")
        .def(py::init<>())
        .def_readwrite("sill",           &VariogramFitResult::sill)
        .def_readwrite("range",          &VariogramFitResult::range)
        .def_readwrite("nugget",         &VariogramFitResult::nugget)
        .def_readwrite("rss",            &VariogramFitResult::rss)
        .def_readwrite("r_squared",      &VariogramFitResult::r_squared)
        .def_readwrite("model_type",     &VariogramFitResult::model_type)
        .def_readwrite("fitted_values",  &VariogramFitResult::fitted_values);

    py::class_<CrossValidationResult>(m, "CrossValidationResult")
        .def(py::init<>())
        .def_readwrite("errors",                  &CrossValidationResult::errors)
        .def_readwrite("squared_errors",          &CrossValidationResult::squared_errors)
        .def_readwrite("mean_error",              &CrossValidationResult::mean_error)
        .def_readwrite("mean_squared_error",      &CrossValidationResult::mean_squared_error)
        .def_readwrite("root_mean_squared_error", &CrossValidationResult::root_mean_squared_error)
        .def_readwrite("mean_absolute_error",     &CrossValidationResult::mean_absolute_error);

    // -------------------------------------------------------------------------
    // Core helpers
    // -------------------------------------------------------------------------
    m.def("spherical_covariance", &spherical_covariance,
          py::arg("h_scaled"), py::arg("sill"), py::arg("nugget"));

    m.def("anisotropic_scaled_distance",
        [](const std::array<double, 3>& a,
           const std::array<double, 3>& b,
           double rmaj, double rmin, double rver,
           double az, double dip, double plunge) {
            Point3 pa = {a[0], a[1], a[2]};
            Point3 pb = {b[0], b[1], b[2]};
            return anisotropic_scaled_distance(pa, pb, rmaj, rmin, rver, az, dip, plunge);
        },
        py::arg("a"), py::arg("b"),
        py::arg("range_major"), py::arg("range_minor"), py::arg("range_vertical"),
        py::arg("azimuth_deg"), py::arg("dip_deg"), py::arg("plunge_deg"));

    // -------------------------------------------------------------------------
    // Ordinary kriging
    // -------------------------------------------------------------------------
    m.def("ordinary_kriging",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           int max_neighbors, int min_neighbors,
           double search_radius, double sill, double nugget) {
            OKResult res = ordinary_kriging(points_to_vector(sample_xyz), sample_values,
                                            points_to_vector(target_xyz),
                                            max_neighbors, min_neighbors,
                                            search_radius, sill, nugget);
            py::dict out;
            out["estimate"]  = res.estimate;
            out["variance"]  = res.variance;
            out["n_samples"] = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("max_neighbors") = 20, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0, py::arg("sill") = 1.0, py::arg("nugget") = 0.1);

    m.def("ordinary_kriging_with_range",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           int max_neighbors, int min_neighbors,
           double search_radius, double variogram_range,
           double sill, double nugget) {
            OKResult res = ordinary_kriging_with_range(points_to_vector(sample_xyz), sample_values,
                                                       points_to_vector(target_xyz),
                                                       max_neighbors, min_neighbors,
                                                       search_radius, variogram_range,
                                                       sill, nugget);
            py::dict out;
            out["estimate"]  = res.estimate;
            out["variance"]  = res.variance;
            out["n_samples"] = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("max_neighbors") = 20, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0, py::arg("variogram_range") = 100.0,
        py::arg("sill") = 1.0, py::arg("nugget") = 0.1);

    m.def("ordinary_kriging_anisotropic",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           const IndicatorVariogram& vg,
           int max_neighbors, int min_neighbors, double search_radius) {
            OKResult res = ordinary_kriging_anisotropic(points_to_vector(sample_xyz), sample_values,
                                                        points_to_vector(target_xyz),
                                                        vg, max_neighbors, min_neighbors,
                                                        search_radius);
            py::dict out;
            out["estimate"]  = res.estimate;
            out["variance"]  = res.variance;
            out["n_samples"] = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("variogram"), py::arg("max_neighbors") = 20,
        py::arg("min_neighbors") = 5, py::arg("search_radius") = 200.0);

    m.def("ordinary_kriging_local",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           double search_radius, int max_neighbors, int min_neighbors,
           int min_points_for_variogram, int n_lags,
           double sill_guess, double range_guess, double nugget_guess) {
            OKResult res = ordinary_kriging_local(
                points_to_vector(sample_xyz), sample_values,
                points_to_vector(target_xyz),
                search_radius, max_neighbors, min_neighbors,
                min_points_for_variogram, n_lags,
                sill_guess, range_guess, nugget_guess);
            py::dict out;
            out["estimate"]  = res.estimate;
            out["variance"]  = res.variance;
            out["n_samples"] = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("search_radius"), py::arg("max_neighbors") = 20,
        py::arg("min_neighbors") = 5, py::arg("min_points_for_variogram") = 10,
        py::arg("n_lags") = 5, py::arg("sill_guess") = -1.0,
        py::arg("range_guess") = -1.0, py::arg("nugget_guess") = -1.0);

    m.def("ordinary_kriging_nested",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           const NestedVariogram& vg,
           int max_neighbors, int min_neighbors, double search_radius) {
            OKResult res = ordinary_kriging_nested(points_to_vector(sample_xyz), sample_values,
                                                   points_to_vector(target_xyz),
                                                   vg, max_neighbors, min_neighbors,
                                                   search_radius);
            py::dict out;
            out["estimate"]  = res.estimate;
            out["variance"]  = res.variance;
            out["n_samples"] = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("variogram"), py::arg("max_neighbors") = 20,
        py::arg("min_neighbors") = 5, py::arg("search_radius") = 200.0);

    // -------------------------------------------------------------------------
    // Indicator kriging
    // -------------------------------------------------------------------------
    m.def("indicator_kriging",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<int>& sample_categories,
           const std::vector<std::array<double, 3>>& target_xyz,
           const std::vector<int>& categories,
           int max_neighbors, int min_neighbors,
           double search_radius, double sill, double nugget) {
            IKResult res = indicator_kriging(points_to_vector(sample_xyz), sample_categories,
                                             points_to_vector(target_xyz), categories,
                                             max_neighbors, min_neighbors,
                                             search_radius, sill, nugget);
            const py::ssize_t n = static_cast<py::ssize_t>(res.n_targets);
            const py::ssize_t c = static_cast<py::ssize_t>(res.n_categories);
            py::array_t<double> probs({n, c});
            auto p = probs.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = 0; j < c; ++j)
                    p(i, j) = res.probabilities[static_cast<size_t>(i) * res.n_categories +
                                                static_cast<size_t>(j)];
            py::dict out;
            out["probabilities"] = probs;
            out["categories"]    = res.categories;
            out["n_samples"]     = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_categories"), py::arg("target_xyz"),
        py::arg("categories"),
        py::arg("max_neighbors") = 20, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0, py::arg("sill") = 1.0, py::arg("nugget") = 0.1);

    m.def("indicator_kriging_anisotropic",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<int>& sample_categories,
           const std::vector<std::array<double, 3>>& target_xyz,
           const std::vector<int>& categories,
           const std::vector<IndicatorVariogram>& variograms,
           int max_neighbors, int min_neighbors, double search_radius) {
            IKResult res = indicator_kriging_anisotropic(
                points_to_vector(sample_xyz), sample_categories,
                points_to_vector(target_xyz), categories, variograms,
                max_neighbors, min_neighbors, search_radius);
            const py::ssize_t n = static_cast<py::ssize_t>(res.n_targets);
            const py::ssize_t c = static_cast<py::ssize_t>(res.n_categories);
            py::array_t<double> probs({n, c});
            auto p = probs.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = 0; j < c; ++j)
                    p(i, j) = res.probabilities[static_cast<size_t>(i) * res.n_categories +
                                                static_cast<size_t>(j)];
            py::dict out;
            out["probabilities"] = probs;
            out["categories"]    = res.categories;
            out["n_samples"]     = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_categories"), py::arg("target_xyz"),
        py::arg("categories"), py::arg("variograms"),
        py::arg("max_neighbors") = 24, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0);

    // -------------------------------------------------------------------------
    // Sequential Gaussian Simulation
    // -------------------------------------------------------------------------
    m.def("run_sgs",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           int n_realizations, int max_neighbors, int min_neighbors,
           double search_radius, double sill, double nugget, int seed) {
            SGSResult res = run_sgs(points_to_vector(sample_xyz), sample_values,
                                    points_to_vector(target_xyz),
                                    n_realizations, max_neighbors, min_neighbors,
                                    search_radius, sill, nugget, seed);
            const py::ssize_t n = static_cast<py::ssize_t>(res.n_targets);
            const py::ssize_t r = static_cast<py::ssize_t>(res.n_realizations);
            py::array_t<double> sims({n, r});
            auto arr = sims.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = 0; j < r; ++j)
                    arr(i, j) = res.simulations[static_cast<size_t>(j) * res.n_targets +
                                                static_cast<size_t>(i)];
            return sims;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("n_realizations") = 100, py::arg("max_neighbors") = 24,
        py::arg("min_neighbors") = 5, py::arg("search_radius") = 200.0,
        py::arg("sill") = 1.0, py::arg("nugget") = 0.05, py::arg("seed") = 42);

    m.def("run_sgs_with_range",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           int n_realizations, int max_neighbors, int min_neighbors,
           double search_radius, double variogram_range,
           double sill, double nugget, int seed, bool standardize) {
            SGSResult res = run_sgs_with_range(points_to_vector(sample_xyz), sample_values,
                                               points_to_vector(target_xyz),
                                               n_realizations, max_neighbors, min_neighbors,
                                               search_radius, variogram_range,
                                               sill, nugget, seed, standardize);
            const py::ssize_t n = static_cast<py::ssize_t>(res.n_targets);
            const py::ssize_t r = static_cast<py::ssize_t>(res.n_realizations);
            py::array_t<double> sims({n, r});
            auto arr = sims.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = 0; j < r; ++j)
                    arr(i, j) = res.simulations[static_cast<size_t>(j) * res.n_targets +
                                                static_cast<size_t>(i)];
            return sims;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("n_realizations") = 100, py::arg("max_neighbors") = 24,
        py::arg("min_neighbors") = 5, py::arg("search_radius") = 200.0,
        py::arg("variogram_range") = 100.0, py::arg("sill") = 1.0,
        py::arg("nugget") = 0.05, py::arg("seed") = 42, py::arg("standardize") = true);

    m.def("run_sgs_summary",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz,
           int n_realizations, int max_neighbors, int min_neighbors,
           double search_radius, double sill, double nugget,
           double cutoff, int seed) {
            SGSSummaryResult res = run_sgs_summary(points_to_vector(sample_xyz), sample_values,
                                                   points_to_vector(target_xyz),
                                                   n_realizations, max_neighbors, min_neighbors,
                                                   search_radius, sill, nugget,
                                                   cutoff, seed);
            py::dict out;
            out["mean"]              = res.mean;
            out["p10"]               = res.p10;
            out["p50"]               = res.p50;
            out["p90"]               = res.p90;
            out["stddev"]            = res.stddev;
            out["prob_above_cutoff"] = res.prob_above_cutoff;
            out["n_samples"]         = res.n_samples;
            out["n_targets"]         = res.n_targets;
            out["n_realizations"]    = res.n_realizations;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"),
        py::arg("n_realizations") = 20, py::arg("max_neighbors") = 24,
        py::arg("min_neighbors") = 5, py::arg("search_radius") = 200.0,
        py::arg("sill") = 1.0, py::arg("nugget") = 0.05,
        py::arg("cutoff") = 0.0, py::arg("seed") = 2026);

    // -------------------------------------------------------------------------
    // Sequential Indicator Simulation
    // -------------------------------------------------------------------------
    m.def("run_sis",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<int>& sample_categories,
           const std::vector<std::array<double, 3>>& target_xyz,
           const std::vector<int>& categories,
           int n_realizations, int max_neighbors, int min_neighbors,
           double search_radius, double sill, double nugget, int seed) {
            SISResult res = run_sis(points_to_vector(sample_xyz), sample_categories,
                                    points_to_vector(target_xyz), categories,
                                    n_realizations, max_neighbors, min_neighbors,
                                    search_radius, sill, nugget, seed);
            const py::ssize_t n = static_cast<py::ssize_t>(res.n_targets);
            const py::ssize_t r = static_cast<py::ssize_t>(res.n_realizations);
            py::array_t<int> sims({n, r});
            auto arr = sims.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = 0; j < r; ++j)
                    arr(i, j) = res.simulations[static_cast<size_t>(j) * res.n_targets +
                                                static_cast<size_t>(i)];
            return sims;
        },
        py::arg("sample_xyz"), py::arg("sample_categories"), py::arg("target_xyz"),
        py::arg("categories"),
        py::arg("n_realizations") = 100, py::arg("max_neighbors") = 20,
        py::arg("min_neighbors") = 5, py::arg("search_radius") = 200.0,
        py::arg("sill") = 1.0, py::arg("nugget") = 0.1, py::arg("seed") = 42);

    m.def("run_sis_anisotropic",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<int>& sample_categories,
           const std::vector<std::array<double, 3>>& target_xyz,
           const std::vector<int>& categories,
           const std::vector<IndicatorVariogram>& variograms,
           int n_realizations, int max_neighbors, int min_neighbors,
           double search_radius, int seed) {
            SISResult res = run_sis_anisotropic(
                points_to_vector(sample_xyz), sample_categories,
                points_to_vector(target_xyz), categories, variograms,
                n_realizations, max_neighbors, min_neighbors,
                search_radius, seed);
            const py::ssize_t n = static_cast<py::ssize_t>(res.n_targets);
            const py::ssize_t r = static_cast<py::ssize_t>(res.n_realizations);
            py::array_t<int> sims({n, r});
            auto arr = sims.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = 0; j < r; ++j)
                    arr(i, j) = res.simulations[static_cast<size_t>(j) * res.n_targets +
                                                static_cast<size_t>(i)];
            return sims;
        },
        py::arg("sample_xyz"), py::arg("sample_categories"), py::arg("target_xyz"),
        py::arg("categories"), py::arg("variograms"),
        py::arg("n_realizations") = 100, py::arg("max_neighbors") = 24,
        py::arg("min_neighbors") = 5, py::arg("search_radius") = 200.0,
        py::arg("seed") = 42);

    // -------------------------------------------------------------------------
    // Experimental variograms
    // -------------------------------------------------------------------------
    m.def("compute_direct_indicator_variograms_fast",
        [](const std::vector<std::array<double, 3>>& points,
           const std::vector<int>& categories,
           const std::vector<int>& all_categories,
           const std::vector<Direction>& directions,
           int n_lags, double max_lag, double lag_width, int max_pairs_per_lag) {
            auto res = compute_direct_indicator_variograms_fast(
                points_to_vector(points), categories, all_categories,
                directions, n_lags, max_lag, lag_width, max_pairs_per_lag);
            py::list out;
            for (const auto& vr : res) {
                py::dict d;
                d["categories"] = vr.categories;
                py::list lags;
                for (const auto& lag : vr.lags) {
                    py::dict ld;
                    ld["lag_center"]  = lag.lag_center;
                    ld["gamma_value"] = lag.gamma_value;
                    ld["n_pairs"]     = lag.n_pairs;
                    lags.append(ld);
                }
                d["lags"] = lags;
                out.append(d);
            }
            return out;
        },
        py::arg("points"), py::arg("categories"), py::arg("all_categories"),
        py::arg("directions"), py::arg("n_lags"),
        py::arg("max_lag") = -1.0, py::arg("lag_width") = -1.0,
        py::arg("max_pairs_per_lag") = 500000);

    m.def("compute_cross_indicator_variograms_fast",
        [](const std::vector<std::array<double, 3>>& points,
           const std::vector<int>& categories,
           const std::vector<std::pair<int, int>>& cross_pairs,
           const std::vector<Direction>& directions,
           int n_lags, double max_lag, double lag_width, int max_pairs_per_lag) {
            auto res = compute_cross_indicator_variograms_fast(
                points_to_vector(points), categories, cross_pairs,
                directions, n_lags, max_lag, lag_width, max_pairs_per_lag);
            py::list out;
            for (const auto& vr : res) {
                py::dict d;
                d["categories"] = vr.categories;
                py::list lags;
                for (const auto& lag : vr.lags) {
                    py::dict ld;
                    ld["lag_center"]  = lag.lag_center;
                    ld["gamma_value"] = lag.gamma_value;
                    ld["n_pairs"]     = lag.n_pairs;
                    lags.append(ld);
                }
                d["lags"] = lags;
                out.append(d);
            }
            return out;
        },
        py::arg("points"), py::arg("categories"), py::arg("cross_pairs"),
        py::arg("directions"), py::arg("n_lags"),
        py::arg("max_lag") = -1.0, py::arg("lag_width") = -1.0,
        py::arg("max_pairs_per_lag") = 500000);

    // -------------------------------------------------------------------------
    // Variogram model fitting
    // -------------------------------------------------------------------------
    m.def("fit_spherical_variogram",
        [](const std::vector<double>& lags,
           const std::vector<double>& gamma,
           double sill_guess, double range_guess, double nugget_guess) {
            return fit_spherical_variogram(lags, gamma, sill_guess, range_guess, nugget_guess);
        },
        py::arg("lag_centers"), py::arg("gamma_values"),
        py::arg("sill_guess") = -1.0, py::arg("range_guess") = -1.0,
        py::arg("nugget_guess") = -1.0);

    m.def("fit_exponential_variogram",
        [](const std::vector<double>& lags,
           const std::vector<double>& gamma,
           double sill_guess, double range_guess, double nugget_guess) {
            return fit_exponential_variogram(lags, gamma, sill_guess, range_guess, nugget_guess);
        },
        py::arg("lag_centers"), py::arg("gamma_values"),
        py::arg("sill_guess") = -1.0, py::arg("range_guess") = -1.0,
        py::arg("nugget_guess") = -1.0);

    m.def("fit_gaussian_variogram",
        [](const std::vector<double>& lags,
           const std::vector<double>& gamma,
           double sill_guess, double range_guess, double nugget_guess) {
            return fit_gaussian_variogram(lags, gamma, sill_guess, range_guess, nugget_guess);
        },
        py::arg("lag_centers"), py::arg("gamma_values"),
        py::arg("sill_guess") = -1.0, py::arg("range_guess") = -1.0,
        py::arg("nugget_guess") = -1.0);

    m.def("auto_fit_variogram",
        [](const std::vector<double>& lags,
           const std::vector<double>& gamma,
           double sill_guess, double range_guess, double nugget_guess) {
            return auto_fit_variogram(lags, gamma, sill_guess, range_guess, nugget_guess);
        },
        py::arg("lag_centers"), py::arg("gamma_values"),
        py::arg("sill_guess") = -1.0, py::arg("range_guess") = -1.0,
        py::arg("nugget_guess") = -1.0);

    m.def("fit_nested_variogram",
        [](const std::vector<double>& lags,
           const std::vector<double>& gamma,
           const std::vector<double>& weights,
           int max_structures, double sill_guess, double range_guess) {
            return fit_nested_variogram(lags, gamma, weights,
                                        max_structures, sill_guess, range_guess);
        },
        py::arg("lag_centers"), py::arg("gamma_exp"),
        py::arg("weights") = std::vector<double>{},
        py::arg("max_structures") = 1,
        py::arg("sill_guess") = -1.0, py::arg("range_guess") = -1.0);

    // -------------------------------------------------------------------------
    // Cross-validation
    // -------------------------------------------------------------------------
    m.def("cross_validate_ordinary_kriging",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           int max_neighbors, int min_neighbors,
           double search_radius, double sill, double nugget) {
            return cross_validate_ordinary_kriging(points_to_vector(sample_xyz), sample_values,
                                                   max_neighbors, min_neighbors,
                                                   search_radius, sill, nugget);
        },
        py::arg("sample_xyz"), py::arg("sample_values"),
        py::arg("max_neighbors") = 20, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0, py::arg("sill") = 1.0, py::arg("nugget") = 0.1);

    m.def("cross_validate_indicator_kriging",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<int>& sample_categories,
           const std::vector<int>& categories,
           int max_neighbors, int min_neighbors,
           double search_radius, double sill, double nugget) {
            return cross_validate_indicator_kriging(points_to_vector(sample_xyz), sample_categories,
                                                    categories, max_neighbors, min_neighbors,
                                                    search_radius, sill, nugget);
        },
        py::arg("sample_xyz"), py::arg("sample_categories"), py::arg("categories"),
        py::arg("max_neighbors") = 20, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0, py::arg("sill") = 1.0, py::arg("nugget") = 0.1);

    // -------------------------------------------------------------------------
    // Cokriging + LMC
    // -------------------------------------------------------------------------
    m.def("build_lmc_from_variograms",
        [](const std::vector<CrossVariogram>& variograms, int n_structures) {
            return build_lmc_from_variograms(variograms, n_structures);
        },
        py::arg("variograms"), py::arg("n_structures") = 2);

    m.def("is_positive_definite", &is_positive_definite, py::arg("lmc"));

    m.def("fit_lmc_from_experimental_variograms",
        [](const std::vector<VariogramResult>& direct,
           const std::vector<VariogramResult>& cross,
           const std::vector<int>& categories,
           int n_structures, const std::string& model_type,
           double tolerance, int max_iter) {
            return fit_lmc_from_experimental_variograms(
                direct, cross, categories,
                n_structures, model_type, tolerance, max_iter);
        },
        py::arg("direct_variograms"), py::arg("cross_variograms"),
        py::arg("categories"),
        py::arg("n_structures") = 2,
        py::arg("model_type") = "spherical",
        py::arg("tolerance") = 1e-6,
        py::arg("max_iter") = 100);

    m.def("ordinary_cokriging",
        [](const std::vector<std::array<double, 3>>& sample_points,
           const std::vector<int>& sample_categories,
           const std::vector<std::array<double, 3>>& target_points,
           const std::vector<int>& categories,
           const LinearModelOfCoregionalization& lmc,
           int max_neighbors, int min_neighbors, double search_radius) {
            CoKrigingResult res = ordinary_cokriging(
                points_to_vector(sample_points), sample_categories,
                points_to_vector(target_points), categories, lmc,
                max_neighbors, min_neighbors, search_radius);

            const py::ssize_t n = static_cast<py::ssize_t>(target_points.size());
            const py::ssize_t c = static_cast<py::ssize_t>(categories.size());
            py::array_t<double> estimates({n, c});
            py::array_t<double> variances({n, c});
            auto est = estimates.mutable_unchecked<2>();
            auto var = variances.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) {
                for (py::ssize_t j = 0; j < c; ++j) {
                    est(i, j) = res.estimates[static_cast<size_t>(i) * static_cast<size_t>(c) +
                                              static_cast<size_t>(j)];
                    var(i, j) = res.variances[static_cast<size_t>(i) * static_cast<size_t>(c) +
                                              static_cast<size_t>(j)];
                }
            }
            py::dict out;
            out["estimates"]            = estimates;
            out["variances"]            = variances;
            out["n_samples"]            = res.n_samples;
            out["samples_per_category"] = res.samples_per_category;
            return out;
        },
        py::arg("sample_points"), py::arg("sample_categories"),
        py::arg("target_points"), py::arg("categories"), py::arg("lmc"),
        py::arg("max_neighbors") = 20, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0);

    m.def("sequential_indicator_cokriging",
        [](const std::vector<std::array<double, 3>>& sample_points,
           const std::vector<int>& sample_categories,
           const std::vector<std::array<double, 3>>& target_points,
           const std::vector<int>& categories,
           const LinearModelOfCoregionalization& lmc,
           int n_realizations, int max_neighbors, int min_neighbors,
           double search_radius, int seed) {
            SICoKResult res = sequential_indicator_cokriging(
                points_to_vector(sample_points), sample_categories,
                points_to_vector(target_points), categories, lmc,
                n_realizations, max_neighbors, min_neighbors,
                search_radius, seed);

            const py::ssize_t n = static_cast<py::ssize_t>(res.n_targets);
            const py::ssize_t r = static_cast<py::ssize_t>(res.n_realizations);
            py::array_t<int> sims({n, r});
            auto arr = sims.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = 0; j < r; ++j)
                    arr(i, j) = res.realizations[static_cast<size_t>(j)][static_cast<size_t>(i)];

            py::dict out;
            out["simulations"]    = sims;
            out["n_targets"]      = res.n_targets;
            out["n_realizations"] = res.n_realizations;
            return out;
        },
        py::arg("sample_points"), py::arg("sample_categories"),
        py::arg("target_points"), py::arg("categories"), py::arg("lmc"),
        py::arg("n_realizations") = 100,
        py::arg("max_neighbors") = 20, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0, py::arg("seed") = 42);

    // -------------------------------------------------------------------------
    // Block kriging + change of support
    // -------------------------------------------------------------------------
    py::class_<Block>(m, "Block")
        .def(py::init<>())
        .def_readwrite("xmin", &Block::xmin)
        .def_readwrite("xmax", &Block::xmax)
        .def_readwrite("ymin", &Block::ymin)
        .def_readwrite("ymax", &Block::ymax)
        .def_readwrite("zmin", &Block::zmin)
        .def_readwrite("zmax", &Block::zmax)
        .def_readwrite("discretization", &Block::discretization);

    m.def("block_kriging",
        [](const std::vector<std::array<double, 3>>& sample_xyz,
           const std::vector<double>& sample_values,
           const std::vector<Block>& blocks,
           int max_neighbors, int min_neighbors,
           double search_radius, double variogram_range,
           double sill, double nugget) {
            BlockKrigingResult res = block_kriging(
                points_to_vector(sample_xyz), sample_values, blocks,
                max_neighbors, min_neighbors, search_radius,
                variogram_range, sill, nugget);
            py::dict out;
            out["estimate"]  = res.estimate;
            out["variance"]  = res.variance;
            out["n_samples"] = res.n_samples;
            return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("blocks"),
        py::arg("max_neighbors") = 24, py::arg("min_neighbors") = 5,
        py::arg("search_radius") = 200.0, py::arg("variogram_range") = 100.0,
        py::arg("sill") = 1.0, py::arg("nugget") = 0.05);

    py::class_<ChangeOfSupportResult>(m, "ChangeOfSupportResult")
        .def(py::init<>())
        .def_readwrite("block_mean",     &ChangeOfSupportResult::block_mean)
        .def_readwrite("block_variance", &ChangeOfSupportResult::block_variance)
        .def_readwrite("variance_reduction_factor",
                       &ChangeOfSupportResult::variance_reduction_factor);

    m.def("compute_variance_reduction_factor",
          &compute_variance_reduction_factor,
          py::arg("block"), py::arg("range"), py::arg("sill"), py::arg("nugget"));

    m.def("affine_correction",
          &affine_correction,
          py::arg("point_mean"), py::arg("point_variance"), py::arg("vrf"));

    m.def("lognormal_correction",
          &lognormal_correction,
          py::arg("point_mean"), py::arg("point_variance"), py::arg("vrf"));

    m.def("apply_change_of_support_to_realizations",
        [](const std::vector<std::vector<double>>& point_realizations,
           double point_mean, double point_variance,
           const Block& block,
           double range, double sill, double nugget,
           const std::string& method) {
            return apply_change_of_support_to_realizations(
                point_realizations, point_mean, point_variance,
                block, range, sill, nugget, method);
        },
        py::arg("point_realizations"), py::arg("point_mean"), py::arg("point_variance"),
        py::arg("block"), py::arg("range"), py::arg("sill"), py::arg("nugget"),
        py::arg("method") = "affine");

    // -------------------------------------------------------------------------
    // SNESIM (Multiple-Point Statistics)
    // Fully integrated here, no separate binding file needed.
    // -------------------------------------------------------------------------
    py::class_<SnesimNativeConfig>(m, "SnesimNativeConfig")
        .def(py::init<>())
        .def_readwrite("max_conditioning_nodes",
                       &SnesimNativeConfig::max_conditioning_nodes)
        .def_readwrite("min_replicates",
                       &SnesimNativeConfig::min_replicates)
        .def_readwrite("n_multigrids",
                       &SnesimNativeConfig::n_multigrids)
        .def_readwrite("radius_x", &SnesimNativeConfig::radius_x)
        .def_readwrite("radius_y", &SnesimNativeConfig::radius_y)
        .def_readwrite("radius_z", &SnesimNativeConfig::radius_z)
        .def_readwrite("servo_lambda",
                       &SnesimNativeConfig::servo_lambda)
        .def_readwrite("seed", &SnesimNativeConfig::seed);

    py::class_<SnesimDiagnostics>(m, "SnesimDiagnostics")
        .def(py::init<>())
        .def_readwrite("simulated_nodes",
                       &SnesimDiagnostics::simulated_nodes)
        .def_readwrite("conditioning_queries",
                       &SnesimDiagnostics::conditioning_queries)
        .def_readwrite("backoff_events",
                       &SnesimDiagnostics::backoff_events);

    m.attr("SNESIM_UNKNOWN") = py::int_(SNESIM_UNKNOWN);

    m.def("snesim_native_static",
        [](py::array_t<std::int32_t,
                       py::array::c_style | py::array::forcecast> training_image,
           py::array_t<std::int32_t,
                       py::array::c_style | py::array::forcecast> hard_data,
           const std::vector<std::int32_t>& categories,
           const std::vector<double>&       target_proportions,
           const SnesimNativeConfig&        cfg,
           const std::vector<int>& simulation_mask,const std::vector<double>& soft_probabilities,double soft_tau) {

            std::vector<std::int64_t> tshape;
            tshape.reserve(static_cast<std::size_t>(training_image.ndim()));
            for (py::ssize_t i = 0; i < training_image.ndim(); ++i)
                tshape.push_back(static_cast<std::int64_t>(training_image.shape(i)));

            std::vector<std::int64_t> sshape;
            sshape.reserve(static_cast<std::size_t>(hard_data.ndim()));
            for (py::ssize_t i = 0; i < hard_data.ndim(); ++i)
                sshape.push_back(static_cast<std::int64_t>(hard_data.shape(i)));

            auto ti_buf  = training_image.request();
            auto hd_buf  = hard_data.request();
            auto* ti_ptr = static_cast<std::int32_t*>(ti_buf.ptr);
            auto* hd_ptr = static_cast<std::int32_t*>(hd_buf.ptr);

            std::vector<std::int32_t> ti_vec(ti_ptr, ti_ptr + ti_buf.size);
            std::vector<std::int32_t> hd_vec(hd_ptr, hd_ptr + hd_buf.size);

            SnesimNativeResult r;
            {py::gil_scoped_release release;
            r = snesim_native_static(ti_vec, tshape,
                                          hd_vec, sshape,
                                          categories, target_proportions, cfg,simulation_mask,soft_probabilities,soft_tau);}

            std::vector<py::ssize_t> py_shape;
            py_shape.reserve(r.shape.size());
            for (auto v : r.shape)
                py_shape.push_back(static_cast<py::ssize_t>(v));

            py::array_t<std::int32_t> out(py_shape);
            auto bo = out.request();
            std::copy(r.values.begin(), r.values.end(),
                      static_cast<std::int32_t*>(bo.ptr));

            py::dict diagnostics;
            diagnostics["simulated_nodes"]      = r.diagnostics.simulated_nodes;
            diagnostics["conditioning_queries"] = r.diagnostics.conditioning_queries;
            diagnostics["backoff_events"]       = r.diagnostics.backoff_events;
            diagnostics["fallback_to_prior"] = r.diagnostics.fallback_to_prior;
            diagnostics["replicate_sum"] = r.diagnostics.replicate_sum;

            py::dict out_dict;
            out_dict["values"]      = out;
            out_dict["diagnostics"] = diagnostics;
            return out_dict;
        },
        py::arg("training_image"),
        py::arg("hard_data"),
        py::arg("categories"),
        py::arg("target_proportions") = std::vector<double>{},
        py::arg("config")             = SnesimNativeConfig{},
        py::arg("simulation_mask")    = std::vector<int>{},
        py::arg("soft_probabilities") = std::vector<double>{},py::arg("soft_tau")=1.
    );

    m.def("stage11_snesim_status", []() {
        const auto s = snesim_runtime_status();
        py::dict d;
        d["stage"]                  = s.stage;
        d["module"]                 = s.module;
        d["native_static_template"] = s.native_static_template;
        d["python_advanced_path"]   = s.python_advanced_path;
        return d;
    });

    // ---------------------------------------------------------------- 3.3.4 GSLIB-convention kriging and simulation
    py::class_<GSLIBStructure>(m, "GSLIBStructure",
        "One nested variogram structure: type spherical|exponential|gaussian, contribution cc, ranges, and GSLIB "
        "angles (azimuth clockwise from north, dip positive down, rake).")
        .def(py::init([](const std::string& type, double cc, double range_major, double range_minor, double range_vertical,
                         double azimuth, double dip, double rake) {
                 GSLIBStructure s; s.type = type; s.cc = cc; s.range_major = range_major; s.range_minor = range_minor;
                 s.range_vertical = range_vertical; s.azimuth = azimuth; s.dip = dip; s.rake = rake; return s; }),
             py::arg("type") = "spherical", py::arg("cc") = 1.0, py::arg("range_major") = 100.0, py::arg("range_minor") = 100.0,
             py::arg("range_vertical") = 100.0, py::arg("azimuth") = 0.0, py::arg("dip") = 0.0, py::arg("rake") = 0.0)
        .def_readwrite("type", &GSLIBStructure::type).def_readwrite("cc", &GSLIBStructure::cc)
        .def_readwrite("range_major", &GSLIBStructure::range_major).def_readwrite("range_minor", &GSLIBStructure::range_minor)
        .def_readwrite("range_vertical", &GSLIBStructure::range_vertical).def_readwrite("azimuth", &GSLIBStructure::azimuth)
        .def_readwrite("dip", &GSLIBStructure::dip).def_readwrite("rake", &GSLIBStructure::rake);

    py::class_<GSLIBVariogram>(m, "GSLIBVariogram", "Nugget plus nested structures (GSLIB conventions).")
        .def(py::init([](double nugget, const std::vector<GSLIBStructure>& structures) {
                 GSLIBVariogram v; v.nugget = nugget; v.structures = structures; return v; }),
             py::arg("nugget") = 0.0, py::arg("structures") = std::vector<GSLIBStructure>{})
        .def_readwrite("nugget", &GSLIBVariogram::nugget).def_readwrite("structures", &GSLIBVariogram::structures)
        .def("sill", &GSLIBVariogram::sill)
        .def("covariance", [](const GSLIBVariogram& v, const std::array<double, 3>& a, const std::array<double, 3>& b) {
            return gslib_covariance(v, a, b); }, py::arg("a"), py::arg("b"));

    py::class_<SearchEllipsoid>(m, "SearchEllipsoid", "Search ellipsoid: radii and GSLIB angles.")
        .def(py::init([](double radius_major, double radius_minor, double radius_vertical, double azimuth, double dip, double rake) {
                 SearchEllipsoid s; s.radius_major = radius_major; s.radius_minor = radius_minor; s.radius_vertical = radius_vertical;
                 s.azimuth = azimuth; s.dip = dip; s.rake = rake; return s; }),
             py::arg("radius_major") = 200.0, py::arg("radius_minor") = 200.0, py::arg("radius_vertical") = 200.0,
             py::arg("azimuth") = 0.0, py::arg("dip") = 0.0, py::arg("rake") = 0.0)
        .def_readwrite("radius_major", &SearchEllipsoid::radius_major).def_readwrite("radius_minor", &SearchEllipsoid::radius_minor)
        .def_readwrite("radius_vertical", &SearchEllipsoid::radius_vertical).def_readwrite("azimuth", &SearchEllipsoid::azimuth)
        .def_readwrite("dip", &SearchEllipsoid::dip).def_readwrite("rake", &SearchEllipsoid::rake);

    m.def("gslib_rotation", &gslib_rotation, py::arg("azimuth"), py::arg("dip"), py::arg("rake"),
          py::arg("range_major"), py::arg("range_minor"), py::arg("range_vertical"),
          "GSLIB setrot matrix (row-major 3x3) mapping a lag to range-scaled (major, minor, vertical) coordinates.");

    m.def("kriging_gslib",
        [](const std::vector<std::array<double, 3>>& sample_xyz, const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz, const GSLIBVariogram& vario, const SearchEllipsoid& search,
           int max_data, int min_data, const std::string& kriging_type, double mean) {
            KrigingGSLIBResult r;
            { py::gil_scoped_release nogil;
              r = kriging_gslib(points_to_vector(sample_xyz), sample_values, points_to_vector(target_xyz), vario, search,
                                max_data, min_data, kriging_type, mean); }
            py::dict out; out["estimate"] = r.estimate; out["variance"] = r.variance; out["n_samples"] = r.n_samples; return out;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"), py::arg("variogram"), py::arg("search"),
        py::arg("max_data") = 24, py::arg("min_data") = 1, py::arg("kriging_type") = "ordinary", py::arg("mean") = 0.0,
        "Ordinary or simple point kriging with a nested GSLIB variogram and an anisotropic search.");

    m.def("normal_score",
        [](const std::vector<double>& values, double zmin, double zmax) {
            const auto t = normal_score_table(values, zmin, zmax);
            std::vector<double> y(values.size());
            for (size_t i = 0; i < values.size(); ++i) y[i] = ns_forward(t, values[i]);
            return y;
        }, py::arg("values"), py::arg("zmin") = std::numeric_limits<double>::quiet_NaN(),
        py::arg("zmax") = std::numeric_limits<double>::quiet_NaN(), "Normal scores of values (ties share a score).");

    m.def("sgs_gslib",
        [](const std::vector<std::array<double, 3>>& sample_xyz, const std::vector<double>& sample_values,
           const std::vector<std::array<double, 3>>& target_xyz, const GSLIBVariogram& vario, const SearchEllipsoid& search,
           int n_realizations, int max_data, int max_nodes, int min_data, const std::string& kriging_type, bool normal_score,
           double zmin, double zmax, int multigrid_levels, std::uint64_t seed) {
            SimulationGSLIBResult r;
            { py::gil_scoped_release nogil;
              r = sgs_gslib(points_to_vector(sample_xyz), sample_values, points_to_vector(target_xyz), vario, search, n_realizations,
                            max_data, max_nodes, min_data, kriging_type, normal_score, zmin, zmax, multigrid_levels, seed); }
            const py::ssize_t n = static_cast<py::ssize_t>(r.n_targets), R = static_cast<py::ssize_t>(r.n_realizations);
            py::array_t<double> sims({n, R});
            auto a = sims.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) for (py::ssize_t j = 0; j < R; ++j) a(i, j) = r.values[static_cast<size_t>(j) * r.n_targets + static_cast<size_t>(i)];
            return sims;
        },
        py::arg("sample_xyz"), py::arg("sample_values"), py::arg("target_xyz"), py::arg("variogram"), py::arg("search"),
        py::arg("n_realizations") = 10, py::arg("max_data") = 16, py::arg("max_nodes") = 16, py::arg("min_data") = 0,
        py::arg("kriging_type") = "simple", py::arg("normal_score") = true,
        py::arg("zmin") = std::numeric_limits<double>::quiet_NaN(), py::arg("zmax") = std::numeric_limits<double>::quiet_NaN(),
        py::arg("multigrid_levels") = 3, py::arg("seed") = 69069,
        "Sequential Gaussian simulation (GSLIB sgsim conventions): normal-score transform, simple kriging, multiple-grid path. "
        "Returns an (n_targets, n_realizations) array in original units.");

    m.def("sis_gslib",
        [](const std::vector<std::array<double, 3>>& sample_xyz, const std::vector<int>& sample_categories,
           const std::vector<std::array<double, 3>>& target_xyz, const std::vector<int>& categories,
           const std::vector<GSLIBVariogram>& variograms, const SearchEllipsoid& search, int n_realizations, int max_data,
           int max_nodes, int multigrid_levels, std::uint64_t seed) {
            SimulationGSLIBResult r;
            { py::gil_scoped_release nogil;
              r = sis_gslib(points_to_vector(sample_xyz), sample_categories, points_to_vector(target_xyz), categories, variograms,
                            search, n_realizations, max_data, max_nodes, multigrid_levels, seed); }
            const py::ssize_t n = static_cast<py::ssize_t>(r.n_targets), R = static_cast<py::ssize_t>(r.n_realizations);
            py::array_t<int> sims({n, R});
            auto a = sims.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) for (py::ssize_t j = 0; j < R; ++j) a(i, j) = r.categories[static_cast<size_t>(j) * r.n_targets + static_cast<size_t>(i)];
            return sims;
        },
        py::arg("sample_xyz"), py::arg("sample_categories"), py::arg("target_xyz"), py::arg("categories"), py::arg("variograms"),
        py::arg("search"), py::arg("n_realizations") = 10, py::arg("max_data") = 16, py::arg("max_nodes") = 16,
        py::arg("multigrid_levels") = 3, py::arg("seed") = 69069,
        "Sequential indicator simulation (GSLIB sisim conventions): simple IK on global proportions with one standardised "
        "variogram per category, order-relation correction, multiple-grid path.");
}
