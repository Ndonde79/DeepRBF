#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include "deeprbf/audited_engine.hpp"
namespace py=pybind11;
using namespace deeprbf;
void bind_audited_engine(py::module_& m){
    py::class_<SparseLatentResult>(m,"SparseLatentResult").def_readonly("values",&SparseLatentResult::values)
        .def_readonly("precision_nonzeros",&SparseLatentResult::precision_nonzeros)
        .def_readonly("estimated_workspace_bytes",&SparseLatentResult::estimated_workspace_bytes)
        .def_readonly("maximum_local_residual",&SparseLatentResult::maximum_local_residual);
    m.def("vecchia_truncated_native",&vecchia_truncated_native,py::arg("coordinates"),py::arg("lower"),py::arg("upper"),
        py::arg("neighbors")=12,py::arg("sweeps")=30,py::arg("seed")=42,py::arg("nugget")=.01,py::arg("max_workspace_mb")=512.,py::call_guard<py::gil_scoped_release>());
    m.def("gibbs_truncated_native",&gibbs_truncated_native,py::arg("precision"),py::arg("lower"),py::arg("upper"),py::arg("sweeps")=30,py::arg("seed")=42,py::call_guard<py::gil_scoped_release>());
    py::class_<NativeCategoricalModel>(m,"NativeCategoricalModel")
        .def(py::init<const std::vector<Point3>&,const std::vector<int>&,const std::vector<int>&,const std::vector<double>&,
             const std::vector<int>&,const std::vector<int>&,const Point3&,const Axes3&,const Point3&,const NeighborhoodSpec&>())
        .def("predict",&NativeCategoricalModel::predict,py::arg("targets"),py::arg("hierarchical")=false,py::arg("prior_mix")=0.,py::call_guard<py::gil_scoped_release>());
    py::class_<NativeIndicatorResult>(m,"NativeIndicatorResult")
        .def_readonly("probabilities",&NativeIndicatorResult::probabilities)
        .def_readonly("simulations",&NativeIndicatorResult::simulations)
        .def_readonly("direct_hard_neighbors",&NativeIndicatorResult::direct_hard_neighbors)
        .def_readonly("direct_hard_holes",&NativeIndicatorResult::direct_hard_holes)
        .def_readonly("supported",&NativeIndicatorResult::supported)
        .def_readonly("prior_only_draws",&NativeIndicatorResult::prior_only_draws)
        .def_readonly("maximum_solve_residual",&NativeIndicatorResult::maximum_solve_residual)
        .def_readonly("n_nodes",&NativeIndicatorResult::n_nodes)
        .def_readonly("n_categories",&NativeIndicatorResult::n_categories)
        .def_readonly("n_realizations",&NativeIndicatorResult::n_realizations);
    m.def("indicator_lithology_native",&indicator_lithology_native,py::arg("coordinates"),py::arg("hard"),py::arg("prior"),
        py::arg("max_neighbors")=20,py::arg("realizations")=0,py::arg("seed")=42,py::arg("candidate_neighbors")=96,
        py::arg("search_radius")=2.,py::arg("hole_ids")=std::vector<int>{},py::arg("max_per_hole")=0,py::arg("min_holes")=0,py::call_guard<py::gil_scoped_release>());
    py::class_<NativeGaussianResult>(m,"NativeGaussianResult").def_readonly("simulations",&NativeGaussianResult::simulations)
        .def_readonly("conditional_mean",&NativeGaussianResult::conditional_mean).def_readonly("conditional_covariance",&NativeGaussianResult::conditional_covariance)
        .def_readonly("minimum_conditional_eigenvalue",&NativeGaussianResult::minimum_conditional_eigenvalue).def_readonly("conditioning_residual",&NativeGaussianResult::conditioning_residual)
        .def_readonly("n_targets",&NativeGaussianResult::n_targets).def_readonly("n_realizations",&NativeGaussianResult::n_realizations);
    m.def("simulate_gaussian_native",&simulate_gaussian_native,py::arg("xyz"),py::arg("values"),py::arg("targets"),py::arg("covariance"),py::arg("realizations"),py::arg("seed"),py::arg("mean")=0.,py::arg("dense_limit")=2048,py::call_guard<py::gil_scoped_release>());
    py::class_<CovarianceSpec>(m,"CovarianceSpec").def(py::init<>())
        .def_readwrite("range_convention",&CovarianceSpec::range_convention)
        .def_readwrite("family",&CovarianceSpec::family).def_readwrite("ranges",&CovarianceSpec::ranges)
        .def_readwrite("axes",&CovarianceSpec::axes).def_readwrite("sill",&CovarianceSpec::sill).def_readwrite("nugget",&CovarianceSpec::nugget)
        .def("validate",&CovarianceSpec::validate)
        .def("covariance",[](const CovarianceSpec& c,const Point3&a,const Point3&b,bool same){c.validate();for(double v:a)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite point");for(double v:b)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite point");return c.covariance(a,b,same);},py::arg("a"),py::arg("b"),py::arg("same_observation")=false);
    py::class_<NeighborhoodSpec>(m,"NeighborhoodSpec").def(py::init<>())
        .def_readwrite("radii",&NeighborhoodSpec::radii).def_readwrite("axes",&NeighborhoodSpec::axes)
        .def_readwrite("max_neighbors",&NeighborhoodSpec::max_neighbors).def_readwrite("min_neighbors",&NeighborhoodSpec::min_neighbors)
        .def_readwrite("min_holes",&NeighborhoodSpec::min_holes).def_readwrite("max_per_hole",&NeighborhoodSpec::max_per_hole)
        .def_readwrite("max_per_octant",&NeighborhoodSpec::max_per_octant).def_readwrite("candidate_neighbors",&NeighborhoodSpec::candidate_neighbors)
        .def_readwrite("restrict_domain",&NeighborhoodSpec::restrict_domain).def("validate",&NeighborhoodSpec::validate);
    py::class_<RotatedSupport>(m,"RotatedSupport").def(py::init<>())
        .def_readwrite("center",&RotatedSupport::center).def_readwrite("size",&RotatedSupport::size)
        .def_readwrite("axes",&RotatedSupport::axes).def_readwrite("discretization",&RotatedSupport::discretization).def_readwrite("domain",&RotatedSupport::domain);
    py::class_<SolveDiagnostics>(m,"SolveDiagnostics")
        .def_readonly("minimum_weight",&SolveDiagnostics::minimum_weight).def_readonly("maximum_weight",&SolveDiagnostics::maximum_weight)
        .def_readonly("weight_sum",&SolveDiagnostics::weight_sum).def_readonly("weight_l1",&SolveDiagnostics::weight_l1)
        .def_readonly("negative_weights",&SolveDiagnostics::negative_weights)
        .def_readonly("status",&SolveDiagnostics::status).def_readonly("rank",&SolveDiagnostics::rank).def_readonly("dimension",&SolveDiagnostics::dimension)
        .def_readonly("n_samples",&SolveDiagnostics::n_samples).def_readonly("n_holes",&SolveDiagnostics::n_holes)
        .def_readonly("relative_residual",&SolveDiagnostics::relative_residual).def_readonly("reciprocal_condition",&SolveDiagnostics::reciprocal_condition)
        .def_readonly("regularization",&SolveDiagnostics::regularization).def_readonly("success",&SolveDiagnostics::success);
    py::class_<AuditedResult>(m,"AuditedResult").def_readonly("estimate",&AuditedResult::estimate).def_readonly("variance",&AuditedResult::variance).def_readonly("diagnostics",&AuditedResult::diagnostics);
    py::class_<NeighborhoodSelection>(m,"NeighborhoodSelection").def_readonly("indices",&NeighborhoodSelection::indices).def_readonly("n_holes",&NeighborhoodSelection::n_holes).def_readonly("supported",&NeighborhoodSelection::supported);
    py::class_<NativeCategoricalResult>(m,"NativeCategoricalResult").def_readonly("probabilities",&NativeCategoricalResult::probabilities)
        .def_readonly("support_holes",&NativeCategoricalResult::support_holes).def_readonly("supported",&NativeCategoricalResult::supported)
        .def_readonly("exact_conditioned",&NativeCategoricalResult::exact_conditioned).def_readonly("n_targets",&NativeCategoricalResult::n_targets).def_readonly("n_categories",&NativeCategoricalResult::n_categories);
    m.def("select_neighborhood",&select_neighborhood,py::arg("xyz"),py::arg("holes"),py::arg("domains"),py::arg("target"),py::arg("target_domain"),py::arg("search"),py::call_guard<py::gil_scoped_release>());
    m.def("audited_ordinary_kriging",&audited_ordinary_kriging,py::arg("xyz"),py::arg("values"),py::arg("holes"),py::arg("domains"),py::arg("targets"),py::arg("covariance"),py::arg("search"),py::arg("regularization")=0.,py::arg("residual_tolerance")=1e-9,py::arg("observation_support")=std::vector<RotatedSupport>{},py::call_guard<py::gil_scoped_release>());
    m.def("categorical_rbf_native",&categorical_rbf_native,py::arg("xyz"),py::arg("labels"),py::arg("holes"),py::arg("support"),py::arg("categories"),py::arg("category_domains"),py::arg("targets"),py::arg("ranges"),py::arg("axes"),py::arg("gate_scale"),py::arg("search"),py::arg("hierarchical")=false,py::arg("prior_mix")=0.,py::call_guard<py::gil_scoped_release>());
}
