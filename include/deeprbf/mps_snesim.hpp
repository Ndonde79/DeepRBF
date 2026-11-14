// deeprbf/mps_snesim.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "deeprbf/kernels.hpp"   // Point3

namespace deeprbf {

// ---------------------------------------------------------------------------
// SNESIM configuration
// Mirrors the shape of the other DeepRBF config structs
// (IndicatorVariogram, LMCStructure, Direction, ...).
// ---------------------------------------------------------------------------
struct SnesimNativeConfig {
    int           max_conditioning_nodes = 32;
    int           min_replicates         = 10;
    int           n_multigrids           = 3;
    double        radius_x               = 8.0;
    double        radius_y               = 8.0;
    double        radius_z               = 4.0;
    double        servo_lambda           = 0.0;   // in [0,1); 0 disables servosystem
    std::uint64_t seed                   = 31001;
};

struct SnesimDiagnostics {
    std::int64_t simulated_nodes      = 0;
    std::int64_t conditioning_queries = 0;
    std::int64_t backoff_events       = 0;
    std::int64_t fallback_to_prior    = 0;
    std::int64_t replicate_sum        = 0;
};

struct SnesimNativeResult {
    std::vector<std::int32_t> values;   // row-major, product(shape)
    std::vector<std::int64_t> shape;    // {ny,nx} or {nz,ny,nx}
    SnesimDiagnostics         diagnostics;
};

// Sentinel for "unsimulated" / unassigned nodes.
constexpr std::int32_t SNESIM_UNKNOWN = static_cast<std::int32_t>(0x80000000u);

// ---------------------------------------------------------------------------
// Static-template, hard-conditioned SNESIM
//
//   training_image      : row-major flat array, size = product(training_shape)
//   training_shape      : {ny, nx} for 2D, {nz, ny, nx} for 3D
//   hard_data           : row-major flat, size = product(simulation_shape)
//                         values are category labels or SNESIM_UNKNOWN
//   simulation_shape    : same dimensionality as training_shape
//   categories          : explicit category labels
//   target_proportions  : same length as categories; empty -> use TI prior
// ---------------------------------------------------------------------------
SnesimNativeResult snesim_native_static(
    const std::vector<std::int32_t>& training_image,
    const std::vector<std::int64_t>& training_shape,
    const std::vector<std::int32_t>& hard_data,
    const std::vector<std::int64_t>& simulation_shape,
    const std::vector<std::int32_t>& categories,
    const std::vector<double>&       target_proportions,
    const SnesimNativeConfig&        cfg,
    const std::vector<int>&          simulation_mask = {},
    const std::vector<double>&       soft_probabilities = {}, double soft_tau=1.
);

// Capability probe (exported as stage11_snesim_status() in Python).
struct SnesimRuntimeStatus {
    std::string stage;
    std::string module;
    bool        native_static_template;
    bool        python_advanced_path;
};
SnesimRuntimeStatus snesim_runtime_status();

} // namespace deeprbf
