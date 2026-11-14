// DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
// 3.3.4: kriging and sequential simulation with GSLIB conventions.
//
// Variogram: nugget + nested structures (spherical, exponential, gaussian), each with its own ranges and
// GSLIB rotation (azimuth clockwise from north, dip positive down, rake), so models transfer from GSLIB,
// Supervisor or Leapfrog variography without conversion. The variogram range is independent of the search.
// Search: an anisotropic ellipsoid (radii and angles), with separate limits on original data and on
// previously simulated nodes, as in GSLIB sgsim/sisim.
// SGS: normal-score transform with linear tails, simple kriging (default) or ordinary kriging, multiple-grid
// path, realisations in parallel. SIS: simple indicator kriging on global proportions, order-relation
// correction, multiple-grid path. Each realisation is reproducible from (seed, realisation index) alone.
#pragma once

#include "deeprbf/core.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace deeprbf {

struct GSLIBStructure {
    std::string type = "spherical";            // spherical | exponential | gaussian
    double cc = 1.0;                           // contribution (partial sill)
    double range_major = 100.0, range_minor = 100.0, range_vertical = 100.0;
    double azimuth = 0.0, dip = 0.0, rake = 0.0;  // degrees, GSLIB ang1, -ang2 (dip down positive), ang3
};

struct GSLIBVariogram {
    double nugget = 0.0;
    std::vector<GSLIBStructure> structures;
    double sill() const;
};

struct SearchEllipsoid {
    double radius_major = 200.0, radius_minor = 200.0, radius_vertical = 200.0;
    double azimuth = 0.0, dip = 0.0, rake = 0.0;
};

// GSLIB setrot: rotation that maps a lag into the (major, minor, vertical) frame, scaled by 1/range.
std::array<double, 9> gslib_rotation(double azimuth, double dip, double rake, double r_major, double r_minor, double r_vertical);
double gslib_covariance(const GSLIBVariogram& v, const Point3& a, const Point3& b);

struct KrigingGSLIBResult {
    std::vector<double> estimate, variance;
    std::vector<int> n_samples;
};

// kriging_type: "ordinary" or "simple" (simple uses `mean`).
KrigingGSLIBResult kriging_gslib(const std::vector<Point3>& sample_xyz, const std::vector<double>& sample_values,
                                 const std::vector<Point3>& target_xyz, const GSLIBVariogram& vario,
                                 const SearchEllipsoid& search, int max_data, int min_data,
                                 const std::string& kriging_type, double mean);

struct NormalScoreTable {
    std::vector<double> z, y;   // sorted original values and their normal scores
    double zmin = 0.0, zmax = 0.0;
};
NormalScoreTable normal_score_table(const std::vector<double>& values, double zmin, double zmax);
double ns_forward(const NormalScoreTable& t, double z);
double ns_back(const NormalScoreTable& t, double y);

struct SimulationGSLIBResult {
    std::vector<double> values;          // n_targets x n_realizations, realisation-major
    std::vector<int> categories;         // SIS only
    size_t n_targets = 0, n_realizations = 0;
    std::vector<int> multigrid_level_counts;
};

// SGS. vario is in normal-score units (sill 1) when normal_score is true.
SimulationGSLIBResult sgs_gslib(const std::vector<Point3>& sample_xyz, const std::vector<double>& sample_values,
                                const std::vector<Point3>& target_xyz, const GSLIBVariogram& vario,
                                const SearchEllipsoid& search, int n_realizations, int max_data, int max_nodes,
                                int min_data, const std::string& kriging_type, bool normal_score, double zmin,
                                double zmax, int multigrid_levels, std::uint64_t seed);

// SIS. One standardised variogram (sill 1) per category, scaled internally by p(1-p).
SimulationGSLIBResult sis_gslib(const std::vector<Point3>& sample_xyz, const std::vector<int>& sample_categories,
                                const std::vector<Point3>& target_xyz, const std::vector<int>& categories,
                                const std::vector<GSLIBVariogram>& varios, const SearchEllipsoid& search,
                                int n_realizations, int max_data, int max_nodes, int multigrid_levels,
                                std::uint64_t seed);

}  // namespace deeprbf
