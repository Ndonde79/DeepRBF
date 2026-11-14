// deeprbf/gc_lmc_fit.cpp
#include "deeprbf/gc_lmc_fit.hpp"
#include "deeprbf/kernels.hpp"
#include "deeprbf/lmc_psd.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace deeprbf {

namespace {

struct LagData {
    double distance;
    Eigen::MatrixXd cov_matrix;
};

std::vector<LagData> build_lag_data(
    const std::vector<VariogramResult>& direct,
    const std::vector<VariogramResult>& cross,
    const std::vector<int>& categories)
{
    int n_cat = static_cast<int>(categories.size());
    std::unordered_map<int, int> cat2idx;
    for (int i = 0; i < n_cat; ++i) cat2idx[categories[i]] = i;

    if (direct.empty() || direct[0].lags.empty()) return {};
    std::vector<double> lag_centers;
    lag_centers.reserve(direct[0].lags.size());
    for (const auto& l : direct[0].lags) lag_centers.push_back(l.lag_center);

    std::vector<LagData> result(lag_centers.size());
    for (size_t l = 0; l < lag_centers.size(); ++l) {
        result[l].distance = lag_centers[l];
        result[l].cov_matrix = Eigen::MatrixXd::Zero(n_cat, n_cat);

        // Direct (diagonal)
        for (int c = 0; c < n_cat; ++c) {
            int cat = categories[c];
            for (const auto& d : direct) {
                if (d.categories.size() == 2 &&
                    d.categories[0] == cat && d.categories[1] == cat) {
                    if (l < d.lags.size()) {
                        double gamma = d.lags[l].gamma_value;
                        double sill = 0.0;
                        int n_last = std::max(1, static_cast<int>(d.lags.size()) / 4);
                        for (size_t k = d.lags.size() - n_last; k < d.lags.size(); ++k)
                            sill += d.lags[k].gamma_value;
                        sill /= n_last;
                        result[l].cov_matrix(c, c) = sill - gamma;
                    }
                    break;
                }
            }
        }

        // Cross (off-diagonal)
        for (const auto& cr : cross) {
            if (cr.categories.size() != 2) continue;
            int ci = cr.categories[0], cj = cr.categories[1];
            auto it_i = cat2idx.find(ci), it_j = cat2idx.find(cj);
            if (it_i == cat2idx.end() || it_j == cat2idx.end()) continue;
            int i = it_i->second, j = it_j->second;

            for (const auto& cr_lag : cr.lags) {
                double tol = (lag_centers.size() > 1)
                    ? 0.5 * (lag_centers[1] - lag_centers[0])
                    : 1.0;
                if (std::abs(cr_lag.lag_center - lag_centers[l]) <= tol) {
                    double cross_sill=0.;
                    const size_t tail=std::max<size_t>(1,cr.lags.size()/4);
                    for(size_t k=cr.lags.size()-tail;k<cr.lags.size();++k)cross_sill+=cr.lags[k].gamma_value/tail;
                    result[l].cov_matrix(i, j) = cross_sill-cr_lag.gamma_value;
                    result[l].cov_matrix(j, i) = cross_sill-cr_lag.gamma_value;
                    break;
                }
            }
        }
    }
    return result;
}

double correlation_value(double h, const std::string& model, double range) {
    if (range <= 0.0) return 0.0;
    double r = h / range;
    if (model == "spherical") {
        if (r >= 1.0) return 0.0;
        return 1.0 - (1.5 * r - 0.5 * r * r * r);
    } else if (model == "exponential") {
        return std::exp(-3.0 * r);
    } else if (model == "gaussian") {
        return std::exp(-3.0 * r * r);
    }
    return 0.0;
}

bool fit_structure(const std::vector<LagData>& lag_data,
                   const std::string& model,
                   double range,
                   Eigen::MatrixXd& B,
                   double& rss) {
    if (lag_data.empty()) return false;
    int n_cat = static_cast<int>(lag_data[0].cov_matrix.rows());
    double sum_rho2 = 0.0;
    Eigen::MatrixXd sum_rho_C = Eigen::MatrixXd::Zero(n_cat, n_cat);
    for (const auto& lag : lag_data) {
        double rho = correlation_value(lag.distance, model, range);
        if (rho <= 0.0) continue;
        sum_rho2 += rho * rho;
        sum_rho_C += rho * lag.cov_matrix;
    }
    if (sum_rho2 < 1e-12) return false;
    B = sum_rho_C / sum_rho2;

    rss = 0.0;
    for (const auto& lag : lag_data) {
        double rho = correlation_value(lag.distance, model, range);
        if (rho <= 0.0) continue;
        Eigen::MatrixXd resid = lag.cov_matrix - rho * B;
        rss += resid.squaredNorm();
    }
    return true;
}

void project_psd(Eigen::MatrixXd& mat) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);
    Eigen::VectorXd ev = solver.eigenvalues();
    for (int i = 0; i < ev.size(); ++i) if (ev(i) < 0.0) ev(i) = 0.0;
    mat = solver.eigenvectors() * ev.asDiagonal() * solver.eigenvectors().transpose();
    mat = (mat + mat.transpose()) * 0.5;
}

} // anonymous namespace

LinearModelOfCoregionalization fit_lmc_from_experimental_variograms(
    const std::vector<VariogramResult>& direct_variograms,
    const std::vector<VariogramResult>& cross_variograms,
    const std::vector<int>& categories,
    int n_structures,
    const std::string& model_type,
    double tolerance,
    int max_iter)
{
    if (max_iter < 1 || !std::isfinite(tolerance) || tolerance < 0.0)
        throw std::invalid_argument("invalid LMC optimizer settings");
    if (categories.empty()) throw std::invalid_argument("LMC categories must not be empty");
    if (model_type != "spherical" && model_type != "exponential" && model_type != "gaussian")
        throw std::invalid_argument("unsupported LMC covariance model");
    if (n_structures < 1) throw std::runtime_error("At least one structure required");
    if (direct_variograms.empty()) throw std::runtime_error("No direct variograms provided");

    auto lag_data = build_lag_data(direct_variograms, cross_variograms, categories);
    if (lag_data.empty()) throw std::runtime_error("No valid lag data");

    int n_cat = static_cast<int>(categories.size());
    double max_lag = lag_data.back().distance;
    double range_global = max_lag * 0.8;

    // Heuristic range from the first direct variogram
    if (!direct_variograms.empty() && !direct_variograms[0].lags.empty()) {
        std::vector<double> lags, gamma;
        for (const auto& l : direct_variograms[0].lags) {
            if (l.n_pairs > 0) {
                lags.push_back(l.lag_center);
                gamma.push_back(l.gamma_value);
            }
        }
        if (lags.size() >= 3) {
            double max_gamma = gamma.back();
            for (size_t i = 0; i < gamma.size(); ++i) {
                if (gamma[i] >= 0.95 * max_gamma) {
                    range_global = lags[i];
                    break;
                }
            }
        }
    }

    std::vector<double> ranges(n_structures);
    for (int s = 0; s < n_structures; ++s) {
        double ratio = std::pow(0.3, s);
        ranges[s] = range_global * ratio;
        if (ranges[s] < max_lag * 0.05) ranges[s] = max_lag * 0.05;
        if (ranges[s] > max_lag * 1.2) ranges[s] = max_lag * 1.2;
    }

    // Joint projected block-coordinate least squares. Each nested structure is
    // updated against the residual of ALL other structures, rather than fitting
    // every structure to the full empirical covariance independently.
    std::vector<Eigen::MatrixXd> matrices(
        n_structures, Eigen::MatrixXd::Zero(n_cat, n_cat));
    std::vector<std::vector<double>> rho(n_structures,
        std::vector<double>(lag_data.size(), 0.0));
    for (int k = 0; k < n_structures; ++k)
        for (size_t l = 0; l < lag_data.size(); ++l)
            rho[k][l] = correlation_value(lag_data[l].distance, model_type, ranges[k]);
    double previous = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < max_iter; ++iteration) {
        for (int k = 0; k < n_structures; ++k) {
            Eigen::MatrixXd numerator = Eigen::MatrixXd::Zero(n_cat, n_cat);
            double denominator = 0.0;
            for (size_t l = 0; l < lag_data.size(); ++l) {
                Eigen::MatrixXd residual = lag_data[l].cov_matrix;
                for (int j = 0; j < n_structures; ++j)
                    if (j != k) residual.noalias() -= rho[j][l] * matrices[j];
                numerator.noalias() += rho[k][l] * residual;
                denominator += rho[k][l] * rho[k][l];
            }
            if (denominator > 1e-14)
                matrices[k] = project_coregionalization_psd(numerator / denominator);
        }
        double objective = 0.0;
        for (size_t l = 0; l < lag_data.size(); ++l) {
            Eigen::MatrixXd residual = lag_data[l].cov_matrix;
            for (int k = 0; k < n_structures; ++k)
                residual.noalias() -= rho[k][l] * matrices[k];
            objective += residual.squaredNorm();
        }
        if (!std::isfinite(objective)) throw std::runtime_error("LMC fitting diverged");
        if (std::isfinite(previous) && std::abs(previous - objective) <= tolerance * (1.0 + previous)) break;
        previous = objective;
    }
    LinearModelOfCoregionalization lmc;
    for (int k = 0; k < n_structures; ++k) {
        LMCStructure structure;
        structure.structure_id = k;
        structure.family = model_type;
        structure.range_major = ranges[k];
        structure.range_minor = ranges[k];
        structure.range_vertical = ranges[k];
        structure.covariance_matrix.assign(n_cat, std::vector<double>(n_cat));
        for (int i = 0; i < n_cat; ++i)
            for (int j = 0; j < n_cat; ++j)
                structure.covariance_matrix[i][j] = matrices[k](i, j);
        lmc.add_structure(structure);
    }
    return lmc;
}

} // namespace deeprbf
