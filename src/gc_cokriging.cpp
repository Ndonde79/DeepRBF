#include "deeprbf/gc_geostatistics.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace deeprbf {

namespace {
constexpr double REGULARIZATION = 1e-8;

inline int get_category_index(int category, const std::vector<int>& categories) {
    if(categories.empty())throw std::invalid_argument("categories must not be empty");

    for (size_t i = 0; i < categories.size(); ++i)
        if (categories[i] == category) return static_cast<int>(i);
    return -1;
}
} // anonymous namespace

// =============================================================================
// LinearModelOfCoregionalization
// =============================================================================
void LinearModelOfCoregionalization::add_structure(const LMCStructure& structure) {
    lmc_correlation(0.,structure.family);
    structures_.push_back(structure);
}

double lmc_correlation(double h,const std::string& family) {
    if(!std::isfinite(h)||h<0)throw std::invalid_argument("Invalid LMC scaled distance");
    if(family=="spherical")return spherical_covariance(h,1.,0.);
    if(family=="exponential")return std::exp(-3*h);
    if(family=="gaussian")return std::exp(-3*h*h);
    throw std::invalid_argument("Unknown LMC covariance family");
}

double LinearModelOfCoregionalization::covariance(
    int cat_i, int cat_j, double h_scaled) const {
    if (cat_i < 0 || cat_j < 0 || !std::isfinite(h_scaled) || h_scaled < 0.0)
        throw std::invalid_argument("LMC covariance requires valid categories and nonnegative finite scaled distance");
    double total_cov = 0.0;
    for (const auto& structure : structures_) {
        const auto n = structure.covariance_matrix.size();
        if (static_cast<size_t>(cat_i) >= n || static_cast<size_t>(cat_j) >= n ||
            structure.covariance_matrix[cat_i].size() != n)
            throw std::invalid_argument("LMC covariance matrix/category mismatch");
        const double coefficient = structure.covariance_matrix[cat_i][cat_j];
        if (!std::isfinite(coefficient))
            throw std::invalid_argument("nonfinite LMC covariance coefficient");
        // h_scaled is already normalized to the structure range by the caller.
        total_cov += coefficient * lmc_correlation(h_scaled, structure.family);
    }
    return total_cov;
}

// =============================================================================
// build_lmc_from_variograms
// =============================================================================
LinearModelOfCoregionalization build_lmc_from_variograms(
    const std::vector<CrossVariogram>& variograms,
    int n_structures)
{
    // The archive invented fixed anisotropy/ranges instead of fitting them.
    throw std::runtime_error("Automatic LMC construction from cross-variograms is not implemented reliably in this archive; specify validated LMCStructure matrices and ranges explicitly");

}

// =============================================================================
// is_positive_definite
// =============================================================================
bool is_positive_definite(const LinearModelOfCoregionalization& lmc) {
    if(lmc.structures().empty())return false;
    for(const auto& st:lmc.structures()){
        const int n=static_cast<int>(st.covariance_matrix.size());if(n==0)return false;
        Eigen::MatrixXd B(n,n);
        for(int i=0;i<n;++i){if(st.covariance_matrix[i].size()!=static_cast<size_t>(n))return false;for(int j=0;j<n;++j)B(i,j)=st.covariance_matrix[i][j];}
        if(!B.allFinite()||!B.isApprox(B.transpose(),1e-10))return false;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(B);
        if(eig.info()!=Eigen::Success||eig.eigenvalues().minCoeff() < -1e-10*std::max(1.,B.norm()))return false;
    }
    return true; // Positive semidefinite is sufficient for a coregionalization matrix.

}

// =============================================================================
// local_cokriging_categorical
// =============================================================================
static bool local_cokriging_categorical(
    const Point3& target,
    const std::vector<Point3>& cond_points,
    const std::vector<int>& cond_categories,
    const std::vector<int>& categories,
    const LinearModelOfCoregionalization& lmc,
    int /*max_neighbors*/,
    std::vector<double>& probs,
    std::vector<double>& variance_estimate)
{
    const int nc=static_cast<int>(categories.size()), m=static_cast<int>(cond_points.size()), n=m*nc;
    if(!m||!nc)return false;
    auto cov=[&](const Point3& a,const Point3& b,int i,int j){
        double sum=0.;for(const auto& st:lmc.structures()){
            double h=anisotropic_scaled_distance(a,b,st.range_major,st.range_minor,st.range_vertical,st.azimuth_deg,st.dip_deg,st.plunge_deg);
            sum+=st.covariance_matrix[i][j]*lmc_correlation(h,st.family);
        }return sum;
    };
    Eigen::MatrixXd A=Eigen::MatrixXd::Zero(n+nc,n+nc);
    Eigen::VectorXd indicators(n);
    for(int i=0;i<m;++i)for(int c=0;c<nc;++c){
        int row=i*nc+c;indicators[row]=cond_categories[i]==categories[c]?1.:0.;
        A(row,n+c)=A(n+c,row)=1.;
        for(int j=0;j<m;++j)for(int k=0;k<nc;++k)A(row,j*nc+k)=cov(cond_points[i],cond_points[j],c,k);
        A(row,row)+=REGULARIZATION;
    }
    auto solver=A.fullPivLu();if(!solver.isInvertible())return false;
    probs.assign(nc,0.);variance_estimate.assign(nc,0.);
    for(int c=0;c<nc;++c){
        Eigen::VectorXd b=Eigen::VectorXd::Zero(n+nc);
        for(int i=0;i<m;++i)for(int k=0;k<nc;++k)b[i*nc+k]=cov(cond_points[i],target,k,c);
        b[n+c]=1.;Eigen::VectorXd w=solver.solve(b);if(!w.allFinite()||(A*w-b).norm()>1e-9*std::max(1.,b.norm()))return false;
        probs[c]=std::clamp(w.head(n).dot(indicators),0.,1.);
        variance_estimate[c]=std::max(0.,cov(target,target,c,c)-w.head(n).dot(b.head(n))-w[n+c]);
    }
    double total=std::accumulate(probs.begin(),probs.end(),0.);
    if(total>1e-12)for(auto& v:probs)v/=total;else std::fill(probs.begin(),probs.end(),1./nc);
    return true;

}

// =============================================================================
// ordinary_cokriging
// =============================================================================
CoKrigingResult ordinary_cokriging(
    const std::vector<Point3>& sample_points,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_points,
    const std::vector<int>& categories,
    const LinearModelOfCoregionalization& lmc,
    int max_neighbors,
    int min_neighbors,
    double search_radius)
{
    for(const auto& p:sample_points)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_points)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_points.size()!=sample_categories.size())throw std::invalid_argument("sample/value size mismatch");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(categories.empty())throw std::invalid_argument("categories must not be empty");
    if(!is_positive_definite(lmc))throw std::invalid_argument("LMC matrices must be symmetric positive semidefinite");
    if(std::set<int>(categories.begin(),categories.end()).size()!=categories.size())throw std::invalid_argument("duplicate categories");
    for(const auto& st:lmc.structures()){
        if(st.covariance_matrix.size()!=categories.size())throw std::invalid_argument("LMC/category dimension mismatch");
        for(double v:{st.range_major,st.range_minor,st.range_vertical})if(!std::isfinite(v)||v<=0)throw std::invalid_argument("invalid LMC range");
        for(double v:{st.azimuth_deg,st.dip_deg,st.plunge_deg})if(!std::isfinite(v))throw std::invalid_argument("invalid LMC angle");
    }
    for(int c:sample_categories)if(std::find(categories.begin(),categories.end(),c)==categories.end())throw std::invalid_argument("unknown sample category");

    std::map<Point3,int> known;
    for(size_t i=0;i<sample_points.size();++i){auto [it,inserted]=known.emplace(sample_points[i],sample_categories[i]);
        if(!inserted&&it->second!=sample_categories[i])throw std::invalid_argument("Conflicting coincident categories");}
    if(categories.size()>512||static_cast<long double>(std::min(max_neighbors,64))*categories.size()>2048||
       static_cast<long double>(target_points.size())*categories.size()>32000000)
        throw std::invalid_argument("Categorical cokriging workspace/output limit exceeded");


    CoKrigingResult result;
    const size_t n_target = target_points.size();
    const size_t n_cat = categories.size();

    result.estimates.assign(n_target * n_cat, 0.0);
    result.variances.assign(n_target * n_cat, std::numeric_limits<double>::quiet_NaN());
    result.n_samples.assign(n_target, 0);
    result.samples_per_category.assign(n_target, std::vector<int>(n_cat, 0));

    if (sample_points.empty() || target_points.empty() || categories.empty())
        return result;

    int max_n = std::clamp(max_neighbors, 1, 64);
    int min_n = std::max(min_neighbors, 1);

    std::vector<double> global_props(n_cat, 1.0 / static_cast<double>(n_cat));
    if (!sample_points.empty()) {
        for (size_t c = 0; c < n_cat; ++c) {
            int count = 0;
            for (int cat : sample_categories)
                if (cat == categories[c]) ++count;
            global_props[c] = static_cast<double>(count) /
                              static_cast<double>(sample_points.size());
        }
        double sum = std::accumulate(global_props.begin(), global_props.end(), 0.0);
        if (sum > 1e-12)
            for (double& p : global_props) p /= sum;
    }

    for (size_t t = 0; t < n_target; ++t) {
        auto exact=known.find(target_points[t]);
        if(exact!=known.end()){
            for(size_t c=0;c<n_cat;++c){result.estimates[t*n_cat+c]=categories[c]==exact->second?1.:0.;result.variances[t*n_cat+c]=0.;}
            result.n_samples[t]=1;result.samples_per_category[t][get_category_index(exact->second,categories)]=1;continue;
        }
        std::vector<std::pair<double, size_t>> distances;
        distances.reserve(sample_points.size());
        for (size_t s = 0; s < sample_points.size(); ++s) {
            double dx = sample_points[s][0] - target_points[t][0];
            double dy = sample_points[s][1] - target_points[t][1];
            double dz = sample_points[s][2] - target_points[t][2];
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist <= search_radius) distances.emplace_back(dist, s);
        }

        std::sort(distances.begin(), distances.end());
        int n_neigh = std::min(static_cast<int>(distances.size()), max_n);

        if (n_neigh < min_n) {
            for (size_t c = 0; c < n_cat; ++c)
                result.estimates[t * n_cat + c] = global_props[c];
            result.n_samples[t] = n_neigh;
            continue;
        }

        std::vector<Point3> cond_points;
        std::vector<int> cond_categories;
        cond_points.reserve(n_neigh);
        cond_categories.reserve(n_neigh);
        for (int i = 0; i < n_neigh; ++i) {
            size_t idx = distances[static_cast<size_t>(i)].second;
            cond_points.push_back(sample_points[idx]);
            cond_categories.push_back(sample_categories[idx]);

            for (size_t c = 0; c < n_cat; ++c) {
                if (sample_categories[idx] == categories[c]) {
                    result.samples_per_category[t][c]++;
                    break;
                }
            }
        }

        std::vector<double> probs;
        std::vector<double> variance_est;
        if (local_cokriging_categorical(target_points[t], cond_points, cond_categories,
                                        categories, lmc, max_n, probs, variance_est)) {
            for (size_t c = 0; c < n_cat; ++c) {
                result.estimates[t * n_cat + c] = probs[c];
                result.variances[t * n_cat + c] = variance_est[c];
            }
            result.n_samples[t] = n_neigh;
        } else {
            for (size_t c = 0; c < n_cat; ++c)
                result.estimates[t * n_cat + c] = global_props[c];
            result.n_samples[t] = 0; // A prior fallback is not a successful local solve.
        }
    }

    return result;
}

// =============================================================================
// sequential_indicator_cokriging
// =============================================================================
SICoKResult sequential_indicator_cokriging(
    const std::vector<Point3>& sample_points,
    const std::vector<int>& sample_categories,
    const std::vector<Point3>& target_points,
    const std::vector<int>& categories,
    const LinearModelOfCoregionalization& lmc,
    int n_realizations,
    int max_neighbors,
    int min_neighbors,
    double search_radius,
    int seed)
{
    for(const auto& p:sample_points)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    for(const auto& p:target_points)for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("non-finite coordinates");
    if(sample_points.size()!=sample_categories.size())throw std::invalid_argument("sample/value size mismatch");
    if(!std::isfinite(search_radius)||search_radius<=0)throw std::invalid_argument("search radius must be positive");
    if(max_neighbors<=0||min_neighbors<=0||min_neighbors>max_neighbors)throw std::invalid_argument("invalid neighbor counts");
    if(n_realizations<=0)throw std::invalid_argument("realization count must be positive");
    if(static_cast<long double>(target_points.size())*n_realizations>32000000.L||categories.size()>512||
       static_cast<long double>(std::min(max_neighbors,64))*categories.size()>2048)
        throw std::invalid_argument("Categorical simulation workspace/output limit exceeded");
    if(categories.empty())throw std::invalid_argument("categories must not be empty");
    if(!is_positive_definite(lmc))throw std::invalid_argument("LMC matrices must be symmetric positive semidefinite");
    if(std::set<int>(categories.begin(),categories.end()).size()!=categories.size())throw std::invalid_argument("duplicate categories");
    for(const auto& st:lmc.structures()){
        if(st.covariance_matrix.size()!=categories.size())throw std::invalid_argument("LMC/category dimension mismatch");
        for(double v:{st.range_major,st.range_minor,st.range_vertical})if(!std::isfinite(v)||v<=0)throw std::invalid_argument("invalid LMC range");
        for(double v:{st.azimuth_deg,st.dip_deg,st.plunge_deg})if(!std::isfinite(v))throw std::invalid_argument("invalid LMC angle");
    }
    for(int c:sample_categories)if(std::find(categories.begin(),categories.end(),c)==categories.end())throw std::invalid_argument("unknown sample category");


    SICoKResult result;
    result.n_targets = target_points.size();
    result.n_realizations = static_cast<size_t>(std::max(n_realizations, 0));
    result.realizations.assign(result.n_realizations,
                               std::vector<int>(result.n_targets, -1));

    if (sample_points.empty() || target_points.empty() || categories.empty() || n_realizations <= 0)
        return result;

    int max_n = std::clamp(max_neighbors, 1, 64);
    int min_n = std::max(min_neighbors, 1);

    const size_t n_cat = categories.size();
    std::vector<double> global_props(n_cat, 1.0 / static_cast<double>(n_cat));
    if (!sample_points.empty()) {
        for (size_t c = 0; c < n_cat; ++c) {
            int count = 0;
            for (int cat : sample_categories)
                if (cat == categories[c]) ++count;
            global_props[c] = static_cast<double>(count) /
                              static_cast<double>(sample_points.size());
        }
        double sum = std::accumulate(global_props.begin(), global_props.end(), 0.0);
        if (sum > 1e-12)
            for (double& p : global_props) p /= sum;
    }

    std::map<Point3,int> hard;
    for(size_t i=0;i<sample_points.size();++i){auto [it,inserted]=hard.emplace(sample_points[i],sample_categories[i]);
        if(!inserted&&it->second!=sample_categories[i])throw std::invalid_argument("Conflicting coincident categories");}
    for (int sim = 0; sim < n_realizations; ++sim) {
        std::mt19937 rng(static_cast<unsigned int>(seed) + static_cast<unsigned int>(sim) * 1298731U);
        auto known=hard;

        std::vector<int> path(result.n_targets);
        std::iota(path.begin(), path.end(), 0);
        std::shuffle(path.begin(), path.end(), rng);

        std::vector<Point3> current_samples = sample_points;
        std::vector<int> current_categories = sample_categories;
        std::vector<int> sim_values(result.n_targets, -1);

        for (int idx : path) {
            const size_t i = static_cast<size_t>(idx);
            const auto& target = target_points[i];
            auto exact=known.find(target);
            if(exact!=known.end()){sim_values[i]=exact->second;continue;}

            std::vector<std::pair<double, size_t>> distances;
            distances.reserve(current_samples.size());
            for (size_t s = 0; s < current_samples.size(); ++s) {
                double dx = current_samples[s][0] - target[0];
                double dy = current_samples[s][1] - target[1];
                double dz = current_samples[s][2] - target[2];
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dist <= search_radius) distances.emplace_back(dist, s);
            }

            std::sort(distances.begin(), distances.end());
            int n_neigh = std::min(static_cast<int>(distances.size()), max_n);

            std::vector<double> probs = global_props;

            if (n_neigh >= min_n) {
                std::vector<Point3> cond_points;
                std::vector<int> cond_categories;
                cond_points.reserve(n_neigh);
                cond_categories.reserve(n_neigh);
                for (int k = 0; k < n_neigh; ++k) {
                    size_t s_idx = distances[static_cast<size_t>(k)].second;
                    cond_points.push_back(current_samples[s_idx]);
                    cond_categories.push_back(current_categories[s_idx]);
                }
                std::vector<double> variance_est;
                if(!local_cokriging_categorical(target, cond_points, cond_categories,
                                            categories, lmc, max_n, probs, variance_est))probs=global_props;
            }

            for (double& p : probs) p = std::max(p, 0.0);
            double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
            if (sum > 1e-12) {
                for (double& p : probs) p /= sum;
            } else {
                std::fill(probs.begin(), probs.end(),
                          1.0 / static_cast<double>(n_cat));
            }

            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int selected_cat = categories[static_cast<size_t>(dist(rng))];
            sim_values[i] = selected_cat;
            known.emplace(target,selected_cat);

            current_samples.push_back(target);
            current_categories.push_back(selected_cat);
        }

        result.realizations[static_cast<size_t>(sim)] = sim_values;
    }

    return result;
}

} // namespace deeprbf
