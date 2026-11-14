// deeprbf/nested_variogram.cpp
#include "deeprbf/nested_variogram.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace deeprbf {

double structure_correlation(double distance, const VariogramStructure& s) {
    if(!std::isfinite(distance)||distance<0)throw std::invalid_argument("Finite nonnegative variogram distance required");
    if (distance <= 0.0) return 1.0;
    const double r = distance / std::max(s.range, 1e-12);

    switch (s.type) {
        case VariogramModelType::Spherical:
            if (r >= 1.0) return 0.0;
            return 1.0 - (1.5 * r - 0.5 * r * r * r);
        case VariogramModelType::Exponential:
            // exp(-3h/a) gives an effective range of ~a
            return std::exp(-3.0 * r);
        case VariogramModelType::Gaussian:
            return std::exp(-3.0 * r * r);
        default:
            return 0.0;
    }
}

NestedVariogram::NestedVariogram(double nugget) { set_nugget(nugget); }

void NestedVariogram::add_structure(const VariogramStructure& s) {
    if (!std::isfinite(s.range) || s.range <= 0.0) {
        throw std::runtime_error("NestedVariogram: structure range must be > 0");
    }
    if (!std::isfinite(s.contribution) || s.contribution < 0.0) {
        throw std::runtime_error("NestedVariogram: structure contribution must be >= 0");
    }
    structures_.push_back(s);
}

void NestedVariogram::set_nugget(double nugget) { if(!std::isfinite(nugget)||nugget<0)throw std::invalid_argument("invalid nugget"); nugget_ = nugget; }

double NestedVariogram::nugget() const { return nugget_; }

double NestedVariogram::sill() const {
    double total = nugget_;
    for (const auto& s : structures_) total += s.contribution;
    return total;
}

std::size_t NestedVariogram::n_structures() const { return structures_.size(); }

double NestedVariogram::correlation(double distance) const {
    if (structures_.empty()) return 0.0;
    double total_contrib = 0.0;
    double weighted = 0.0;
    for (const auto& s : structures_) {
        weighted += s.contribution * structure_correlation(distance, s);
        total_contrib += s.contribution;
    }
    if (total_contrib <= 0.0) return 0.0;
    return weighted / total_contrib;
}

double NestedVariogram::covariance(double distance) const {
    double cov = 0.0;
    for (const auto& s : structures_) {
        cov += s.contribution * structure_correlation(distance, s);
    }
    if (distance == 0.0) cov += nugget_;
    return cov;
}

} // namespace deeprbf
