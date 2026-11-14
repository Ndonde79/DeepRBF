// deeprbf/nested_variogram.hpp
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace deeprbf {

enum class VariogramModelType {
    Spherical,
    Exponential,
    Gaussian
};

struct VariogramStructure {
    VariogramModelType type = VariogramModelType::Spherical;
    double range = 100.0;          // practical range
    double contribution = 1.0;     // partial sill
};

class NestedVariogram {
public:
    NestedVariogram() = default;
    explicit NestedVariogram(double nugget);

    void add_structure(const VariogramStructure& s);
    void set_nugget(double nugget);

    double nugget() const;
    double sill() const;            // nugget + sum of contributions
    std::size_t n_structures() const;

    // Normalised correlation (weighted mean of structure correlations)
    double correlation(double distance) const;

    // Covariance: nugget*delta(0) + sum contribution * correlation(h)
    double covariance(double distance) const;

    const std::vector<VariogramStructure>& structures() const { return structures_; }

private:
    double nugget_ = 0.0;
    std::vector<VariogramStructure> structures_;
};

// Helper used by both nested and LMC code paths
double structure_correlation(double distance, const VariogramStructure& s);

} // namespace deeprbf