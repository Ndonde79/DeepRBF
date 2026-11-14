#include "deeprbf/gc_change_of_support.hpp"
#include "deeprbf/block_support.hpp"
#include "deeprbf/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace deeprbf {
namespace {
void moments(double mean,double variance,double vrf,bool lognormal=false) {
    if (!std::isfinite(mean)||!std::isfinite(variance)||variance<0||
        !std::isfinite(vrf)||vrf<0||vrf>1||(lognormal&&mean<=0))
        throw std::invalid_argument("Finite mean, nonnegative variance and VRF in [0,1] required; lognormal mean must be positive");
}
}

double compute_variance_reduction_factor(const Block& block,double range,double sill,double nugget) {
    if (!std::isfinite(range)||range<=0||!std::isfinite(sill)||sill<=0||
        !std::isfinite(nugget)||nugget<0||nugget>sill)
        throw std::invalid_argument("Invalid change-of-support covariance parameters");
    const auto points=block_midpoints(block);
    long double sum=0;
    for (const auto& a:points) for (const auto& b:points)
        sum+=spherical_covariance(euclidean_dist(a,b)/range,1.,nugget/sill);
    return static_cast<double>(sum/(static_cast<long double>(points.size())*points.size()));
}

ChangeOfSupportResult affine_correction(double mean,double variance,double vrf) {
    moments(mean,variance,vrf);
    return {mean,variance*vrf,vrf};
}

ChangeOfSupportResult lognormal_correction(double mean,double variance,double vrf) {
    // Direct lognormal moment matching: preserve arithmetic mean and reduce
    // arithmetic variance, not log variance. This is not a discrete Gaussian model.
    moments(mean,variance,vrf,true);
    return {mean,variance*vrf,vrf};
}

std::vector<double> apply_change_of_support_to_realizations(
    const std::vector<std::vector<double>>& draws,double mean,double variance,
    const Block& block,double range,double sill,double nugget,const std::string& method) {
    if (method!="affine"&&method!="lognormal"&&method!="average")
        throw std::invalid_argument("Unknown support method; use affine, lognormal or average");
    const double vrf=compute_variance_reduction_factor(block,range,sill,nugget);
    moments(mean,variance,vrf,method=="lognormal");
    const size_t required=method=="average"?block_midpoints(block).size():1;
    for (const auto& row:draws) {
        if (row.size()!=required)
            throw std::invalid_argument("Affine/lognormal need one point draw per row; use average with one value per block quadrature node for joint spatial realizations");
        for (double v:row) if (!std::isfinite(v)||(method=="lognormal"&&v<=0))
            throw std::invalid_argument("Finite realization values required; lognormal values must be positive");
    }
    std::vector<double> result; result.reserve(draws.size());
    const long double cv2=method=="lognormal"?static_cast<long double>(variance)/mean/mean:0;
    const long double sp=std::log1p(cv2), sb=std::log1p(cv2*vrf);
    for (const auto& row:draws) {
        long double value=mean;
        if (method=="average") { value=0; for (double v:row)value+=static_cast<long double>(v)/row.size(); }
        else if (variance>0 && method=="affine") value=mean+std::sqrt(vrf)*(static_cast<long double>(row[0])-mean);
        else if (variance>0 && vrf>0) {
            const long double logmean=std::log(static_cast<long double>(mean));
            value=std::exp(logmean-sb/2+std::sqrt(sb/sp)*(std::log(static_cast<long double>(row[0]))-logmean+sp/2));
        }
        const double output=static_cast<double>(value);
        if (!std::isfinite(output)) throw std::overflow_error("Support transform exceeds finite output range");
        result.push_back(output);
    }
    return result;
}
}
