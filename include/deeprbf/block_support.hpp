#pragma once
#include "deeprbf/gc_block_kriging.hpp"
#include <cmath>
#include <stdexcept>

namespace deeprbf {
// Equal-volume midpoint quadrature shared by estimation and support moments.
inline std::vector<Point3> block_midpoints(const Block& b) {
    for (double v : {b.xmin,b.xmax,b.ymin,b.ymax,b.zmin,b.zmax})
        if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite block extent");
    const Point3 low{b.xmin,b.ymin,b.zmin}, high{b.xmax,b.ymax,b.zmax};
    size_t count=1;
    for (int j=0;j<3;++j) {
        if (high[j]<low[j] || !std::isfinite(high[j]-low[j]))
            throw std::invalid_argument("Invalid block extent");
        if (b.discretization[j]<1 || b.discretization[j]>32)
            throw std::invalid_argument("Block discretization outside 1..32");
        count*=b.discretization[j];
    }
    if (count>4096) throw std::invalid_argument("Block quadrature exceeds 4096 nodes");
    std::vector<Point3> result; result.reserve(count);
    for (int i=0;i<b.discretization[0];++i)
        for (int j=0;j<b.discretization[1];++j)
            for (int k=0;k<b.discretization[2];++k) {
                const int index[]{i,j,k}; Point3 p{};
                for (int axis=0;axis<3;++axis)
                    p[axis]=low[axis]+((index[axis]+.5)/b.discretization[axis])*(high[axis]-low[axis]);
                result.push_back(p);
            }
    return result;
}
}
