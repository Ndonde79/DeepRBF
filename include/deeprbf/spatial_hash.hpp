#pragma once
#include "deeprbf/core.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
namespace deeprbf {
struct Neighbor { int index; double distance; };
class SpatialHash {
    using Cell=std::array<std::int64_t,3>;
    struct CellHash { size_t operator()(const Cell& c) const noexcept {
        size_t h=0; for(auto v:c) h^=std::hash<std::int64_t>{}(v)+0x9e3779b9+(h<<6)+(h>>2); return h;
    }};
    double cell_size;
    std::unordered_map<Cell,std::vector<int>,CellHash> buckets;
    Cell cell(const Point3& p) const {
        Cell out{};
        for(size_t k=0;k<3;++k) {
            double v=std::floor(p[k]/cell_size);
            if(!std::isfinite(v)||std::abs(v)>9e15) throw std::invalid_argument("coordinate outside spatial hash range");
            out[k]=static_cast<std::int64_t>(v);
        }
        return out;
    }
public:
    void validate_target(const Point3& p) const { (void)cell(p); }
    std::vector<Point3> points;
    std::vector<double> values;
    explicit SpatialHash(double size):cell_size(size) {
        if(!std::isfinite(size)||size<=0) throw std::invalid_argument("cell size must be finite and positive");
    }
    void clear(){points.clear();values.clear();buckets.clear();}
    void reserve(size_t n){points.reserve(n);values.reserve(n);buckets.reserve(n);}
    void add(const Point3& p,double v) {
        auto key=cell(p);
        if(!std::isfinite(v)) throw std::invalid_argument("non-finite sample value");
        if(points.size()>=static_cast<size_t>(std::numeric_limits<int>::max())) throw std::length_error("too many samples");
        int i=static_cast<int>(points.size()); points.push_back(p); values.push_back(v); buckets[key].push_back(i);
    }
    void build(const std::vector<Point3>& pts,const std::vector<double>& vals) {
        if(pts.size()!=vals.size()) throw std::invalid_argument("sample/value size mismatch");
        for(size_t i=0;i<pts.size();++i){cell(pts[i]); if(!std::isfinite(vals[i])) throw std::invalid_argument("non-finite values");}
        clear();reserve(pts.size());for(size_t i=0;i<pts.size();++i)add(pts[i],vals[i]);
    }
    std::vector<Neighbor> query(const Point3& target,double radius,int max_neighbors) const {
        auto c=cell(target);
        if(!std::isfinite(radius)||radius<0||max_neighbors<=0) throw std::invalid_argument("invalid neighbor search parameters");
        using Pair=std::pair<double,int>; std::priority_queue<Pair> heap;
        auto consider=[&](int i) {
            const auto& p=points[i]; double d=std::hypot(target[0]-p[0],target[1]-p[1],target[2]-p[2]);
            if(d<=radius) { Pair value{d,i}; if(heap.size()<static_cast<size_t>(max_neighbors))heap.push(value);
                else if(value<heap.top()){heap.pop();heap.push(value);} }
        };
        double nr=std::ceil(radius/cell_size);
        // For large/sparse searches a full scan is cheaper and avoids huge cell loops.
        if(nr>64 || std::pow(2*nr+1,3)>2*static_cast<double>(buckets.size())) {
            for(size_t i=0;i<points.size();++i)consider(static_cast<int>(i));
        } else {
            int ring=static_cast<int>(nr);
            for(int dx=-ring;dx<=ring;++dx)for(int dy=-ring;dy<=ring;++dy)for(int dz=-ring;dz<=ring;++dz){
                auto it=buckets.find(Cell{c[0]+dx,c[1]+dy,c[2]+dz}); if(it!=buckets.end()) for(int i:it->second)consider(i);
            }
        }
        std::vector<Neighbor> out;while(!heap.empty()){out.push_back({heap.top().second,heap.top().first});heap.pop();}
        std::reverse(out.begin(),out.end());return out;
    }
};
}
