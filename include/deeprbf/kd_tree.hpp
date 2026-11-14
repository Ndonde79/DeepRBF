#pragma once
#include "deeprbf/spatial_hash.hpp"
#include <numeric>

namespace deeprbf {
// Immutable balanced 3D tree. Radius/k boundaries include ties, with canonical
// (Euclidean distance, input index) ordering, independent of tree traversal.
class KDTree3 {
    struct Node {int point,axis,left=-1,right=-1;};
    std::vector<Node> nodes_;
    bool built_=false;
    int build_range(std::vector<int>& order,int begin,int end,int depth){
        if(begin==end)return -1;
        int axis=depth%3,middle=begin+(end-begin)/2;
        std::nth_element(order.begin()+begin,order.begin()+middle,order.begin()+end,[&](int a,int b){
            return points[a][axis]==points[b][axis]?a<b:points[a][axis]<points[b][axis];});
        int node=static_cast<int>(nodes_.size());nodes_.push_back({order[middle],axis});
        const int left=build_range(order,begin,middle,depth+1),right=build_range(order,middle+1,end,depth+1);
        nodes_[node].left=left;nodes_[node].right=right;return node;
    }
    using Pair=std::pair<double,int>;
    void visit(int node,const Point3& q,double radius,size_t k,std::priority_queue<Pair>& heap)const {
        if(node<0)return;
        const auto& n=nodes_[node];const auto& p=points[n.point];
        const double distance=std::hypot(q[0]-p[0],q[1]-p[1],q[2]-p[2]);
        if(distance<=radius){Pair candidate{distance,n.point};
            if(heap.size()<k)heap.push(candidate);
            else if(candidate<heap.top()){heap.pop();heap.push(candidate);}}
        const double delta=q[n.axis]-p[n.axis];
        visit(delta<=0?n.left:n.right,q,radius,k,heap);
        const double bound=heap.size()<k?radius:std::min(radius,heap.top().first);
        if(std::abs(delta)<=bound)visit(delta<=0?n.right:n.left,q,radius,k,heap);
    }
public:
    std::vector<Point3> points;
    void reserve(size_t n){points.reserve(n);nodes_.reserve(n);}
    void add(const Point3& point,double){
        if(built_)throw std::logic_error("Cannot mutate built KD tree");
        for(double v:point)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite KD coordinate");
        if(points.size()>=static_cast<size_t>(std::numeric_limits<int>::max()))throw std::length_error("Too many KD points");
        points.push_back(point);
    }
    void build(){
        if(built_)throw std::logic_error("KD tree already built");
        std::vector<int> order(points.size());std::iota(order.begin(),order.end(),0);
        build_range(order,0,static_cast<int>(order.size()),0);built_=true;
    }
    std::vector<Neighbor> query(const Point3& target,double radius,int max_neighbors)const {
        if(!built_||!std::isfinite(radius)||radius<0||max_neighbors<1)throw std::invalid_argument("Invalid KD search controls");
        for(double v:target)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite KD target");
        std::priority_queue<Pair> heap;
        if(!nodes_.empty())visit(0,target,radius,static_cast<size_t>(max_neighbors),heap);
        std::vector<Neighbor> result;result.reserve(heap.size());
        while(!heap.empty()){result.push_back({heap.top().second,heap.top().first});heap.pop();}
        std::reverse(result.begin(),result.end());return result;
    }
};
} // namespace deeprbf
