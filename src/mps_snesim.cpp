// deeprbf/mps_snesim.cpp
#include "deeprbf/mps_snesim.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <map>
#include <limits>
#include <set>

namespace deeprbf {

namespace {

std::int64_t product(const std::vector<std::int64_t>& s) {
    std::int64_t n = 1;
    for (auto v : s) {
        if(v<=0||v>32000000||n>32000000/v)throw std::invalid_argument("SNESIM shape exceeds 32 million cells");
        n *= v;
    }
    return n;
}

std::int64_t flat3(std::int64_t z, std::int64_t y, std::int64_t x,
                   std::int64_t /*nz*/, std::int64_t ny, std::int64_t nx) {
    return (z * ny + y) * nx + x;
}

struct Offset3 { int z, y, x; double d; };

std::vector<Offset3> make_offsets(const SnesimNativeConfig& cfg,
                                  int stride, bool is2d) {
    std::vector<Offset3> out;
    const int rx = static_cast<int>(std::ceil(cfg.radius_x));
    const int ry = static_cast<int>(std::ceil(cfg.radius_y));
    const int rz = is2d ? 0
                        : static_cast<int>(std::ceil(cfg.radius_z));

    for (int z = -rz; z <= rz; ++z) {
        for (int y = -ry; y <= ry; ++y) {
            for (int x = -rx; x <= rx; ++x) {
                if (x == 0 && y == 0 && z == 0) continue;
                const double q =
                    (rx? (x / cfg.radius_x) * (x / cfg.radius_x):0.) +
                    (ry? (y / cfg.radius_y) * (y / cfg.radius_y):0.) +
                    (is2d || !rz ? 0.0
                          : (z / cfg.radius_z) * (z / cfg.radius_z));
                if (q <= 1.0 + 1e-12) {
                    out.push_back({z * stride, y * stride, x * stride,
                                   std::sqrt(double(x * x + y * y + z * z))});
                }
            }
        }
    }

    std::sort(out.begin(), out.end(),
              [](const Offset3& a, const Offset3& b) {
                  if (a.d != b.d) return a.d < b.d;
                  if (a.z != b.z) return a.z < b.z;
                  if (a.y != b.y) return a.y < b.y;
                  return a.x < b.x;
              });

    if (static_cast<int>(out.size()) > cfg.max_conditioning_nodes)
        out.resize(cfg.max_conditioning_nodes);
    return out;
}

struct PatternSet {
    std::vector<std::vector<std::int32_t>> patterns;
    std::vector<std::int32_t>              centers;
    std::vector<double>                    prior;
    std::vector<std::map<std::int32_t,std::vector<int>>> postings;
};

PatternSet scan_patterns(const std::vector<std::int32_t>& ti,
                         const std::vector<std::int64_t>& shape,
                         const std::vector<std::int32_t>& cats,
                         const std::vector<Offset3>&      offs) {
    const bool is2d = shape.size() == 2;
    const std::int64_t nz = is2d ? 1 : shape[0];
    const std::int64_t ny = is2d ? shape[0] : shape[1];
    const std::int64_t nx = is2d ? shape[1] : shape[2];

    PatternSet ps;
    if(static_cast<long double>(ti.size())*std::max<size_t>(1,offs.size())>32000000.L)
        throw std::invalid_argument("SNESIM pattern workspace exceeds 32 million entries");
    ps.postings.resize(offs.size());
    std::vector<double> prior_count(cats.size(), 0.0);

    for (std::int64_t z = 0; z < nz; ++z) {
        for (std::int64_t y = 0; y < ny; ++y) {
            for (std::int64_t x = 0; x < nx; ++x) {
                const auto ci = flat3(z, y, x, nz, ny, nx);
                const auto cv = ti[ci];
                if (cv == SNESIM_UNKNOWN) continue;

                std::vector<std::int32_t> pat;
                pat.reserve(offs.size());
                bool ok = true;
                for (const auto& o : offs) {
                    const auto zz = z + o.z;
                    const auto yy = y + o.y;
                    const auto xx = x + o.x;
                    if (zz < 0 || zz >= nz || yy < 0 || yy >= ny ||
                        xx < 0 || xx >= nx) { ok = false; break; }
                    auto v = ti[flat3(zz, yy, xx, nz, ny, nx)];
                    if (v == SNESIM_UNKNOWN) { ok = false; break; }
                    pat.push_back(v);
                }
                if (!ok) continue;

                const int pattern_index=static_cast<int>(ps.patterns.size());
                for(size_t j=0;j<pat.size();++j)ps.postings[j][pat[j]].push_back(pattern_index);
                ps.patterns.push_back(std::move(pat));
                ps.centers.push_back(cv);
                auto it = std::find(cats.begin(), cats.end(), cv);
                if (it != cats.end())
                    prior_count[std::distance(cats.begin(), it)] += 1.0;
            }
        }
    }

    if (ps.patterns.empty())
        throw std::runtime_error(
            "SNESIM native: zero valid TI patterns.");

    const double s = std::accumulate(prior_count.begin(),
                                     prior_count.end(), 0.0);
    ps.prior.assign(cats.size(), 1.0 / cats.size());
    if (s > 0)
        for (std::size_t i = 0; i < cats.size(); ++i)
            ps.prior[i] = prior_count[i] / s;
    return ps;
}

std::vector<double> normalize(std::vector<double> p) {
    for (auto& v : p) if (!(v > 0.0)) v = 0.0;
    double s = std::accumulate(p.begin(), p.end(), 0.0);
    if (s <= 0.0) {
        std::fill(p.begin(), p.end(), 1.0 / p.size());
        return p;
    }
    for (auto& v : p) v /= s;
    return p;
}

std::vector<double> query(const PatternSet& ps,
                          const std::vector<std::int32_t>& cats,
                          const std::vector<int>& pos,
                          const std::vector<std::int32_t>& vals,
                          int minrep,
                          int& dropped,
                          int& nrep) {
    dropped = 0;
    nrep    = 0;

    // Longest valid prefix is found by incrementally filtering a first-node
    // posting list. It gives the same nearest-first backoff counts as rescanning
    // the full TI for every attempted prefix.
    if(!pos.empty()){
        auto found=ps.postings[pos[0]].find(vals[0]);
        if(found!=ps.postings[pos[0]].end()&&found->second.size()>=static_cast<size_t>(minrep)){
            std::vector<int> current=found->second;size_t kept=1;
            for(;kept<pos.size();++kept){
                std::vector<int> next;next.reserve(current.size());
                for(int index:current)if(ps.patterns[index][pos[kept]]==vals[kept])next.push_back(index);
                if(next.size()<static_cast<size_t>(minrep))break;
                current=std::move(next);
            }
            std::vector<double> counts(cats.size(),0.);
            for(int index:current){auto it=std::find(cats.begin(),cats.end(),ps.centers[index]);++counts[std::distance(cats.begin(),it)];}
            dropped=static_cast<int>(pos.size()-kept);nrep=static_cast<int>(current.size());return normalize(std::move(counts));
        }
    }
    dropped=static_cast<int>(pos.size());nrep=static_cast<int>(ps.patterns.size());return ps.prior;
}

} // namespace

SnesimNativeResult snesim_native_static(
    const std::vector<std::int32_t>& training_image,
    const std::vector<std::int64_t>& training_shape,
    const std::vector<std::int32_t>& hard_data,
    const std::vector<std::int64_t>& simulation_shape,
    const std::vector<std::int32_t>& categories,
    const std::vector<double>&       target_proportions,
    const SnesimNativeConfig&        cfg,
    const std::vector<int>& simulation_mask,const std::vector<double>& soft,double soft_tau)
{
    if (training_shape.size() != 2 && training_shape.size() != 3)
        throw std::invalid_argument("SNESIM native supports 2D/3D.");
    if (simulation_shape.size() != training_shape.size())
        throw std::invalid_argument("SNESIM native dimension mismatch.");
    if (product(training_shape) !=
        static_cast<std::int64_t>(training_image.size()))
        throw std::invalid_argument("training_image size mismatch.");
    if (product(simulation_shape) !=
        static_cast<std::int64_t>(hard_data.size()))
        throw std::invalid_argument("hard_data size mismatch.");
    if(!simulation_mask.empty()&&simulation_mask.size()!=hard_data.size())throw std::invalid_argument("SNESIM mask size mismatch");
    if(cfg.max_conditioning_nodes<1||cfg.max_conditioning_nodes>128||cfg.min_replicates<1||cfg.n_multigrids<1||cfg.n_multigrids>12||
       !std::isfinite(cfg.servo_lambda)||cfg.servo_lambda<0||cfg.servo_lambda>=1)
        throw std::invalid_argument("Invalid SNESIM configuration");
    for(double r:{cfg.radius_x,cfg.radius_y,cfg.radius_z})if(!std::isfinite(r)||r<0||r>32)throw std::invalid_argument("SNESIM radii must be in [0,32]");
    if(cfg.radius_x+cfg.radius_y+(training_shape.size()==3?cfg.radius_z:0)<=0)throw std::invalid_argument("SNESIM needs a positive radius on an active dimension");
    const std::set<std::int32_t> catalogue(categories.begin(),categories.end());
    if(catalogue.size()!=categories.size()||catalogue.count(SNESIM_UNKNOWN)||categories.size()>512)throw std::invalid_argument("Invalid SNESIM category catalogue");
    for(auto v:training_image)if(v!=SNESIM_UNKNOWN&&!catalogue.count(v))throw std::invalid_argument("Unknown TI category");
    for(auto v:hard_data)if(v!=SNESIM_UNKNOWN&&!catalogue.count(v))throw std::invalid_argument("Unknown hard category");
    if(!target_proportions.empty()){
        if(target_proportions.size()!=categories.size())throw std::invalid_argument("SNESIM target proportion size mismatch");
        double sum=0;for(double p:target_proportions){if(!std::isfinite(p)||p<0)throw std::invalid_argument("Invalid SNESIM target proportion");sum+=p;}
        if(std::abs(sum-1)>1e-10)throw std::invalid_argument("SNESIM target proportions must sum to one");
    }
    if (categories.empty())
        throw std::invalid_argument("categories empty.");
    if(!std::isfinite(soft_tau)||soft_tau<0)throw std::invalid_argument("Invalid SNESIM soft weight");
    if(!soft.empty()){
        if(soft.size()!=hard_data.size()*categories.size()||soft.size()>32000000)throw std::invalid_argument("SNESIM soft shape/workspace mismatch");
        for(size_t node=0;node<hard_data.size();++node){double sum=0;
            for(size_t c=0;c<categories.size();++c){double v=soft[node*categories.size()+c];if(!std::isfinite(v)||v<0)throw std::invalid_argument("Invalid soft probability");sum+=v;}
            if((simulation_mask.empty()||simulation_mask[node])&&std::abs(sum-1)>1e-8)throw std::invalid_argument("Active soft probabilities must sum to one");
        }
    }

    const bool is2d = simulation_shape.size() == 2;
    const std::int64_t nz = is2d ? 1                : simulation_shape[0];
    const std::int64_t ny = is2d ? simulation_shape[0] : simulation_shape[1];
    const std::int64_t nx = is2d ? simulation_shape[1] : simulation_shape[2];

    SnesimNativeResult result;
    result.shape  = simulation_shape;
    result.values = hard_data;
    if(!simulation_mask.empty())for(size_t i=0;i<simulation_mask.size();++i)if(!simulation_mask[i])result.values[i]=SNESIM_UNKNOWN;

    std::mt19937_64 rng(cfg.seed);

    std::vector<double> running(categories.size(), 0.0);
    double informed = 0.0;
    for (auto v : result.values) {
        if (v == SNESIM_UNKNOWN) continue;
        auto it = std::find(categories.begin(), categories.end(), v);
        if (it != categories.end()) {
            running[std::distance(categories.begin(), it)] += 1.0;
            informed += 1.0;
        }
    }

    std::vector<double> target = target_proportions;
    if (target.size() != categories.size())
        target.assign(categories.size(), 0.0);

    for (int lev = std::max(1, cfg.n_multigrids) - 1; lev >= 0; --lev) {
        const int stride = 1 << lev;
        auto offs = make_offsets(cfg, stride, is2d);
        auto ps   = scan_patterns(training_image, training_shape,
                                  categories, offs);

        if (std::accumulate(target.begin(), target.end(), 0.0) <= 0.0)
            target = ps.prior;
        target = normalize(target);

        std::vector<std::int64_t> path;
        for (std::int64_t z = 0; z < nz; ++z) {
            for (std::int64_t y = 0; y < ny; ++y) {
                for (std::int64_t x = 0; x < nx; ++x) {
                    if ((z % stride) || (y % stride) || (x % stride)) continue;
                    auto fi = flat3(z, y, x, nz, ny, nx);
                    if(!simulation_mask.empty()&&!simulation_mask[fi])continue;
                    if (result.values[fi] == SNESIM_UNKNOWN)
                        path.push_back(fi);
                }
            }
        }
        std::shuffle(path.begin(), path.end(), rng);

        for (auto fi : path) {
            std::int64_t tmp = fi;
            const std::int64_t x = tmp % nx; tmp /= nx;
            const std::int64_t y = tmp % ny; tmp /= ny;
            const std::int64_t z = tmp;

            std::vector<int>                 pos;
            std::vector<std::int32_t>        vals;
            for (std::size_t j = 0; j < offs.size(); ++j) {
                auto zz = z + offs[j].z;
                auto yy = y + offs[j].y;
                auto xx = x + offs[j].x;
                if (zz < 0 || zz >= nz || yy < 0 || yy >= ny ||
                    xx < 0 || xx >= nx) continue;
                auto v = result.values[flat3(zz, yy, xx, nz, ny, nx)];
                if (v != SNESIM_UNKNOWN) {
                    pos.push_back(static_cast<int>(j));
                    vals.push_back(v);
                }
            }

            int dropped = 0, nrep = 0;
            auto p = query(ps, categories, pos, vals,
                           std::max(1, cfg.min_replicates), dropped, nrep);
            ++result.diagnostics.conditioning_queries;
            if (dropped > 0) ++result.diagnostics.backoff_events;
            if(dropped==static_cast<int>(pos.size()))++result.diagnostics.fallback_to_prior;
            result.diagnostics.replicate_sum+=nrep;

            // Target-prior Bayesian correction
            for (std::size_t i = 0; i < p.size(); ++i)
                p[i] *= target[i] / std::max(ps.prior[i], 1e-12);
            p = normalize(p);

            // Servosystem
            if (cfg.servo_lambda > 0.0 && cfg.servo_lambda < 1.0 &&
                informed > 0.0) {
                const double mu = cfg.servo_lambda /
                                  (1.0 - cfg.servo_lambda);
                for (std::size_t i = 0; i < p.size(); ++i) {
                    const double current = running[i] / informed;
                    p[i] = std::max(0.0,
                                    p[i] + mu * (target[i] - current));
                }
                p = normalize(p);
            }

            if(!soft.empty()){
                std::vector<double> logp(p.size());double largest=-std::numeric_limits<double>::infinity();
                for(size_t c=0;c<p.size();++c){logp[c]=p[c]>0?std::log(p[c])+soft_tau*(std::log(std::max(soft[fi*p.size()+c],1e-12))-std::log(std::max(ps.prior[c],1e-12))):-std::numeric_limits<double>::infinity();largest=std::max(largest,logp[c]);}
                for(size_t c=0;c<p.size();++c)p[c]=std::exp(logp[c]-largest);
                p=normalize(p);
            }
            std::discrete_distribution<int> dd(p.begin(), p.end());
            int ci = dd(rng);
            result.values[fi] = categories[ci];
            running[ci] += 1.0;
            informed    += 1.0;
            ++result.diagnostics.simulated_nodes;
        }
    }

    return result;
}

SnesimRuntimeStatus snesim_runtime_status() {
    SnesimRuntimeStatus s;
    s.stage                  = "11";
    s.module                 = "SNESIM mineral-deposit MPS";
    s.native_static_template = true;
    s.python_advanced_path   = true;
    return s;
}

} // namespace deeprbf
