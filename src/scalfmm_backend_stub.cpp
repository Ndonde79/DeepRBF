#include "deeprbf/detail/scalfmm_backend.hpp"

namespace deeprbf::detail {

std::unique_ptr<IScalFMMBackend> make_scalfmm_backend() { return nullptr; }

bool scalfmm_backend_compiled() noexcept { return false; }

std::string scalfmm_backend_build_status() {
    return "ScalFMM 3 backend was not compiled; exact OpenMP direct evaluation is active";
}

}  // namespace deeprbf::detail