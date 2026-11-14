#ifndef DEEPRBF_SCALFMM_CONFIG_H
#define DEEPRBF_SCALFMM_CONFIG_H

// Compatibility include retained for downstream projects.
//
// DeepRBF no longer defines ScalFMM implementation macros in a public header.
// ScalFMM configuration is supplied exclusively by the imported CMake target
// scalfmm::scalfmm, which propagates the exact compile definitions and
// transitive dependencies of the installed ScalFMM build.

#endif  // DEEPRBF_SCALFMM_CONFIG_H