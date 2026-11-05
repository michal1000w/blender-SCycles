/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

/* Kernel Feature Flags
 *
 * These are set from the host side depending on which features are used by the scene,
 * either for runtime checks or specialization.
 *
 * NOTE: Keep kernel features as defines as they are used by the preprocessor to do
 * compile time optimization while using adaptive kernel compilation. */

/* Shader nodes. */
#define KERNEL_FEATURE_NODE_BSDF (1ULL << 0ULL)
#define KERNEL_FEATURE_NODE_EMISSION (1ULL << 1ULL)
#define KERNEL_FEATURE_NODE_VOLUME (1ULL << 2ULL)
#define KERNEL_FEATURE_NODE_BUMP (1ULL << 3ULL)
#define KERNEL_FEATURE_NODE_BUMP_STATE (1ULL << 4ULL)
#define KERNEL_FEATURE_NODE_VORONOI_EXTRA (1ULL << 5ULL)
#define KERNEL_FEATURE_NODE_RAYTRACE (1ULL << 6ULL)
#define KERNEL_FEATURE_NODE_AOV (1ULL << 7ULL)
#define KERNEL_FEATURE_NODE_LIGHT_PATH (1ULL << 8ULL)
#define KERNEL_FEATURE_NODE_PRINCIPLED_HAIR (1ULL << 9ULL)
#define KERNEL_FEATURE_NODE_PORTAL (1ULL << 10ULL)

/* Use path tracing kernels. */
#define KERNEL_FEATURE_PATH_TRACING (1ULL << 11ULL)

/* BVH/sampling kernel features. */
#define KERNEL_FEATURE_POINTCLOUD (1ULL << 12ULL)
#define KERNEL_FEATURE_HAIR_RIBBON (1ULL << 13ULL)
#define KERNEL_FEATURE_HAIR_THICK (1ULL << 14ULL)
#define KERNEL_FEATURE_HAIR (KERNEL_FEATURE_HAIR_RIBBON | KERNEL_FEATURE_HAIR_THICK)
#define KERNEL_FEATURE_OBJECT_MOTION (1ULL << 15ULL)

/* Denotes whether baking functionality is needed. */
#define KERNEL_FEATURE_BAKING (1ULL << 16ULL)

/* Use subsurface scattering materials. */
#define KERNEL_FEATURE_SUBSURFACE (1ULL << 17ULL)

/* Use volume materials. */
#define KERNEL_FEATURE_VOLUME (1ULL << 18ULL)

/* Use Transparent shadows */
#define KERNEL_FEATURE_TRANSPARENT (1ULL << 19ULL)

/* Use shadow catcher. */
#define KERNEL_FEATURE_SHADOW_CATCHER (1ULL << 20ULL)

/* Light render passes. */
#define KERNEL_FEATURE_LIGHT_PASSES (1ULL << 21ULL)

/* AO. */
#define KERNEL_FEATURE_AO_PASS (1ULL << 22ULL)
#define KERNEL_FEATURE_AO_ADDITIVE (1ULL << 23ULL)
#define KERNEL_FEATURE_AO (KERNEL_FEATURE_AO_PASS | KERNEL_FEATURE_AO_ADDITIVE)

/* MNEE. */
#define KERNEL_FEATURE_MNEE (1ULL << 24ULL)

/* Path guiding. */
#define KERNEL_FEATURE_PATH_GUIDING (1ULL << 25ULL)

/* OSL. */
#define KERNEL_FEATURE_OSL_SHADING (1ULL << 26ULL)
#define KERNEL_FEATURE_OSL_CAMERA (1ULL << 27ULL)

/* Light and shadow linking. */
#define KERNEL_FEATURE_LIGHT_LINKING (1ULL << 28ULL)
#define KERNEL_FEATURE_SHADOW_LINKING (1ULL << 29ULL)

/* Use denoising kernels and output denoising passes. */
#define KERNEL_FEATURE_DENOISING (1ULL << 30ULL)

/* Light tree. */
#define KERNEL_FEATURE_LIGHT_TREE (1ULL << 31ULL)

/* Gaussian splats. */
#define KERNEL_FEATURE_GSPLATS (1ULL << 32ULL)

/* Metal bidirectional path tracing state. */
#define KERNEL_FEATURE_BDPT (1ULL << 33ULL)
#define KERNEL_FEATURE_PHOTON_MAPPING (1ULL << 34ULL)

/* Joint direct-source field evaluation; absent for ordinary radiometric scenes. */
#define KERNEL_FEATURE_COHERENT_DIRECT (1ULL << 35ULL)
#define KERNEL_FEATURE_COHERENT_SPECULAR (1ULL << 36ULL)
#define KERNEL_FEATURE_POLARIZATION (1ULL << 37ULL)

/* Nested dielectrics: materials with a nested priority, which track the media a path is inside
 * of to resolve overlapping refractive objects. */
#define KERNEL_FEATURE_NESTED_DIELECTRICS (1ULL << 38ULL)

/* Volumes with a spatially varying index of refraction, which bend the rays that cross them. */
#define KERNEL_FEATURE_VOLUME_IOR (1ULL << 39ULL)

/* Pixel-level displacement. Only requested for CUDA and OptiX devices, which keep it in a
 * separate kernel module; Metal and CPU kernels test the scene settings at runtime. */
#define KERNEL_FEATURE_PIXEL_DISPLACEMENT (1ULL << 40ULL)
/* Diffraction closures. Only set for devices with a transport kernel module, which contains them,
 * see __DIFFRACTION__ below. */
#define KERNEL_FEATURE_DIFFRACTION (1ULL << 41ULL)

/* Shader node feature mask, to specialize shader evaluation for kernels. */

#define KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT \
  (KERNEL_FEATURE_NODE_EMISSION | KERNEL_FEATURE_NODE_VORONOI_EXTRA | \
   KERNEL_FEATURE_NODE_LIGHT_PATH | KERNEL_FEATURE_NODE_PORTAL)
#define KERNEL_FEATURE_NODE_MASK_SURFACE_BACKGROUND \
  (KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT | KERNEL_FEATURE_NODE_AOV)
#define KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW \
  (KERNEL_FEATURE_NODE_BSDF | KERNEL_FEATURE_NODE_EMISSION | KERNEL_FEATURE_NODE_BUMP | \
   KERNEL_FEATURE_NODE_BUMP_STATE | KERNEL_FEATURE_NODE_VORONOI_EXTRA | \
   KERNEL_FEATURE_NODE_LIGHT_PATH | KERNEL_FEATURE_NODE_PRINCIPLED_HAIR | \
   KERNEL_FEATURE_NODE_PORTAL)
#define KERNEL_FEATURE_NODE_MASK_SURFACE \
  (KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW | KERNEL_FEATURE_NODE_RAYTRACE | \
   KERNEL_FEATURE_NODE_AOV | KERNEL_FEATURE_NODE_LIGHT_PATH)
#define KERNEL_FEATURE_NODE_MASK_VOLUME \
  (KERNEL_FEATURE_NODE_EMISSION | KERNEL_FEATURE_NODE_VOLUME | \
   KERNEL_FEATURE_NODE_VORONOI_EXTRA | KERNEL_FEATURE_NODE_LIGHT_PATH | \
   KERNEL_FEATURE_NODE_PORTAL)
#define KERNEL_FEATURE_NODE_MASK_DISPLACEMENT \
  (KERNEL_FEATURE_NODE_VORONOI_EXTRA | KERNEL_FEATURE_NODE_BUMP | \
   KERNEL_FEATURE_NODE_BUMP_STATE | KERNEL_FEATURE_NODE_PORTAL)
#define KERNEL_FEATURE_NODE_MASK_BUMP KERNEL_FEATURE_NODE_MASK_DISPLACEMENT

#define IF_KERNEL_FEATURE(feature) \
  if constexpr ((node_feature_mask & (KERNEL_FEATURE_##feature)) != 0U)
#define IF_KERNEL_NODES_FEATURE(feature) \
  if constexpr ((node_feature_mask & (KERNEL_FEATURE_NODE_##feature)) != 0U)
#define IF_NOT_KERNEL_NODES_FEATURE(feature) \
  if constexpr ((node_feature_mask & (KERNEL_FEATURE_NODE_##feature)) == 0U)

/* Kernel Feature Guards
 *
 * These are used throughout the code to disable code for certain features entirely.
 * They can be commented out for faster builds, or to measure the performance impact
 * of the existence of the extra code in the kernel.
 *
 * Commenting out the first set of feature guards should work well to reduce kernel
 * compile time while still working for many scenes. */

#define __AO__
#define __LIGHT_LINKING__
#define __MNEE__
#define __OBJECT_MOTION__
#define __POINTCLOUD__
#define __PRINCIPLED_HAIR__
#define __SHADER_RAYTRACE__
#define __SHADOW_CATCHER__
#define __SHADOW_LINKING__
#define __GSPLATS__
#define __SUBSURFACE__
#define __TRANSPARENT_SHADOWS__
#define __VOLUME__
#define __VOLUME_IOR__
#define __NESTED_DIELECTRICS__

#define __CAUSTICS_TRICKS__
#define __CLAMP_SAMPLE__
#define __DENOISING_FEATURES__
#define __DPDU__
#define __HAIR__
#define __LIGHT_TREE__
#define __PASSES__
#define __RAY_DIFFERENTIALS__
#define __VISIBILITY_FLAG__
#define __SVM__
#define __SPECTRAL__

/* Device specific features */

#ifdef WITH_OSL
#  define __OSL__
#  ifdef __KERNEL_OPTIX__
/* Kernels with OSL support are built separately in OptiX and don't need SVM. */
#    undef __SVM__
#  endif
#endif
#ifndef __KERNEL_GPU__
#  if defined(WITH_PATH_GUIDING)
#    define __PATH_GUIDING__
#  endif
#  define __VOLUME_RECORD_ALL__
#endif /* !__KERNEL_GPU__ */

/* Light-cache transport (bidirectional path tracing and photon mapping) and pixel-level
 * displacement. Metal and CPU kernels always contain them. CUDA, OptiX and HIP build them into a
 * separate "transport" kernel module (__KERNEL_GPU_TRANSPORT__), which the device loads only for
 * scenes that use these features, so ordinary path tracing keeps the smaller kernels. */
#if defined(__KERNEL_METAL__) || !defined(__KERNEL_GPU__) || defined(__KERNEL_GPU_TRANSPORT__)
#  define __BDPT__
#  define __PHOTON_MAPPING__
#  define __PIXEL_DISPLACEMENT__
#endif
/* Path guiding with a field trained on the GPU, see `integrator/guiding_gpu.h`. CUDA and OptiX
 * build it into their transport kernels. The CPU guides with OpenPGL instead. */
#if defined(__KERNEL_METAL__) || (defined(__KERNEL_CUDA__) && defined(__KERNEL_GPU_TRANSPORT__))
#  define __GPU_GUIDING__
#endif
/* Diffraction closures, coherent light, polarization, nested dielectrics and volumes with an
 * index of refraction.
 * Metal removes them from the kernels of scenes without them, as it specializes kernels for the
 * scene features. CUDA and OptiX kernels are not specialized: their regular kernels would carry
 * the registers and stack memory of these features into every render. They are built into the
 * transport kernels instead, which the device loads for scenes that use them. The OptiX OSL
 * kernels do not have them: with all of them, a single OSL module needs over 27 GB of memory to
 * compile. Scenes that use them with OSL shading on OptiX are refused by the device. */
#if !defined(__KERNEL_CUDA__) || defined(__KERNEL_GPU_TRANSPORT__)
#  define __COHERENT_LIGHT__
#  define __DIFFRACTION__
#  define __POLARIZATION__
#else
#  undef __NESTED_DIELECTRICS__
#  undef __VOLUME_IOR__
#endif
/* Scene settings of coherent light, false in kernels without it so that its code is removed. */
#ifdef __COHERENT_LIGHT__
#  define kernel_use_coherent_specular() (kernel_data.integrator.coherent_specular_enabled != 0)
#  define kernel_use_coherent_direct() \
    ((kernel_data.kernel_features & KERNEL_FEATURE_COHERENT_DIRECT) != 0)
#else
#  define kernel_use_coherent_specular() false
#  define kernel_use_coherent_direct() false
#endif
#ifdef __POLARIZATION__
#  define kernel_use_polarization() \
    ((kernel_data.kernel_features & KERNEL_FEATURE_POLARIZATION) != 0)
#else
#  define kernel_use_polarization() false
#endif
/* Metal compiles displaced-surface intersection and shading into scene specialized variants
 * only. CPU translation units that render (the kernels and OSL services) always contain them and
 * test the scene settings at runtime; they define __KERNEL_CPU_PIXEL_DISPLACEMENT__ and include
 * `kernel/geom/pixel_displacement_shader.h`. Other host code using kernel headers does not.
 * The CUDA transport kernels contain them and test the scene settings at runtime too. OptiX does
 * not intersect displaced surfaces, the host renders them with ordinary displacement there. */
#if defined(__KERNEL_METAL_PIXEL_DISPLACEMENT__) || \
    (!defined(__KERNEL_GPU__) && defined(__KERNEL_CPU_PIXEL_DISPLACEMENT__)) || \
    (defined(__KERNEL_GPU_TRANSPORT__) && !defined(__KERNEL_OPTIX__))
#  define __PIXEL_DISPLACEMENT_INTERSECT__
#endif
#if defined(__KERNEL_METAL_PIXEL_DISPLACEMENT_SHADE__) || \
    (!defined(__KERNEL_GPU__) && defined(__KERNEL_CPU_PIXEL_DISPLACEMENT__)) || \
    (defined(__KERNEL_GPU_TRANSPORT__) && !defined(__KERNEL_OPTIX__))
#  define __PIXEL_DISPLACEMENT_SHADE__
#endif

/* Scene-based selective features compilation. */

#ifdef __KERNEL_FEATURES__
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_OBJECT_MOTION)
#    undef __OBJECT_MOTION__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_HAIR)
#    undef __HAIR__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_POINTCLOUD)
#    undef __POINTCLOUD__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_VOLUME)
#    undef __VOLUME__
#    undef __VOLUME_IOR__
#    if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_TRANSPARENT)
#      undef __TRANSPARENT_SHADOWS__
#    endif
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_VOLUME_IOR)
#    undef __VOLUME_IOR__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_NESTED_DIELECTRICS)
#    undef __NESTED_DIELECTRICS__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_SUBSURFACE)
#    undef __SUBSURFACE__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_SHADOW_CATCHER)
#    undef __SHADOW_CATCHER__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_DENOISING)
#    undef __DENOISING_FEATURES__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_AO)
#    undef __AO__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_MNEE)
#    undef __MNEE__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_PATH_GUIDING)
#    undef __PATH_GUIDING__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_NODE_PRINCIPLED_HAIR)
#    undef __PRINCIPLED_HAIR__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_LIGHT_LINKING)
#    undef __LIGHT_LINKING__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_SHADOW_LINKING)
#    undef __SHADOW_LINKING__
#  endif
#  if !(__KERNEL_FEATURES__ & KERNEL_FEATURE_GSPLATS)
#    undef __GSPLATS__
#  endif
#endif

#ifdef WITH_CYCLES_DEBUG_NAN
#  define __KERNEL_DEBUG_NAN__
#endif

/* Features that enable others */

#if defined(__SUBSURFACE__) || defined(__SHADER_RAYTRACE__)
#  define __BVH_LOCAL__
#endif

/* Feature mask for integrator states, to allocate state memory only when
 * certain features are on or off. */

struct KernelFeatureRequest {
  uint64_t on;
  uint64_t off;

  explicit KernelFeatureRequest(const uint64_t on) : on(on), off(0) {}
  KernelFeatureRequest(const uint64_t on, const uint64_t off) : on(on), off(off) {}

  bool test(const uint64_t features) const
  {
    return (features & on) && !(features & off);
  }
};

CCL_NAMESPACE_END
