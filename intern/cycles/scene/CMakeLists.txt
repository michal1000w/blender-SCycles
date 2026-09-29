# SPDX-FileCopyrightText: 2011-2021 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

set(INC
  ..
  ../../sky/include
  ../../mikktspace
)

set(INC_SYS
)

set(SRC
  attribute.cpp
  background.cpp
  bake.cpp
  camera.cpp
  constant_fold.cpp
  devicescene.cpp
  diffraction_albedo.cpp
  diffraction.cpp
  diffraction_tensor.cpp
  diffraction_manager.cpp
  diffraction_manager_device.cpp
  film.cpp
  geometry.cpp
  geometry_attributes.cpp
  geometry_bvh.cpp
  geometry_mesh.cpp
  hair.cpp
  image.cpp
  image_cache.cpp
  image_loader.cpp
  image_oiio.cpp
  image_sky.cpp
  image_vdb.cpp
  integrator.cpp
  scene_attributes.cpp
  light.cpp
  light_tree.cpp
  light_tree_debug.cpp
  mesh.cpp
  mesh_displace.cpp
  mesh_subdivision.cpp
  procedural.cpp
  pointcloud.cpp
  object.cpp
  osl.cpp
  particles.cpp
  pass.cpp
  curves.cpp
  scene.cpp
  shader.cpp
  shader_displacement.cpp
  shader_displacement_image.cpp
  shader.tables
  shader_graph.cpp
  shader_nodes.cpp
  stats.cpp
  svm.cpp
  tables.cpp
  tabulated_sobol.cpp
  volume.cpp
)

set(SRC_HEADERS
  attribute.h
  bake.h
  background.h
  camera.h
  constant_fold.h
  coherent_convex_mesh.h
  coherent_planar_cluster.h
  coherent_sphere_host.h
  devicescene.h
  diffraction_albedo.h
  diffraction.h
  diffraction_index.h
  diffraction_convergence.h
  diffraction_tensor.h
  diffraction_manager.h
  film.h
  geometry.h
  hair.h
  image.h
  image_cache.h
  image_loader.h
  image_oiio.h
  image_sky.h
  image_vdb.h
  integrator.h
  light.h
  light_tree.h
  light_tree_debug.h
  mesh.h
  object.h
  osl.h
  particles.h
  pass.h
  procedural.h
  pointcloud.h
  curves.h
  scene.h
  scene_attributes.h
  shader.h
  shader_displacement.h
  shader_graph.h
  shader_nodes.h
  stats.h
  svm.h
  tables.h
  tabulated_sobol.h
  volume.h
)

set(LIB
  PUBLIC cycles_bvh
  PUBLIC cycles_device
  PUBLIC cycles_integrator
  PUBLIC cycles_subd
  PUBLIC cycles_util
  PRIVATE bf::dependencies::optional::openvdb
  PRIVATE bf::dependencies::optional::nanovdb
)

if(CYCLES_STANDALONE_REPOSITORY)
  find_package(Eigen3 REQUIRED CONFIG)
  list(APPEND LIB PRIVATE extern_sky Eigen3::Eigen)
else()
  list(APPEND LIB PRIVATE bf_intern_sky bf::dependencies::eigen)
endif()

if(WITH_CYCLES_OSL)
  list(APPEND LIB
    PRIVATE cycles_kernel_osl
  )
endif()

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

cycles_add_library(cycles_scene "${LIB}" ${SRC} ${SRC_HEADERS})
