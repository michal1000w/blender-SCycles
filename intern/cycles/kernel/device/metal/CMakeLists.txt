# SPDX-FileCopyrightText: 2011-2026 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

set(INC
  ../../..
)

set(INC_SYS

)

set(SRC_KERNEL_DEVICE_METAL
  kernel.metal
)

set(SRC_KERNEL_DEVICE_METAL_HEADERS
  bvh.h
  compat.h
  context_begin.h
  context_end.h
  function_constants.h
  globals.h
)


set(LIB

)

if(WITH_CYCLES_DEVICE_METAL)
  add_library(cycles_kernel_metal INTERFACE)
  target_sources(cycles_kernel_metal INTERFACE
    ${SRC_KERNEL_DEVICE_METAL}
    ${SRC_KERNEL_DEVICE_METAL_HEADERS}
  )
  cycles_set_solution_folder(cycles_kernel_metal)

  source_group("device\\metal" FILES ${SRC_KERNEL_DEVICE_METAL} ${SRC_KERNEL_DEVICE_METAL_HEADERS})

  add_dependencies(cycles_kernel cycles_kernel_metal)

  # Runtime MSL compilation can take several minutes for the complete Cycles kernel. Build the
  # common generic libraries ahead of time instead. Scene-specialized variants still use the
  # runtime compiler, and the device code falls back to it whenever none of these exact variants
  # match the current OS or debug settings.
  if(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "(^|;)arm64(;|$)")
    find_program(CYCLES_METAL_XCRUN_EXECUTABLE xcrun REQUIRED)
    execute_process(
      COMMAND sw_vers -productVersion
      OUTPUT_VARIABLE CYCLES_METAL_MACOS_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "^[0-9]+" CYCLES_METAL_MACOS_MAJOR "${CYCLES_METAL_MACOS_VERSION}")

    if(CYCLES_METAL_MACOS_MAJOR GREATER_EQUAL 15)
      set(CYCLES_METAL_LANGUAGE_VERSION metal3.2)
    elseif(CYCLES_METAL_MACOS_MAJOR GREATER_EQUAL 14)
      set(CYCLES_METAL_LANGUAGE_VERSION metal3.1)
    elseif(CYCLES_METAL_MACOS_MAJOR GREATER_EQUAL 13)
      set(CYCLES_METAL_LANGUAGE_VERSION metal3.0)
    else()
      set(CYCLES_METAL_LANGUAGE_VERSION metal2.4)
    endif()

    set(cycles_metal_common_flags
      -c
      -std=${CYCLES_METAL_LANGUAGE_VERSION}
      -ffast-math
      -fmodules-cache-path=${CMAKE_CURRENT_BINARY_DIR}/module_cache
      -I${CMAKE_SOURCE_DIR}/intern/cycles
      -D__KERNEL_LOCAL_ATOMIC_SORT__
      -D__KERNEL_METAL_APPLE__
      -D__METAL_FUNCTION_CONSTANTS_64BIT__
      -D__KERNEL_METAL_TARGET_CPU_ARM64__
      -D__KERNEL_METAL_MACOS__=${CYCLES_METAL_MACOS_MAJOR}
    )
    if(CYCLES_METAL_MACOS_MAJOR GREATER_EQUAL 14)
      list(APPEND cycles_metal_common_flags -D__METAL_GLOBAL_BUILTINS__)
    endif()
    if(WITH_NANOVDB)
      list(APPEND cycles_metal_common_flags -DWITH_NANOVDB)
    endif()
    if(WITH_CYCLES_DEBUG)
      list(APPEND cycles_metal_common_flags -DWITH_CYCLES_DEBUG)
    endif()

    set(cycles_metal_generic_libraries)
    foreach(metal_variant software metalrt metalrt_motion)
      set(metal_variant_flags)
      if(metal_variant MATCHES "^metalrt")
        list(APPEND metal_variant_flags -D__KERNEL_METALRT__)
      endif()
      if(metal_variant STREQUAL "metalrt_motion")
        list(APPEND metal_variant_flags -D__METALRT_MOTION__)
      endif()

      set(metal_air
        ${CMAKE_CURRENT_BINARY_DIR}/cycles_kernel_metal_generic_${metal_variant}_macos_${CYCLES_METAL_MACOS_MAJOR}.air
      )
      set(metal_library
        ${CMAKE_CURRENT_BINARY_DIR}/cycles_kernel_metal_generic_${metal_variant}_macos_${CYCLES_METAL_MACOS_MAJOR}.metallib
      )
      add_custom_command(
        OUTPUT ${metal_air}
        COMMAND ${CYCLES_METAL_XCRUN_EXECUTABLE} -sdk macosx metal
          ${cycles_metal_common_flags}
          ${metal_variant_flags}
          -o ${metal_air}
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel.metal
        DEPENDS
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel.metal
          $<TARGET_PROPERTY:cycles_kernel,INTERFACE_SOURCES>
        VERBATIM
      )
      add_custom_command(
        OUTPUT ${metal_library}
        COMMAND ${CYCLES_METAL_XCRUN_EXECUTABLE} -sdk macosx metallib
          ${metal_air}
          -o ${metal_library}
        DEPENDS ${metal_air}
        VERBATIM
      )
      list(APPEND cycles_metal_generic_libraries ${metal_library})
      delayed_install("" "${metal_library}" ${CYCLES_INSTALL_PATH}/lib)
    endforeach()

    add_custom_target(cycles_kernel_metal_precompile DEPENDS ${cycles_metal_generic_libraries})
    cycles_set_solution_folder(cycles_kernel_metal_precompile)
    add_dependencies(cycles_kernel_metal cycles_kernel_metal_precompile)
  endif()
endif()

delayed_install(${CMAKE_CURRENT_SOURCE_DIR} "${SRC_KERNEL_DEVICE_METAL}" ${CYCLES_INSTALL_PATH}/source/kernel/device/metal)
delayed_install(${CMAKE_CURRENT_SOURCE_DIR} "${SRC_KERNEL_DEVICE_METAL_HEADERS}" ${CYCLES_INSTALL_PATH}/source/kernel/device/metal)
