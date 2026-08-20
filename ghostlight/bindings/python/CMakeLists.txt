cmake_minimum_required(VERSION 3.18)
project(ghostlight_python LANGUAGES CXX CUDA)

# Pull in ghostlight_core from the repo root (two directories up) unless it is
# already a known target (e.g. when built via the top-level CMakeLists with
# GHOSTLIGHT_PYTHON=ON).
if(NOT TARGET ghostlight_core)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. ghostlight_core_build)
endif()

# scikit-build-core adds the pip-installed pybind11 CMake config dir to
# CMAKE_PREFIX_PATH automatically, so this resolves without a git submodule.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ghostlight
    src/bindings.cpp
    src/bind_geometry.cpp
    src/bind_trace.cpp
    src/bind_calibration.cpp
    src/bind_ghost.cpp
    src/bind_config.cpp
    src/bind_renderers.cpp
)

target_link_libraries(_ghostlight PRIVATE ghostlight_core)

set_target_properties(_ghostlight PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY None
    MSVC_RUNTIME_LIBRARY MultiThreadedDLL
)

target_include_directories(_ghostlight PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/../../third_party
)

# Propagate the same Windows defines as ghostlight_core so shared headers compile.
if(WIN32)
    target_compile_definitions(_ghostlight PRIVATE
        NOMINMAX
        _USE_MATH_DEFINES
        WIN32_LEAN_AND_MEAN
    )
endif()

# scikit-build-core uses this install destination to place the .pyd/.so inside
# the ghostlight/ package directory in the wheel.
install(TARGETS _ghostlight LIBRARY DESTINATION ghostlight)
