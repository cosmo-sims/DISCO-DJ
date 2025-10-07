cmake_minimum_required(VERSION 3.21)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)


project(DISCO-DJ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# need to set to native to be able to use BMI2 intrinsics
add_compile_options(-march=native -mtune=native)

# Set source directory
set(SOURCE_DIR "src/discodj_native")

file( GLOB SOURCES
  ${SOURCE_DIR}/*.cc
)

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})

FIND_PACKAGE(GSL REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
FIND_PACKAGE(pybind11 REQUIRED)

# Generate Python module
# add_subdirectory(pybind11)

pybind11_add_module(_discodj_native  MODULE ${SOURCES})
target_link_libraries(_discodj_native PUBLIC GSL::gsl GSL::gslcblas)

install(TARGETS _discodj_native LIBRARY DESTINATION "discodj_native")
