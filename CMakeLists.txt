cmake_minimum_required(VERSION 3.24...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(vbasa MODULE src/vbasa.cc)

set_target_properties(vbasa PROPERTIES COMPILE_WARNING_AS_ERROR ON)
target_compile_options(vbasa PRIVATE -Wall -Wextra)
target_link_libraries(vbasa PRIVATE Eigen3::Eigen)
install(TARGETS vbasa DESTINATION .)
