cmake_minimum_required(VERSION 3.12)
project(stl_bindings)

# Buscar donde esta instalado PyBind11
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Agregar el modulo
add_library(stl_bindings MODULE archivo.cpp)
set_target_properties(stl_bindings PROPERTIES PREFIX "" SUFFIX ".so")

# Enlazar con PyBind11
target_link_libraries(stl_bindings PRIVATE pybind11::module)