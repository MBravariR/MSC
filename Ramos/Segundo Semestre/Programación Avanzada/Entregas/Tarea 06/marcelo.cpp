#include <iostream>
#include <pybind11/pybind11.h>

std::string imprimir() {
    return "Hola, mi nombre es Marcelo Bravari.";
}

PYBIND11_MODULE(stl_bindings, m) {
    m.def("imprimir", &imprimir, "Una función que imprime un mensaje.");
}