#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Para la integración de STL
#include <vector>
#include <map>

namespace py = pybind11;

// Función que suma dos vectores
std::vector<int> sum_vectors(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result;
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] + b[i]);
    }
    return result;
}

// Función que crea un mapa y lo retorna
std::map<std::string, int> create_map() {
    std::map<std::string, int> my_map;
    my_map["uno"] = 1;
    my_map["dos"] = 2;
    my_map["tres"] = 3;
    return my_map;
}




// Definir el módulo y las funciones a exportar
PYBIND11_MODULE(stl_bindings, m) {
    m.def("sum_vectors", &sum_vectors, "Suma dos vectores de enteros");
    m.def("create_map", &create_map, "Crea un mapa de ejemplo");
    m.def("saludo", []() {
        return "Hola, desde C++!";
    });
}