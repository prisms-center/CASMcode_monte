#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "pybind11_json/pybind11_json.hpp"

// CASM
#include <casm/crystallography/BasicStructure.hh>
#include <casm/monte/Conversions.hh>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

std::shared_ptr<monte::Conversions> make_monte_conversions(
    xtal::BasicStructure const &prim,
    Eigen::Matrix3l const &transformation_matrix_to_super) {
  return std::make_shared<monte::Conversions>(
      monte::Conversions(prim, transformation_matrix_to_super));
};
}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(_monte, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Methods for evaluating functions of configurations

        libcasm.monte
        -------------

    )pbdoc";
  py::module::import("libcasm.xtal");

  py::class_<monte::Conversions, std::shared_ptr<monte::Conversions>>(
      m, "Conversions", R"pbdoc(
      Data structure used for Conversions


      )pbdoc")
      .def(py::init<>(&make_monte_conversions),
           R"pbdoc(
          Construct Monte Conversions 
          )pbdoc",
           py::arg("xtal_prim"), py::arg("transformation_matrix_to_super"))
      .def("lat_column_mat",
           [](monte::Conversions const &conversions) {
             return conversions.lat_column_mat();
           })
      .def("l_size",
           [](monte::Conversions const &conversions) {
             return conversions.l_size();
           })
      .def(
          "l_to_b",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_b(l);
          },
          py::arg("l"))
      .def(
          "l_to_ijk",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_ijk(l);
          },
          py::arg("l"))
      .def(
          "l_to_bijk",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_bijk(l);
          },
          py::arg("l"))
      .def(
          "l_to_unitl",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_unitl(l);
          },
          py::arg("l"))
      .def(
          "l_to_asym",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_asym(l);
          },
          py::arg("l"))
      .def(
          "l_to_cart",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_cart(l);
          },
          py::arg("l"))
      .def(
          "l_to_frac",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_frac(l);
          },
          py::arg("l"))
      .def(
          "l_to_basis_cart",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_basis_cart(l);
          },
          py::arg("l"))
      .def(
          "l_to_basis_frac",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_basis_frac(l);
          },
          py::arg("l"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
