
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
constexpr double pi = 3.141592653589793238462643383279502884;

#include "model.hh"
#include "functions.hh"

PYBIND11_MODULE(vbasa, m) {
  /* vbasa.Model */
  py::class_<vbasa::Model>(m, "Model")
    /* constructor: */
    .def(py::init<double, double, double>(),
         py::arg("prior_alpha"),
         py::arg("prior_beta"),
         py::arg("prior_gamma"))
    /* sizes: */
    .def_property_readonly(
         "m", [](vbasa::Model& model) -> int { return model.Xa.rows(); },
         "Number of target points")
    .def_property_readonly(
         "n", [](vbasa::Model& model) -> int { return model.Ya.rows(); },
         "Number of source points")
    .def_property_readonly(
         "k", [](vbasa::Model& model) -> int { return model.Xa.cols(); },
         "Point cloud dimensionality")
    /* input point clouds: */
    .def_readonly(
         "Xa", &vbasa::Model::Xa,
         "Target point cloud (alpha)")
    .def_readonly(
         "Xb", &vbasa::Model::Xb,
         "Target point cloud (beta)")
    .def_readonly(
         "Ya", &vbasa::Model::Ya,
         "Source point cloud (alpha)")
    .def_readonly(
         "Yb", &vbasa::Model::Yb,
         "Source point cloud (beta)")
    /* prior parameters: */
    .def_readonly(
         "alpha0", &vbasa::Model::alpha0,
         "Prior shape of noise precision")
    .def_readonly(
         "beta0", &vbasa::Model::beta0,
         "Prior rate of noise precision")
    .def_readonly(
         "gamma0", &vbasa::Model::gamma0,
         "Prior precision of the translation vector")
    .def_readonly(
         "omega0", &vbasa::Model::omega0,
         "Prior assignment weights")
    .def_readonly(
         "A0", &vbasa::Model::A0,
         "Prior rotation parameter")
    /* parameters: */
    .def_readonly(
         "alpha", &vbasa::Model::alpha,
         "Shape of noise precision")
    .def_readonly(
         "beta", &vbasa::Model::beta,
         "Rate of noise precision")
    .def_readonly(
         "eta", &vbasa::Model::eta,
         "Shift parameter of the translation vector")
    .def_readonly(
         "A", &vbasa::Model::A,
         "Rotation parameter")
    .def_readonly(
         "gamma", &vbasa::Model::gamma,
         "Precision of the translation vector")
    /* expectations: */
    .def_readonly(
         "lambda", &vbasa::Model::lambda,
         "Expected noise precision")
    .def_readonly(
         "t", &vbasa::Model::t,
         "Expected translation")
    .def_readonly(
         "R", &vbasa::Model::R,
         "Expected rotation")
    .def_readonly(
         "Z", &vbasa::Model::Z,
         "Expected assigmnets")
    /* methods: */
    .def("observe",
         &vbasa::Model::observe,
         py::arg("target_points_alpha"),
         py::arg("target_points_beta"),
         py::arg("source_points_alpha"),
         py::arg("source_points_beta"),
         "Observe a dataset of two paired point clouds")
    .def("step",
         &vbasa::Model::step,
         "Perform an optimization step")
    .def("optimize",
         &vbasa::Model::optimize,
         py::arg("iterations") = 1,
         py::arg("absolute_tolerance") = 1e3,
         py::arg("relative_tolerance") = 0.001,
         "Optimize the variational lower bound");

  /* vbasa.model() */
  m.def("model",
        &vbasa::create_model,
        py::arg("target_points_alpha"),
        py::arg("target_points_beta"),
        py::arg("source_points_alpha"),
        py::arg("source_points_beta"),
        py::arg("prior_noise_shape") = 0.001,
        py::arg("prior_noise_rate") = 0.001,
        py::arg("prior_precision") = 0.001,
        "Create an initialize a new model from point cloud data");
}

