
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/LU>
#include <Eigen/SVD>
#include <unsupported/Eigen/SpecialFunctions>

namespace py = pybind11;
constexpr double pi = 3.141592653589793238462643383279502884;

#include "model.hh"
#include "matrix.hh"
#include "matrices.hh"
#include "functions.hh"

PYBIND11_MODULE(vbasa, m) {
  /* vbasa.Model */
  py::class_<vbasa::Model>(m, "Model")
    /* constructor: */
    .def(py::init<double, double, double, double>(),
         py::arg("prior_alpha") = 1e-3,
         py::arg("prior_beta") = 1e-3,
         py::arg("prior_gamma") = 1e-4,
         py::arg("prior_rho") = 1e-6)
    /* sizes: */
    .def_property_readonly(
         "m",
         &vbasa::Model::num_target,
         "Number of target points")
    .def_property_readonly(
         "n",
         &vbasa::Model::num_source,
         "Number of source points")
    .def_property_readonly(
         "k",
         &vbasa::Model::dims,
         "Point cloud dimensionality")
    .def_property_readonly(
         "sizes",
         &vbasa::Model::sizes,
         "Triple of (m,n,k)")
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
         "lmb", &vbasa::Model::lambda,
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
    .def_property_readonly(
         "bound",
         &vbasa::Model::bound,
         "Variational lower bound on the model evidence (ELBO)")
    /* methods: */
    .def("observe",
         &vbasa::Model::observe,
         py::arg("target_points_alpha"),
         py::arg("target_points_beta"),
         py::arg("source_points_alpha"),
         py::arg("source_points_beta"),
         py::arg("prior_matching") = py::none(),
         "Observe a dataset of two paired point clouds")
    .def("step",
         &vbasa::Model::step,
         py::arg("modify") = true,
         "Perform an optimization step")
    .def("optimize",
         &vbasa::Model::optimize,
         py::arg("iterations") = 1000,
         py::arg("absolute_tolerance") = 0.001,
         py::arg("relative_tolerance") = 0.01,
         "Optimize the variational lower bound");

  /* vbasa.SubstitutionMatrix */
  py::class_<vbasa::SubstitutionMatrix>(m, "SubstitutionMatrix")
    .def(py::init<const vbasa::SubstitutionMatrix::sequence_t&,
                  std::optional<const Eigen::Ref<const Eigen::MatrixXd>>,
                  double>(),
         py::arg("vocab"),
         py::arg("init") = py::none(),
         py::arg("scaling") = 1.0)
    .def_readwrite(
         "scale", &vbasa::SubstitutionMatrix::scale,
         "Factor to scale scores by to get log-probability")
    .def_property(
         "coeffs",
         &vbasa::SubstitutionMatrix::get_coeffs,
         &vbasa::SubstitutionMatrix::set_coeffs,
         "The complete matrix")
    .def_property_readonly(
         "vocab",
         &vbasa::SubstitutionMatrix::get_labels,
         "The substitution matrix vocabulary")
    .def("__getitem__",
         &vbasa::SubstitutionMatrix::get_item,
         py::arg("key"),
         "Get the value of a matrix element")
    .def("__setitem__",
         &vbasa::SubstitutionMatrix::set_item,
         py::arg("key"),
         py::arg("value"),
         "Set the value of a matrix element")
    .def("__call__",
         &vbasa::SubstitutionMatrix::eval,
         py::arg("a"),
         py::arg("b"),
         "Evaluate the matrix on a pair of sequences");

  /* vbasa.model() */
  m.def("model",
        &vbasa::create_model,
        py::arg("target_points_alpha"),
        py::arg("target_points_beta"),
        py::arg("source_points_alpha"),
        py::arg("source_points_beta"),
        py::arg("target_sequence"),
        py::arg("source_sequence"),
        py::arg("prior_noise_shape") = 1e-3,
        py::arg("prior_noise_rate") = 1e-3,
        py::arg("prior_precision") = 1e-4,
        py::arg("prior_diag") = 1e-6,
        py::arg("sub_matrix") = py::none(),
        "Create an initialize a new model from point cloud data");

  /* vbasa.matrices */
  py::module matrices = m.def_submodule("matrices");

  /* vbasa.matrices.pam250() */
  matrices.def("pam250",
               &vbasa::matrices::pam250,
               py::arg("scale") = 1.0,
               "Create a PAM250 substitution matrix");

  /* vbasa.matrices.blosum62() */
  matrices.def("blosum62",
               &vbasa::matrices::blosum62,
               py::arg("scale") = 1.0,
               "Create a BLOSUM62 substitution matrix");
}

