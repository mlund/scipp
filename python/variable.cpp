// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock

#include "scipp/units/unit.h"

#include "scipp/common/numeric.h"

#include "scipp/core/dtype.h"
#include "scipp/core/except.h"
#include "scipp/core/time_point.h"

#include "scipp/variable/matrix.h"
#include "scipp/variable/operations.h"
#include "scipp/variable/rebin.h"
#include "scipp/variable/variable.h"

#include "scipp/dataset/dataset.h"
#include "scipp/dataset/util.h"

#include "bind_data_access.h"
#include "bind_operators.h"
#include "bind_slice_methods.h"
#include "docstring.h"
#include "dtype.h"
#include "make_variable.h"
#include "numpy.h"
#include "pybind11.h"
#include "rename.h"
#include "unit.h"

using namespace scipp;
using namespace scipp::variable;

namespace py = pybind11;

template <class T> void bind_init_0D(py::class_<Variable> &c) {
  c.def(py::init([](const T &value, const std::optional<T> &variance,
                    const units::Unit &unit) {
          return do_init_0D(value, variance, unit);
        }),
        py::arg("value"), py::arg("variance") = std::nullopt,
        py::arg("unit") = units::one);
  if constexpr (std::is_same_v<T, Variable> || std::is_same_v<T, DataArray> ||
                std::is_same_v<T, Dataset>) {
    c.def(py::init([](const T &value, const std::optional<T> &variance,
                      const units::Unit &unit) {
            return do_init_0D(copy(value), variance, unit);
          }),
          py::arg("value"), py::arg("variance") = std::nullopt,
          py::arg("unit") = units::one);
  }
}

// This function is used only to bind native python types: pyInt -> int64_t;
// pyFloat -> double; pyBool->bool
template <class T>
void bind_init_0D_native_python_types(py::class_<Variable> &c) {
  c.def(py::init([](const T &value, const std::optional<T> &variance,
                    const units::Unit &unit, py::object &dtype) {
          static_assert(std::is_same_v<T, int64_t> ||
                        std::is_same_v<T, double> || std::is_same_v<T, bool>);
          if (dtype.is_none())
            return do_init_0D(value, variance, unit);
          else {
            return MakeODFromNativePythonTypes<T>::make(unit, value, variance,
                                                        dtype);
          }
        }),
        py::arg("value").noconvert(), py::arg("variance") = std::nullopt,
        py::arg("unit") = units::one, py::arg("dtype") = py::none());
}

void bind_init_0D_numpy_types(py::class_<Variable> &c) {
  c.def(py::init([](py::buffer &b, const std::optional<py::buffer> &v,
                    const units::Unit &unit, py::object &dtype) {
          static auto np_datetime64_type =
              py::module::import("numpy").attr("datetime64").get_type();

          py::buffer_info info = b.request();
          if (info.ndim == 0) {
            auto arr = py::array(b);
            auto varr = v ? std::optional{py::array(*v)} : std::nullopt;
            return doMakeVariable({}, arr, varr, unit, dtype);
          } else if (info.ndim == 1 &&
                     scipp_dtype(dtype) == core::dtype<Eigen::Vector3d>) {
            return do_init_0D<Eigen::Vector3d>(
                b.cast<Eigen::Vector3d>(),
                v ? std::optional(v->cast<Eigen::Vector3d>()) : std::nullopt,
                unit);
          } else if (info.ndim == 2 &&
                     scipp_dtype(dtype) == core::dtype<Eigen::Matrix3d>) {
            return do_init_0D<Eigen::Matrix3d>(
                b.cast<Eigen::Matrix3d>(),
                v ? std::optional(v->cast<Eigen::Matrix3d>()) : std::nullopt,
                unit);
          } else if ((info.ndim == 1) &&
                     py::isinstance(b.get_type(), np_datetime64_type)) {
            if (v.has_value()) {
              throw except::VariancesError("datetimes cannot have variances.");
            }
            const auto [actual_unit, value_factor] =
                get_time_unit(b, dtype, unit);
            return do_init_0D<core::time_point>(
                make_time_point(b, value_factor), std::nullopt, actual_unit);

          } else {
            throw scipp::except::VariableError(
                "Wrong overload for making 0D variable.");
          }
        }),
        py::arg("value").noconvert(), py::arg("variance") = std::nullopt,
        py::arg("unit") = units::one, py::arg("dtype") = py::none());
}

void bind_init_list(py::class_<Variable> &c) {
  c.def(py::init([](const std::array<Dim, 1> &label, const py::list &values,
                    const std::optional<py::list> &variances,
                    const units::Unit &unit, py::object &dtype) {
          auto arr = py::array(values);
          auto varr =
              variances ? std::optional(py::array(*variances)) : std::nullopt;
          auto dims = std::vector<Dim>{label[0]};
          return doMakeVariable(dims, arr, varr, unit, dtype);
        }),
        py::arg("dims"), py::arg("values"), py::arg("variances") = std::nullopt,
        py::arg("unit") = units::one, py::arg("dtype") = py::none());
}

void bind_init_0D_list_eigen(py::class_<Variable> &c) {
  c.def(
      py::init([](const py::list &value,
                  const std::optional<py::list> &variance,
                  const units::Unit &unit, py::object &dtype) {
        if (scipp_dtype(dtype) == core::dtype<Eigen::Vector3d>) {
          return do_init_0D<Eigen::Vector3d>(
              Eigen::Vector3d(value.cast<std::vector<double>>().data()),
              variance ? std::optional(variance->cast<Eigen::Vector3d>())
                       : std::nullopt,
              unit);
        } else {
          throw scipp::except::VariableError(
              "Cannot create 0D Variable from list of values with this dtype.");
        }
      }),
      py::arg("value"), py::arg("variance") = std::nullopt,
      py::arg("unit") = units::one, py::arg("dtype") = py::none());
}

void init_variable(py::module &m) {
  // Needed to let numpy arrays keep alive the scipp buffers.
  // VariableConcept must ALWAYS be passed to Python by its handle.
  py::class_<VariableConcept, VariableConceptHandle> variable_concept(
      m, "_VariableConcept");

  py::class_<Variable> variable(m, "Variable",
                                R"(
Array of values with dimension labels and a unit, optionally including an array
of variances.)");
  bind_init_0D<Variable>(variable);
  bind_init_0D<DataArray>(variable);
  bind_init_0D<Dataset>(variable);
  bind_init_0D<std::string>(variable);
  bind_init_0D<Eigen::Vector3d>(variable);
  bind_init_0D<Eigen::Matrix3d>(variable);
  variable
      .def(py::init(&makeVariableDefaultInit),
           py::arg("dims") = std::vector<Dim>{},
           py::arg("shape") = std::vector<scipp::index>{},
           py::arg("unit") = units::one,
           py::arg("dtype") = py::dtype::of<double>(),
           py::arg("variances").noconvert() = false)
      .def(py::init(&doMakeVariable), py::arg("dims"),
           py::arg("values"), // py::array
           py::arg("variances") = std::nullopt, py::arg("unit") = units::one,
           py::arg("dtype") = py::none())
      .def("rename_dims", &rename_dims<Variable>, py::arg("dims_dict"),
           "Rename dimensions.")
      .def_property_readonly("dtype", &Variable::dtype)
      .def(
          "__radd__", [](Variable &a, double &b) { return a + b * units::one; },
          py::is_operator())
      .def(
          "__radd__", [](Variable &a, int &b) { return a + b * units::one; },
          py::is_operator())
      .def(
          "__rsub__", [](Variable &a, double &b) { return b * units::one - a; },
          py::is_operator())
      .def(
          "__rsub__", [](Variable &a, int &b) { return b * units::one - a; },
          py::is_operator())
      .def(
          "__rmul__",
          [](Variable &a, double &b) { return a * (b * units::one); },
          py::is_operator())
      .def(
          "__rmul__", [](Variable &a, int &b) { return a * (b * units::one); },
          py::is_operator())
      .def(
          "__rtruediv__",
          [](Variable &a, double &b) { return (b * units::one) / a; },
          py::is_operator())
      .def(
          "__rtruediv__",
          [](Variable &a, int &b) { return (b * units::one) / a; },
          py::is_operator())
      .def("__sizeof__",
           [](const Variable &self) {
             return size_of(self, SizeofTag::ViewOnly);
           })
      .def("underlying_size",
           [](const Variable &self) {
             return size_of(self, SizeofTag::Underlying);
           })
      .def("elems",
           [](Variable &self, const scipp::index i) -> py::object {
             if (self.dtype() != dtype<Eigen::Vector3d>)
               return py::none();
             return py::cast(self.elements<Eigen::Vector3d>(i));
           })
      .def("elems",
           [](Variable &self, scipp::index i, scipp::index j) -> py::object {
             if (self.dtype() != dtype<Eigen::Matrix3d>)
               return py::none();
             return py::cast(self.elements<Eigen::Matrix3d>(i, j));
           });

  bind_init_list(variable);
  // Order matters for pybind11's overload resolution. Do not change.
  bind_init_0D_numpy_types(variable);
  bind_init_0D_native_python_types<bool>(variable);
  bind_init_0D_native_python_types<int64_t>(variable);
  bind_init_0D_native_python_types<double>(variable);
  bind_init_0D<py::object>(variable);
  bind_init_0D_list_eigen(variable);
  //------------------------------------

  bind_common_operators(variable);

  bind_astype(variable);

  bind_slice_methods(variable);

  bind_comparison<Variable>(variable);

  bind_in_place_binary<Variable>(variable);
  bind_in_place_binary_scalars(variable);

  bind_binary<Variable>(variable);
  bind_binary<DataArray>(variable);
  bind_binary_scalars(variable);

  bind_unary(variable);

  bind_boolean_unary(variable);
  bind_logical<Variable>(variable);

  bind_data_properties(variable);

  py::implicitly_convertible<std::string, Dim>();

  m.def(
      "islinspace",
      [](const Variable &x) {
        if (x.dims().ndim() != 1)
          throw scipp::except::VariableError(
              "islinspace can only be called on a 1D Variable.");
        else
          return scipp::numeric::islinspace(x.template values<double>());
      },
      py::call_guard<py::gil_scoped_release>());

  m.def("rebin",
        py::overload_cast<const Variable &, const Dim, const Variable &,
                          const Variable &>(&rebin),
        py::arg("x"), py::arg("dim"), py::arg("old"), py::arg("new"),
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "vectors",
      [](const std::vector<Dim> &labels, py::array_t<double> &values,
         units::Unit unit) {
        if (scipp::size(labels) != values.ndim() - 1)
          throw std::runtime_error("bad shape to make vecs");
        std::vector<scipp::index> shape(values.shape(),
                                        values.shape() + labels.size());
        Dimensions dims(labels, shape);
        auto var = variable::make_vectors(
            dims, unit,
            element_array<double>(dims.volume() * 3,
                                  core::default_init_elements));
        auto elems = var.elements<Eigen::Vector3d>();
        copy_array_into_view(values, elems.values<double>(), elems.dims());
        return var;
      },
      py::arg("dims"), py::arg("values"), py::arg("unit") = units::one);
  m.def(
      "matrices",
      [](const Variable &elements) {
        return Variable{};
        // return variable::make_matrices(elements);
      },
      py::arg("elements"));
}
