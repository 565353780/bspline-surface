#include "value.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(bsp_cpp, m) {
  m.doc() = "pybind11 bspline surface fitting plugin";

  m.def("toSpans", &toSpans, "value.toSpans");
  m.def("toBasisFunctions", &toBasisFunctions, "value.toBasisFunctions");
  m.def("toEvalPointsOld", &toEvalPointsOld, "value.toEvalPointsOld");
  m.def("toEvalPoints", &toEvalPoints, "value.toEvalPoints");
  m.def("toTorchPoints", &toTorchPoints, "value.toTorchPoints");
  m.def("toUVTorchPoints", &toUVTorchPoints, "value.toUVTorchPoints");
}
