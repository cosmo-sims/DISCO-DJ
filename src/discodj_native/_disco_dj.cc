
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_ngenic_grf_3d(py::module &);
void init_nbody2d(py::module &);
void init_nbody3d(py::module &);


PYBIND11_MODULE(_discodj_native, m)
{
    m.doc() = "DiSCO-DJ module with low-level C++ routines for generating NGENIC ICs and TreePM";

    init_ngenic_grf_3d(m);
    init_nbody2d(m);
    init_nbody3d(m);
}
