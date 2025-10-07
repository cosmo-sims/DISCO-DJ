#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <stdint.h>
#include <complex>

#include <gsl/gsl_rng.h>

namespace py = pybind11;

using complex_t = std::complex<double>;

// RNG based on the N-GenIC code
struct rng_ngenic
{
    uint64_t seed_, nres_, nresp_;

    std::vector<unsigned int> SeedTable_;
    gsl_rng *pRandomGenerator_;

public:
    rng_ngenic(uint64_t seed, uint64_t nres)
        : seed_(seed), nres_(nres), nresp_(nres / 2 + 1)
    {
        pRandomGenerator_ = gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng_set(pRandomGenerator_, seed_);

        SeedTable_.assign(nres_ * nres_, 0u);

        for (size_t i = 0; i < nres_ / 2; ++i)
        {
            for (size_t j = 0; j < i; j++)
                SeedTable_[i * nres_ + j] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[j * nres_ + i] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i; j++)
                SeedTable_[(nres_ - 1 - i) * nres_ + j] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[(nres_ - 1 - j) * nres_ + i] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i; j++)
                SeedTable_[i * nres_ + (nres_ - 1 - j)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[j * nres_ + (nres_ - 1 - i)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i; j++)
                SeedTable_[(nres_ - 1 - i) * nres_ + (nres_ - 1 - j)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[(nres_ - 1 - j) * nres_ + (nres_ - 1 - i)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);
        }
    }

    virtual ~rng_ngenic()
    {
        gsl_rng_free(pRandomGenerator_);
    }

    template <typename T, typename i1, typename i2, typename i3>
    inline complex_t &kelem(T &f, i1 i, i2 j, i3 k) const
    {
        return f[(i * nres_ + j) * nresp_ + k];
    }

    py::array_t<complex_t> get_field( void ) const
    {
        auto result = py::array_t<complex_t>(nres_ * nres_ * nresp_);
        std::fill(result.mutable_data(), result.mutable_data() + result.size(), complex_t(0));

        py::buffer_info buf_r = result.request();

        complex_t *ptr = static_cast<complex_t *>(buf_r.ptr);

        // transform is transposed!
        for (size_t i = 0; i < nres_; ++i)
        {
            size_t ii = (i > 0) ? nres_ - i : 0;

            for (size_t j = 0; j < nres_; ++j)
            {
                ptrdiff_t jj = (j > 0) ? nres_ - j : 0;
                gsl_rng_set(pRandomGenerator_, SeedTable_[i * nres_ + j]);

                for (size_t k = 0; k < nres_ / 2 + 1; ++k)
                {
                    double phase = gsl_rng_uniform(pRandomGenerator_) * 2 * M_PI;
                    double ampl = 0;

                    do
                    {
                        ampl = gsl_rng_uniform(pRandomGenerator_);
                    } while (ampl == 0 || ampl == 1);

                    if (i == nres_ / 2 || j == nres_ / 2 || k == nres_ / 2)
                        continue;
                    if (i == 0 && j == 0 && k == 0)
                        continue;

                    ampl = std::sqrt(-std::log(ampl));
                    complex_t zrand(ampl * std::cos(phase), ampl * std::sin(phase));

                    if (k > 0)
                    {
                        kelem(ptr, i, j, k) = zrand;
                    }
                    else
                    { /* k=0 plane needs special treatment */

                        if (i == 0)
                        {
                            if (j < nres_ / 2)
                            {
                                kelem(ptr, i, j, k) = zrand;
                                kelem(ptr, i, jj, k) = std::conj(zrand);
                            }
                        }
                        else if (i < nres_ / 2)
                        {
                            kelem(ptr, i, j, k) = zrand;
                            kelem(ptr, ii, jj, k) = std::conj(zrand);
                        }
                    }
                }
            }
        }

        result.resize({nres_, nres_, nresp_});

        return result;
    }
};

// Make it available to Python
void init_ngenic_grf_3d(py::module &m)
{
    py::class_<rng_ngenic>(m, "rng_ngenic", R"pbdoc(
      Generate N-GenIC random field.

      Parameters
      ----------
      seed : int
          N-GenIC seed
      nres : int
          Resolution
      )pbdoc")
        .def(py::init<uint64_t, uint64_t>(), R"pbdoc(
      Create a new RNG instance.

      Parameters
      ----------
      seed : int
          N-GenIC seed
      nres : int
          Resolution
      )pbdoc")
        .def("get_field", &rng_ngenic::get_field, R"pbdoc(
      Compute the random field.

      Returns
      -------
      numpy.ndarray
          The generated 3D field as a NumPy array.
      )pbdoc");
}
