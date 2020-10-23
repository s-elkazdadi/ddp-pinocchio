#include "ddp/pinocchio_model.ipp"

#include "mpfr/mpfr.hpp"

namespace pinocchio {
template <mpfr::precision_t P>
struct is_floating_point<mpfr::mp_float_t<P>> : boost::integral_constant<bool, true> {};

template <mpfr::precision_t P>
struct SINCOSAlgo<mpfr::mp_float_t<P>, mpfr::mp_float_t<P>, mpfr::mp_float_t<P>> {
  static void run(mpfr::mp_float_t<P> const& a, mpfr::mp_float_t<P>* sa, mpfr::mp_float_t<P>* ca) {
    mpfr::sin_cos_result_t<P> res = mpfr::sin_cos(a);
    *sa = res.sin;
    *ca = res.cos;
  }
};
} // namespace pinocchio

template struct ddp::pinocchio::model_t<mpfr::mp_float_t<mpfr::digits10{1000}>>;
