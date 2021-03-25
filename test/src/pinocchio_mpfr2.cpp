#include <pinocchio/math/multiprecision-mpfr.hpp>
#include <ddp/pinocchio_model.ipp>
#include <mpfr/mpfr.hpp>

using scalar = mpfr::mp_float_t<mpfr::digits10(500)>;

namespace pinocchio {
// We check std::numeric_limits<_>::has_infinity to exclude integral, rational
// and complex types
template <mpfr::precision_t P>
struct is_floating_point<mpfr::mp_float_t<P>> : boost::true_type {};

template <mpfr::precision_t P>
struct SINCOSAlgo<
		mpfr::mp_float_t<P>,
		mpfr::mp_float_t<P>,
		mpfr::mp_float_t<P>> {
	static void
	run(mpfr::mp_float_t<P> const& a,
	    mpfr::mp_float_t<P>* sa,
	    mpfr::mp_float_t<P>* ca) noexcept {
		auto res = mpfr::sin_cos(a);
		*sa = res.sin;
		*ca = res.cos;
	}
};
} // namespace pinocchio

template struct ddp::pinocchio::model<scalar>;
