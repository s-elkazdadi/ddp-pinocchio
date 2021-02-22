#ifndef MATH_HPP_IED2CNIL
#define MATH_HPP_IED2CNIL

#include "mpfr/detail/handle_as_mpfr.hpp"
#include "mpfr/detail/prologue.hpp"

namespace mpfr {

namespace _ {
template <precision_t P> auto pi_impl() -> mp_float_t<P> {
  mp_float_t<P> out;
  {
    mpfr_raii_setter_t&& g = impl_access::mpfr_setter(out);
    mpfr_const_pi(&g.m, MPFR_RNDN);
  }
  return out;
}

} // namespace _

/// \return `true` if the argument is negative, `false` otherwise.\n
/// If the argument is a NaN, detects whether the sign bit is set.
template <precision_t P> auto signbit(mp_float_t<P> const& arg) noexcept -> bool {
  return _::impl_access::actual_prec_sign_const(arg) < 0;
}

/// \return `a` with the sign copied from `b`.
template <typename U, typename V>
sfinae_first_param_type copysign(U const& a, V const& b) noexcept {
  typename _::common_type<U, V>::type out = a;
  auto& prec_sign = _::impl_access::actual_prec_sign_mut(a);
  prec_sign = _::prec_negate_if(prec_sign, mpfr::signbit(a) != mpfr::signbit(b));
  return out;
}

/// Decomposes `arg` into a normalized fraction and a power of two.
/// \return Normalized fraction. The power of two exponent is stored in `*exp`.
template <precision_t P>
auto frexp(mp_float_t<P> const& arg, mpfr_prec_t* exp) noexcept -> mp_float_t<P> {
  mp_float_t<P> out = arg;
  {
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(arg);
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_frexp(exp, &g.m, &x.m, _::get_rnd());
  }
  return out;
}

/// \return `arg` multiplied by two to the power of `exp`
template <precision_t P> auto ldexp(mp_float_t<P> const& arg, long exp) noexcept -> mp_float_t<P> {
  mp_float_t<P> out = arg;
  {
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(arg);
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_mul_2si(&g.m, &x.m, exp, _::get_rnd());
  }
  return out;
}

/// \return `arg` multiplied by two to the power of `exp`
template <precision_t P> auto scalbn(mp_float_t<P> const& arg, long exp) noexcept -> mp_float_t<P> {
  return mpfr::ldexp(arg, exp);
}

/// \return `arg` multiplied by two to the power of `exp`
template <precision_t P>
auto scalbln(mp_float_t<P> const& arg, long exp) noexcept -> mp_float_t<P> {
  return mpfr::ldexp(arg, exp);
}

/// \return The unbiased exponent of the argument. \f$\log_2(\lvert\text{arg}\rvert)\f$
template <precision_t P> auto ilogb(mp_float_t<P> const& arg) noexcept -> mpfr_exp_t {
  _::mpfr_cref_t a = _::impl_access::mpfr_cref(arg);
  return mpfr_custom_get_exp(&a.m) - 1;
}

/// \return The unbiased exponent of the argument. \f$\log_2(\lvert\text{arg}\rvert)\f$
template <precision_t P> auto logb(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mpfr_exp_t i = mpfr::ilogb(arg);
  mp_float_t<P> out;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_set_si(&g.m, i, _::get_rnd());
  }
  return out;
}

/// \return The remainder of `a` divided by `b`, when the quotient is rounded toward zero.
/// If `quotient_ptr` is not null, the least significant bits of the quotient are stored in
/// `*quotient_ptr`.
template <typename U, typename V>
sfinae_common_return_type fmod(U const& a, U const& b, long* quotient_ptr = nullptr) noexcept {
  typename _::common_type<U, V>::type out;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(a);
    _::mpfr_cref_t y = _::impl_access::mpfr_cref(b);
    if (quotient_ptr == nullptr) {
      mpfr_fmod(&g.m, &x.m, &y.m, _::get_rnd());
    } else {
      mpfr_fmodquo(&g.m, quotient_ptr, &x.m, &y.m, _::get_rnd());
    }
  }
  return out;
}

/// \return The fractional part of the argument. The value of the integral part is stored in
/// `*iptr`.
template <precision_t P>
auto modf(mp_float_t<P> const& arg, mp_float_t<P>* iptr) noexcept -> mp_float_t<P> {
  if (iptr == nullptr) {
    mp_float_t<P> i;
    return mpfr::modf(arg, &i);
  }
  mp_float_t<P> out;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_raii_setter_t&& g_i = _::impl_access::mpfr_setter(*iptr);
    _::mpfr_cref_t a = _::impl_access::mpfr_cref(arg);
    mpfr_modf(&g_i.m, &g.m, &a.m, _::get_rnd());
  }
  return out;
}

/// \return The remainder of `a` divided by `b`, when the quotient is rounded to the nearest
/// integer.
/// If `quotient_ptr` is not null, the least significant bits of the quotient are stored in
/// `*quotient_ptr`.
template <typename U, typename V>
sfinae_common_return_type remainder(U const& a, V const& b, long* quotient_ptr = nullptr) noexcept {
  typename _::common_type<U, V>::type out;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(a);
    _::mpfr_cref_t y = _::impl_access::mpfr_cref(b);
    if (quotient_ptr == nullptr) {
      mpfr_remainder(&g.m, &x.m, &y.m, _::get_rnd());
    } else {
      mpfr_remquo(&g.m, quotient_ptr, &x.m, &y.m, _::get_rnd());
    }
  }
  return out;
}

/// \return The remainder of `a` divided by `b`, when the quotient is rounded to the nearest
/// integer.
/// If `quotient_ptr` is not null, the least significant bits of the quotient are stored in
/// `*quotient_ptr`.
template <typename U, typename V>
sfinae_common_return_type remquo(U const& a, U const& b, long* quotient_ptr) noexcept {
  return mpfr::remainder(a, b, quotient_ptr);
}

/// \return Fused multiply add operation. `a * b + c`
template <precision_t P>
auto fma(mp_float_t<P> const& a, mp_float_t<P> const& b, mp_float_t<P> const& c) noexcept
    -> mp_float_t<P> {
  mp_float_t<P> out;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(a);
    _::mpfr_cref_t y = _::impl_access::mpfr_cref(b);
    _::mpfr_cref_t z = _::impl_access::mpfr_cref(c);
    mpfr_fma(&g.m, &x.m, &y.m, &z.m, _::get_rnd());
  }
  return out;
}

/// \return `true` if the argument is zero, `false` otherwise.
template <precision_t P> auto iszero(mp_float_t<P> const& arg) noexcept -> bool {
  _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(arg);
  return mpfr_zero_p(&x_.m) != 0;
}

/// \return `true` if the argument is infinite, `false` otherwise.
template <precision_t P> auto isinf(mp_float_t<P> const& arg) noexcept -> bool {
  _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(arg);
  return mpfr_inf_p(&x_.m) != 0;
}

/// \return `true` if the argument is normal. (always `true`)
template <precision_t P> auto isnormal(mp_float_t<P> const& arg) noexcept -> bool {
  static_cast<void>(arg);
  return true;
}

/// \return `true` if the argument is finite, `false` otherwise.
template <precision_t P> auto isfinite(mp_float_t<P> const& arg) noexcept -> bool {
  _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(arg);
  return mpfr_number_p(&x_.m) != 0;
}

/// \return `true` if the argument is a NaN, `false`, otherwise.
template <precision_t P> auto isnan(mp_float_t<P> const& arg) noexcept -> bool {
  _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(arg);
  return mpfr_nan_p(&x_.m) != 0;
}

/// Category of a floating point number
enum struct fp_class_e { nan, inf, zero, normal };

/// \return Category of the argument.
template <precision_t P> auto fpclassify(mp_float_t<P> const& arg) noexcept -> fp_class_e {
  mpfr_prec_t ps = _::impl_access::actual_prec_sign_const(arg);
  return mpfr::isnan(arg)                   //
             ? fp_class_e::nan              //
             : mpfr::isinf(arg)             //
                   ? fp_class_e::inf        //
                   : (ps == 0 or ps == -1)  //
                         ? fp_class_e::zero //
                         : fp_class_e::normal;
}

/// \return `true` if \f$a > b\f$, `false` otherwise.
template <typename U, typename V> sfinae_bool isgreater(U const& a, U const& b) { return a > b; }
/// \return `true` if \f$a \geq b\f$, `false` otherwise.
template <typename U, typename V> sfinae_bool isgreaterequal(U const& a, V const& b) noexcept {
  return a >= b;
}
/// \return `true` if \f$a < b\f$, `false` otherwise.
template <typename U, typename V> sfinae_bool isless(U const& a, V const& b) noexcept {
  return a < b;
}
/// \return `true` if \f$a \leq b\f$, `false` otherwise.
template <typename U, typename V> sfinae_bool islessequal(U const& a, V const& b) noexcept {
  return a <= b;
}
/// \return `true` if `a` or `b` is NaN, `false` otherwise.
template <typename U, typename V> sfinae_bool isunordered(U const& a, V const& b) noexcept {
  return mpfr::isnan(a) or mpfr::isnan(b);
}
/// \return `true` if \f$a < b\f$ or \f$a > b\f$, `false` otherwise.
template <typename U, typename V> sfinae_bool islessgreater(U const& a, V const& b) noexcept {
  return a != b;
}

/// \return Smaller of the two arguments. If one of the arguments is NaN, returns the other
/// argument.
template <typename U, typename V> sfinae_common_return_type fmin(U const& a, V const& b) noexcept {
  return (mpfr::isnan(b) or a < b) ? a : b;
}

/// \return Larger of the two arguments. If one of the arguments is NaN, returns the other
/// argument.
template <typename U, typename V> sfinae_common_return_type fmax(U const& a, V const& b) noexcept {
  return (mpfr::isnan(b) or a > b) ? a : b;
}

/// \return Positive difference of the two arguments. If one of the arguments is NaN, returns NaN.
template <typename U, typename V> sfinae_common_return_type fdim(U const& a, V const& b) noexcept {
  return (mpfr::isnan(a) or mpfr::isnan(b)) //
             ? std::numeric_limits<typename _::common_type<U, V>::type>::quiet_NaN()
             : mpfr::fmax(a - b, 0);
}

/// \return The absolute value of the argument.
template <precision_t P> auto fabs(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mp_float_t<P> out = arg;
  _::impl_access::actual_prec_sign_mut(out) =
      _::prec_abs(_::impl_access::actual_prec_sign_const(arg));
  return out;
}

/// \return The absolute value of the argument.
template <precision_t P> auto abs(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return mpfr::fabs(arg);
}

/// \return The base to the power of the exponent.
template <typename U, typename V>
sfinae_common_return_type pow(U const& base, V const& exponent) noexcept {
  return _::apply_binary_op(base, exponent, mpfr_pow);
}

/// Pair of the sine and cosine.
template <precision_t P> struct sin_cos_result_t {
  mp_float_t<P> sin;
  mp_float_t<P> cos;
};

/// Pair of the hyperbolic sine and cosine.
template <precision_t P> struct sinh_cosh_result_t {
  mp_float_t<P> sinh;
  mp_float_t<P> cosh;
};

/// \return Sine and cosine of the argument.
template <precision_t P> auto sin_cos(mp_float_t<P> const& arg) noexcept -> sin_cos_result_t<P> {
  _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(arg);
  sin_cos_result_t<P> out;
  {
    _::mpfr_raii_setter_t&& sg = _::impl_access::mpfr_setter(out.sin);
    _::mpfr_raii_setter_t&& cg = _::impl_access::mpfr_setter(out.cos);
    mpfr_sin_cos(&sg.m, &cg.m, &x_.m, _::get_rnd());
  }
  return out;
}

/// \return Hyperbolic sine and cosine of the argument.
template <precision_t P>
auto sinh_cosh(mp_float_t<P> const& arg) noexcept -> sinh_cosh_result_t<P> {
  _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(arg);
  sinh_cosh_result_t<P> out;
  {
    _::mpfr_raii_setter_t&& sg = _::impl_access::mpfr_setter(out.sinh);
    _::mpfr_raii_setter_t&& cg = _::impl_access::mpfr_setter(out.cosh);
    mpfr_sinh_cosh(&sg.m, &cg.m, &x_.m, _::get_rnd());
  }
  return out;
}

/// \return Arc tangent of y/x in the correct quadrant depending on the signs of the arguments.
template <typename U, typename V> sfinae_common_return_type atan2(U const& y, V const& x) noexcept {
  return _::apply_binary_op(y, x, mpfr_atan2);
}

/// \return Square root of the sum of the squares of the arguments.
template <typename U, typename V> sfinae_common_return_type hypot(U const& x, V const& y) noexcept {
  return _::apply_binary_op(x, y, mpfr_hypot);
}

/// \return The next representable number of `from` in the direction of `to`.
template <typename U, typename V>
sfinae_common_return_type nexttoward(U const& from, V const& to) noexcept {
  typename _::common_type<U, V>::type out = from;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_cref_t y_ = _::impl_access::mpfr_cref(to);
    mpfr_nexttoward(&g.m, &y_.m);
  }
  return out;
}

/// \return The next representable number of `from` in the direction of `to`.
template <typename U, typename V>
sfinae_common_return_type nextafter(U const& from, V const& to) noexcept {
  return mpfr::nexttoward(from, to);
}

/// \return The next representable number of `from` in the direction of \f$+\infty\f$.
template <precision_t P> auto nextabove(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mp_float_t<P> out = arg;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_nextabove(&g.m);
  }
  return out;
}

/// \return The next representable number of `from` in the direction of \f$-\infty\f$.
template <precision_t P> auto nextbelow(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mp_float_t<P> out = arg;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_nextbelow(&g.m);
  }
  return out;
}

/// \return Square root of the argument.
template <precision_t P> auto sqrt(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_sqrt);
}

/// \return Cubic root of the argument.
template <precision_t P> auto cbrt(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_cbrt);
}

/// \return Sine of the argument.
template <precision_t P> auto sin(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_sin);
}
/// \return Cosine of the argument.
template <precision_t P> auto cos(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_cos);
}
/// \return Tangent of the argument.
template <precision_t P> auto tan(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_tan);
}
/// \return Inverse of the sine of the argument.
template <precision_t P> auto asin(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_asin);
}
/// \return Inverse of the cosine of the argument.
template <precision_t P> auto acos(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_acos);
}
/// \return Inverse of the tangent of the argument.
template <precision_t P> auto atan(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_atan);
}

/// \return Hyperbolic sine of the argument.
template <precision_t P> auto sinh(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_sinh);
}
/// \return Hyperbolic cosine of the argument.
template <precision_t P> auto cosh(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_cosh);
}
/// \return Hyperbolic tangent of the argument.
template <precision_t P> auto tanh(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_tanh);
}
/// \return Inverse of the hyperbolic sine of the argument.
template <precision_t P> auto asinh(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_asinh);
}
/// \return Inverse of the hyperbolic cosine of the argument.
template <precision_t P> auto acosh(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_acosh);
}
/// \return Inverse of the hyperbolic tangent of the argument.
template <precision_t P> auto atanh(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_atanh);
}

/// \return Exponential of the argument with base \f$e := e^{\text{arg}}\f$.
template <precision_t P> auto exp(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_exp);
}
/// \return Exponential of the argument with base \f$2 := 2^{\text{arg}}\f$.
template <precision_t P> auto exp2(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_exp2);
}
/// \return Exponential of the argument with base \f$10 := 10^{\text{arg}}\f$.
template <precision_t P> auto exp10(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_exp10);
}
/// \return Exponential of the argument with base \f$e\f$ minus 1 \f$:= e^{\text{arg}} - 1\f$.
template <precision_t P> auto expm1(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_expm1);
}
/// \return Logarithm of the argument to base \f$e := \log_e(\text{arg})\f$.
template <precision_t P> auto log(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_log);
}
/// \return Logarithm of the argument to base \f$2 := \log_2(\text{arg})\f$.
template <precision_t P> auto log2(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_log2);
}
/// \return Logarithm of the argument to base \f$10 := \log_10(\text{arg})\f$.
template <precision_t P> auto log10(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_log10);
}
/// \return Logarithm to base \f$e\f$ of \f$1\f$ plus the argument \f$:=\log_e(1 + \text{arg})\f$.
template <precision_t P> auto log1p(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_log1p);
}

/// \return Error function. \f\[\text{erf}(x) = \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}\mathrm{d}t\f\]
template <precision_t P> auto erf(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_erf);
}

/// \return Complementary error function.
/// \f\[\text{erfc}(x) = 1 - \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}\mathrm{d}t\f\]
template <precision_t P> auto erfc(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_erfc);
}

/// \return Gamma function. \f\[\Gamma(x) = \int_0^\infty t^{x-1}e^{-t}\mathrm{d}t\f\]
template <precision_t P> auto tgamma(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_gamma);
}

/// \return Natural log of the absolute value of the gamma function.
template <precision_t P> auto lgamma(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_lgamma);
}

/// \return Beta function. \f\[\text{B}(x, y) = \int_0^1 t^{x-1}(1-t)^{y-1} \mathrm{d}t\f\]
template <typename U, typename V> sfinae_common_return_type beta(U const& x, V const& y) noexcept {
  return _::apply_binary_op(x, y, mpfr_beta);
}

/// \return Exponential integral. \f\[\text{Ei}(x) = \int_{-x}^\infty
/// \frac{e^{-t}}{t}\mathrm{d}t\f\]
template <precision_t P> auto expint(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_eint);
}

/// \return Zeta function.
template <precision_t P> auto riemann_zeta(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_zeta);
}

/// \return Nearby int using the current rounding mode.
template <precision_t P> auto rint(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return _::apply_unary_op(arg, mpfr_rint);
}

/// \return Nearby int using the current rounding mode.
template <precision_t P> auto nearbyint(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  return mpfr::rint(arg);
}

/// \return Next higher or equal representable integer.
template <precision_t P> auto ceil(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mp_float_t<P> out;
  {
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(arg);
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_ceil(&g.m, &x.m);
  }
  return out;
}

/// \return \f$\pi\f$ constant.
template <precision_t P> auto pi_c() noexcept -> mp_float_t<P> const& {
  static mp_float_t<P> const pi = _::pi_impl<P>();
  return pi;
}

/// \return Next lower or equal representable integer.
template <precision_t P> auto floor(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mp_float_t<P> out;
  {
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(arg);
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_floor(&g.m, &x.m);
  }
  return out;
}

/// \return Nearest representable integer, rounding away from zero.
template <precision_t P> auto round(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mp_float_t<P> out;
  {
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(arg);
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_round(&g.m, &x.m);
  }
  return out;
}

/// \return Nearest representable integer, rounding toward zero.
template <precision_t P> auto trunc(mp_float_t<P> const& arg) noexcept -> mp_float_t<P> {
  mp_float_t<P> out;
  {
    _::mpfr_cref_t x = _::impl_access::mpfr_cref(arg);
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    mpfr_trunc(&g.m, &x.m);
  }
  return out;
}

} // namespace mpfr

#include "mpfr/detail/epilogue.hpp"

#endif /* end of include guard MATH_HPP_IED2CNIL */
