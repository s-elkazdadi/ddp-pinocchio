#ifndef MP_FLOAT_HPP_KC35IAEF
#define MP_FLOAT_HPP_KC35IAEF

#include "mpfr/math.hpp"
#include "mpfr/detail/prologue.hpp"

namespace mpfr {

/// Stack allocated fixed precision floating point.\n
/// Arithmetic and comparison operators follow IEEE 754 rules.
template <precision_t Precision> struct mp_float_t {
  static constexpr precision_t precision = Precision;
  static_assert(static_cast<mpfr_prec_t>(Precision) > 0, "precision must be positive.");

  /// Default/Zero initialization sets the number to positive zero.
  ///
  /// `mp_float_t<_> a;`
  ///
  /// `mp_float_t<_> a{};`
  mp_float_t() noexcept { std::memset(this, 0, sizeof(*this)); };

  /// \n
  template <typename T, typename = _::enable_if_t<_::is_arithmetic<T>::value>>
  mp_float_t(T const& a) noexcept : mp_float_t() {
    *this = a;
  }

  /// \n
  template <typename T, typename = _::enable_if_t<_::is_arithmetic<T>::value>>
  auto operator=(T const& a) noexcept -> mp_float_t& {
    _::is_arithmetic<T>::set(
        m_exponent,
        m_actual_prec_sign,
        precision_mpfr,
        m_mantissa,
        sizeof(m_mantissa) / sizeof(m_mantissa[0]),
        a);
    return *this;
  }

  /// \n
  [[MPFR_CXX_NODISCARD]] explicit operator long double() const noexcept {
    _::mpfr_cref_t m = _::impl_access::mpfr_cref(*this);
    return mpfr_get_ld(&m.m, _::get_rnd());
  }
  [[MPFR_CXX_NODISCARD]] explicit operator intmax_t() const noexcept {
    _::mpfr_cref_t m = _::impl_access::mpfr_cref(*this);
    return mpfr_get_sj(&m.m, _::get_rnd());
  }
  [[MPFR_CXX_NODISCARD]] explicit operator uintmax_t() const noexcept {
    _::mpfr_cref_t m = _::impl_access::mpfr_cref(*this);
    return mpfr_get_uj(&m.m, _::get_rnd());
  }
  /** @name Arithmetic operators
   */
  ///@{
  /// \n
  [[MPFR_CXX_NODISCARD]] auto operator+() const noexcept -> mp_float_t { return *this; }
  /// \n
  [[MPFR_CXX_NODISCARD]] auto operator-() const noexcept -> mp_float_t {
    mp_float_t out{*this};
    out.m_actual_prec_sign = _::prec_negate_if(out.m_actual_prec_sign, true);
    return out;
  }
  ///@}

  /// @name Assignment arithmetic operators
  ///
  ///@{
  /// \n
  auto operator+=(mp_float_t const& b) noexcept -> mp_float_t& {
    *this = *this + b;
    return *this;
  }
  /// \n
  auto operator-=(mp_float_t const& b) noexcept -> mp_float_t& {
    *this = *this - b;
    return *this;
  }
  /// \n
  auto operator*=(mp_float_t const& b) noexcept -> mp_float_t& {
    *this = *this * b;
    return *this;
  }
  /// \n
  auto operator/=(mp_float_t const& b) noexcept -> mp_float_t& {
    *this = *this / b;
    return *this;
  }
  ///@}

  /// Write the number to an output stream.
  template <typename CharT, typename Traits>
  friend auto operator<<(std::basic_ostream<CharT, Traits>& out, mp_float_t const& a)
      -> std::basic_ostream<CharT, Traits>& {
    constexpr std::size_t stack_bufsize =
        _::digits2_to_10(static_cast<std::size_t>(precision_mpfr) + 64);
    char stack_buffer[stack_bufsize];
    _::write_to_ostream(out, _::impl_access::mpfr_cref(a), stack_buffer, stack_bufsize);
    return out;
  }

private:
  friend struct _::impl_access;

  static constexpr mpfr_prec_t precision_mpfr = static_cast<mpfr_prec_t>(Precision);

  mp_limb_t m_mantissa[_::prec_to_nlimb(static_cast<std::uint64_t>(Precision))]{};
  mpfr_exp_t m_exponent{};
  mpfr_exp_t m_actual_prec_sign{};
}; // namespace mpfr

/// \n
template <typename U, typename V>
sfinae_common_return_type operator+(U const& a, V const& b) noexcept {
  return _::arithmetic_op(a, b, _::set_add);
}
/// \n
template <typename U, typename V>
sfinae_common_return_type operator-(U const& a, V const& b) noexcept {
  return _::arithmetic_op(a, b, _::set_sub);
}
/// \n

template <typename U, typename V>
sfinae_common_return_type operator*(U const& a, V const& b) noexcept {

  typename _::into_mp_float_lossless<U>::type const& a_{a};
  typename _::into_mp_float_lossless<V>::type const& b_{b};

  if ((a == 0 and mpfr::isfinite(b_)) or (b == 0 and mpfr::isfinite(a_))) {
    return 0;
  }

  bool const b_is_pow2 = _::prec_abs(_::impl_access::actual_prec_sign_const(b_)) == 1;
  bool const a_is_pow2 = _::prec_abs(_::impl_access::actual_prec_sign_const(a_)) == 1;

  if (b_is_pow2 and
      _::mul_b_is_pow2_check(_::impl_access::exp_const(a_), _::impl_access::exp_const(b_), false)) {
    auto out = static_cast<typename _::common_type<U, V>::type>(a);
    _::mul_b_is_pow2(
        &_::impl_access::exp_mut(out),
        &_::impl_access::actual_prec_sign_mut(out),
        _::impl_access::exp_const(b_),
        _::impl_access::actual_prec_sign_const(b_),
        false);
    return out;
  }

  if (a_is_pow2 and
      _::mul_b_is_pow2_check(_::impl_access::exp_const(b_), _::impl_access::exp_const(a_), false)) {
    auto out = static_cast<typename _::common_type<U, V>::type>(b);
    _::mul_b_is_pow2(
        &_::impl_access::exp_mut(out),
        &_::impl_access::actual_prec_sign_mut(out),
        _::impl_access::exp_const(a_),
        _::impl_access::actual_prec_sign_const(a_),
        false);
    return out;
  }

  return _::arithmetic_op(a_, b_, _::set_mul);
}

/// \n
template <typename U, typename V>
sfinae_common_return_type operator/(U const& a, V const& b) noexcept {

  typename _::into_mp_float_lossless<V>::type const& a_{a};
  typename _::into_mp_float_lossless<V>::type const& b_{b};

  if (mpfr::iszero(a_) and mpfr::isfinite(b_) and not mpfr::iszero(b_)) {
    return 0;
  }

  if (_::prec_abs(_::impl_access::actual_prec_sign_const(b_)) == 1 and
      _::mul_b_is_pow2_check(_::impl_access::exp_const(a_), _::impl_access::exp_const(b_), true)) {
    typename _::common_type<U, V>::type out{a};
    _::mul_b_is_pow2( //
        &_::impl_access::exp_mut(out),
        &_::impl_access::actual_prec_sign_mut(out),
        _::impl_access::exp_const(b_),
        _::impl_access::actual_prec_sign_const(b_),
        true);
  }
  return _::arithmetic_op(a, b_, _::set_div);
}

/// \n
template <typename U, typename V> sfinae_bool operator==(U const& a, V const& b) noexcept {
  return _::comparison_op(a, b, mpfr_equal_p);
}
/// \n
template <typename U, typename V> sfinae_bool operator!=(U const& a, V const& b) noexcept {
  return _::comparison_op(a, b, mpfr_lessgreater_p);
}
/// \n
template <typename U, typename V> sfinae_bool operator<(U const& a, V const& b) noexcept {
  return _::comparison_op(a, b, mpfr_less_p);
}
/// \n
template <typename U, typename V> sfinae_bool operator<=(U const& a, V const& b) noexcept {
  return _::comparison_op(a, b, mpfr_lessequal_p);
}
/// \n
template <typename U, typename V> sfinae_bool operator>(U const& a, V const& b) noexcept {
  return _::comparison_op(a, b, mpfr_greater_p);
}
/// \n
template <typename U, typename V> sfinae_bool operator>=(U const& a, V const& b) noexcept {
  return _::comparison_op(a, b, mpfr_greaterequal_p);
}

/// Allows handling `mp_float_t<_>` objects through `mpfr_ptr`/`mpfr_srcptr` proxy objects.
/// If after the callable is executed, one of the arguments has been modified by mpfr, the
/// corresponding `mp_float_t<_>` object is set to the equivalent value.
///
/// If any of the following occurs, the behavior is undefined:
/// * Any of the arguments are passed to allocating MPFR functions.
/// * Any two arguments alias, and one of them is accessed after the other has been modified.
/// * A `mpfr_srcptr` argument is `const_cast` to a `mpfr_ptr`.
/// * The `mp_float_t<_>` object referenced by a parameter that is modified is accessed
/// while the callable is being run (except through the `mpfr_ptr` proxy).
///
/// \par Side effects
/// Arguments are read before the callable is executed regardless of whether the
/// corresponding `mpfr_srcptr`/`mpfr_ptr` is accessed by the callable
//
/// \return The return value of the callable.
///
/// @param[in] fn   Arguments of type `mp_float_t<_>`.
/// @param[in] fn   A callable that takes arguments of type `mpfr_ptr` for mutable
/// parameters, or `mpfr_srcptr` for immutable parameters.
template <typename Fn, typename... Args>
sfinae_callable_return_type
handle_as_mpfr_t(Fn&& fn, Args&&... args) noexcept(callable_is_noexcept) {
  return _::impl_handle_as_mpfr_t<callable_is_noexcept>(static_cast<Fn&&>(fn), args...);
}

} // namespace mpfr

#if defined(min) or defined(max)
HEDLEY_WARNING("min/max macros definitions are undone.")
#undef min
#undef max
#endif
namespace std {
/// Specialization of the standard library numeric limits
template <mpfr::precision_t Precision> struct numeric_limits<mpfr::mp_float_t<Precision>> {
  using T = mpfr::mp_float_t<Precision>;
  static constexpr bool is_specialized = true;

  /// Largest finite number.
  static auto max() noexcept -> T {
    T out = one_m_eps_impl();
    mpfr::_::impl_access::exp_mut(out) = mpfr_get_emax();
    return out;
  }
  /// Smallest strictly positive number.
  static auto min() noexcept -> T {
    T out{0.5F};
    mpfr::_::impl_access::exp_mut(out) = mpfr_get_emin();
    return out;
  }
  /// Smallest finite number.
  static auto lowest() noexcept -> T { return -(max)(); }

  static constexpr int digits = static_cast<int>(Precision);
  static constexpr int digits10 =
      static_cast<int>(::mpfr::_::digits2_to_10(static_cast<uint64_t>(Precision)) + 1);
  static constexpr int max_digits10 = digits10;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = true;
  static constexpr int radix = 2;
  /// Distance between 1 and the smallest number larger than 1.
  static auto epsilon() noexcept -> T {
    static auto eps = epsilon_impl();
    return eps;
  }
  /// Largest possible error in ULP.
  static constexpr auto round_error() noexcept -> T {
    return T{mpfr::_::get_rnd() == MPFR_RNDN ? 0.5 : 1.0};
  }

  static constexpr int min_exponent = (MPFR_EMIN_DEFAULT >= INT_MIN) ? MPFR_EMIN_DEFAULT : INT_MIN;
  static constexpr int min_exponent10 =
      -static_cast<int>(::mpfr::_::digits2_to_10(static_cast<uint64_t>(-min_exponent)) + 1);
  static constexpr int max_exponent = (MPFR_EMAX_DEFAULT <= INT_MAX) ? MPFR_EMAX_DEFAULT : INT_MAX;
  static constexpr int max_exponent10 =
      static_cast<int>(::mpfr::_::digits2_to_10(static_cast<uint64_t>(max_exponent)) + 1);

  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_absent;
  static constexpr bool has_denorm_loss = false;
  /// Positive infinity.
  static auto infinity() noexcept -> T { return T{1.0L} / T{0.0L}; }
  /// Not a number.
  static auto quiet_NaN() noexcept -> T { return T{0.0L} / T{0.0L}; }
  /// Not a number.
  static constexpr auto signaling_NaN() noexcept -> T { return quiet_NaN(); }
  static constexpr auto denorm_min() noexcept -> T { return T{}; }

  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;

  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;
  static constexpr float_round_style round_style = round_toward_zero;

private:
  static auto epsilon_impl() noexcept -> T {
    T x{1};
    {
      mpfr::_::mpfr_raii_setter_t&& g = mpfr::_::impl_access::mpfr_setter(x);
      mpfr_nextabove(&g.m);
      mpfr_sub_ui(&g.m, &g.m, 1, mpfr::_::get_rnd());
    }
    return x;
  }
  static auto one_m_eps_impl() noexcept -> T {
    T x{1};
    {
      mpfr::_::mpfr_raii_setter_t&& g = mpfr::_::impl_access::mpfr_setter(x);
      mpfr_nextbelow(&g.m);
    }
    return x;
  }
};
} // namespace std
#include "mpfr/detail/epilogue.hpp"

#endif /* end of include guard MP_FLOAT_HPP_KC35IAEF */
