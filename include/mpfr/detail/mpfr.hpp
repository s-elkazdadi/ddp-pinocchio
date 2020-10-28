#ifndef MPFR_HPP_NZTOL31N
#define MPFR_HPP_NZTOL31N

#include "mpfr/enums.hpp"
#include "mpfr/detail/prologue.hpp"

#include <exception>
#include <limits>
#include <iosfwd>
#include <initializer_list>

#if MPFR_CXX_HAS_MATH_BUILTINS == 0
// for std::{fabs,frexp,signbit}
#include <cmath>
#endif

namespace mpfr {

template <precision_t> struct mp_float_t;

namespace _ {

[[noreturn]] inline void crash_with_message(char const* message) {
  ::fprintf(stderr, "%s\n", message);
  std::terminate();
}

using std::size_t;
using std::uint64_t;

constexpr auto all_of_impl(bool const conds[], size_t n) -> bool {
  return n == 0 ? true : (*conds and all_of_impl(conds + 1, n - 1));
}

constexpr auto all_of(std::initializer_list<bool> conds) -> bool {
  return all_of_impl(conds.begin(), conds.size());
}

constexpr auto round_up_to_multiple(uint64_t n, uint64_t k) -> uint64_t {
  return (n + k - 1) / k * k;
}

constexpr auto digits10_to_2(uint64_t n) -> uint64_t {
  return 3 * n +
         static_cast<uint64_t>(
             0.321928094887362347870319429489390175864831393024580612054756395L *
             static_cast<long double>(n)) +
         1;
}

constexpr auto midpoint(uint64_t a, uint64_t b) -> uint64_t {
  return (a / 2 + b / 2 + ((a % 2) * (b % 2)));
}

constexpr auto cmp_inverse(uint64_t estimate, uint64_t target, uint64_t (*func)(uint64_t)) -> int {
  return (func(estimate) >= target and func(estimate - 1) < target)
             ? 0
             : (func(estimate) > target ? 1 : -1);
}

constexpr auto
inverse_binary_search(uint64_t n, uint64_t low, uint64_t high, uint64_t (*func)(uint64_t))
    -> uint64_t {
  return cmp_inverse(midpoint(low, high), n, func) == 0 ? midpoint(low, high)
         : cmp_inverse(midpoint(low, high), n, func) == 1
             ? inverse_binary_search(n, low, midpoint(low, high) - 1, func)
             : inverse_binary_search(n, midpoint(low, high) + 1, high, func);
}

/// func must be strictly increasing
/// returns smallest m such that func(m) >= n
constexpr auto inverse(uint64_t n, uint64_t (*func)(uint64_t)) -> uint64_t {
  return n == 0 ? 0 : inverse_binary_search(n, 0, n, func);
}

constexpr auto _sqr(uint64_t n) -> uint64_t { return n * n; }
constexpr auto sqrt(uint64_t n) -> uint64_t { return inverse(n, _sqr); }
constexpr auto digits2_to_10(uint64_t n) -> uint64_t { return inverse(n, digits10_to_2); }

constexpr auto prec_to_nlimb(mpfr_prec_t prec) -> uint64_t {
  return round_up_to_multiple(
             static_cast<uint64_t>(mpfr_custom_get_size(static_cast<uint64_t>(prec))),
             sizeof(mp_limb_t)) /
         sizeof(mp_limb_t);
}

template <typename T> struct remove_pointer;
template <typename T> struct remove_pointer<T*> { using type = T; };

static constexpr uint64_t limb_pack_size = 32 / sizeof(mp_limb_t);
static constexpr mp_limb_t zero_pack[limb_pack_size] = {};
static constexpr mp_limb_t pow2_mantissa_last = mp_limb_t{1}
                                                << mp_limb_t{sizeof(mp_limb_t) * CHAR_BIT - 1};

inline auto is_zero_pack(mp_limb_t const* ptr) -> bool {
  return std::memcmp(ptr, zero_pack, sizeof(zero_pack)) == 0;
}

inline constexpr auto prec_negate_if(mpfr_prec_t p, bool cond) -> mpfr_prec_t {
  return cond ? static_cast<mpfr_prec_t>(~static_cast<mpfr_uprec_t>(p)) : p;
}

inline constexpr auto prec_abs(mpfr_prec_t p) -> mpfr_prec_t { return prec_negate_if(p, p < 0); }

#if HEDLEY_HAS_BUILTIN(__builtin_clz)
constexpr auto count_leading_zeros(unsigned char c) -> int {
  return c == 0 ? CHAR_BIT : __builtin_clz(c) - int((sizeof(unsigned) - 1) * CHAR_BIT);
}
#else
constexpr auto count_leading_zeros2_impl(unsigned char c, unsigned int idx) -> int {
  return idx == CHAR_BIT //
             ? 0
             : ((c < (1U << idx)) //
                    ? static_cast<int>(CHAR_BIT - idx)
                    : leading_zeros2_impl(c, idx + 1));
}
constexpr auto count_leading_zeros(unsigned char c) -> int {
  return c == 0 ? CHAR_BIT : leading_zeros2_impl(c, 0);
}
#endif

inline auto count_trailing_zeros(unsigned long long x) -> int {
#if HEDLEY_HAS_BUILTIN(__builtin_ctzll)
  int zero_bits = __builtin_ctzll(x);
#else
  int zero_bits = 0;
  while ((x % 2) == 0) {
    x /= 2;
    ++zero_bits;
  }
#endif
  return zero_bits;
}

inline auto compute_actual_prec(mpfr_srcptr x) -> mpfr_prec_t {

  if (mpfr_custom_get_kind(x) != MPFR_REGULAR_KIND and
      mpfr_custom_get_kind(x) != -MPFR_REGULAR_KIND) {
    return 0;
  }

  auto const* xp = static_cast<mp_limb_t const*>(mpfr_custom_get_mantissa(x));

  // size of mantissa minus last block
  size_t size = prec_to_nlimb(mpfr_get_prec(x)) - 1;

  // count leading mantissa zeros
  mpfr_prec_t zero_bits = 0;

  size_t head = size / limb_pack_size * limb_pack_size;
  size_t end = size;

  constexpr mpfr_prec_t bits_limb = sizeof(mp_limb_t) * CHAR_BIT;
  size_t i = 0;
  for (; i < head; i += limb_pack_size) {
    if (is_zero_pack(xp + i)) {
      zero_bits += static_cast<mpfr_prec_t>(limb_pack_size) * bits_limb;
    } else {
      end = i + limb_pack_size;
      break;
    }
  }

  for (; i < end; ++i) {
    if (xp[i] == 0) {
      zero_bits += bits_limb;
    } else {
      break;
    }
  }

  // count trailing zeros
  mp_limb_t last_limb = xp[i];
  if (last_limb == 0) {
    zero_bits += bits_limb;
  } else {
    zero_bits += count_trailing_zeros(last_limb);
  }
  return static_cast<mpfr_prec_t>(
             round_up_to_multiple(static_cast<uint64_t>(mpfr_get_prec(x)), bits_limb)) //
         - zero_bits;
}

template <bool Cond, typename T> struct enable_if { using type = T; };
template <typename T> struct enable_if<false, T> {};
template <bool Cond, typename T = void>
using enable_if_t = typename mpfr::_::enable_if<Cond, T>::type;

struct mpfr_cref_t {
  typename remove_pointer<mpfr_ptr>::type m;
};

struct mpfr_raii_setter_t /* NOLINT */ {
  typename remove_pointer<mpfr_ptr>::type m;
  mpfr_prec_t m_actual_precision;
  mpfr_prec_t m_old_actual_prec_sign;
  mpfr_exp_t m_old_exponent;
  mpfr_exp_t* m_exponent_ptr{};
  mpfr_prec_t* m_actual_prec_sign_ptr{};

  mpfr_raii_setter_t(
      mpfr_prec_t precision,
      mp_limb_t* mantissa,
      mpfr_exp_t* exponent_ptr,
      mpfr_prec_t* actual_prec_sign_ptr)
      : m{
        precision,
        (*actual_prec_sign_ptr) < 0 ? -1 : 1,
        (*exponent_ptr),
        mantissa,
      },
        m_actual_precision{precision},
        m_old_actual_prec_sign{*actual_prec_sign_ptr},
        m_old_exponent{*exponent_ptr},
        m_exponent_ptr{exponent_ptr},
        m_actual_prec_sign_ptr{actual_prec_sign_ptr} {

    mpfr_sign_t sign = (m_old_actual_prec_sign < 0) ? -1 : 1;
    mpfr_prec_t actual_prec = prec_abs(m_old_actual_prec_sign);

    if (actual_prec == 0 and m_old_exponent == 0) {
      mpfr_custom_init_set(&m, sign * MPFR_ZERO_KIND, 0, precision, mantissa);
    }
  }

  mpfr_raii_setter_t(mpfr_raii_setter_t const&) = delete;
  mpfr_raii_setter_t(mpfr_raii_setter_t&&) = delete;
  auto operator=(mpfr_raii_setter_t const&) -> mpfr_raii_setter_t& = delete;
  auto operator=(mpfr_raii_setter_t&&) -> mpfr_raii_setter_t& = delete;

  ~mpfr_raii_setter_t() {

    // if precision of m is equal to actual_precision, compute actual_precision
    // otherwise, set actual_precision_ptr's value to actual_precision
    mpfr_prec_t actual_prec_sign = prec_negate_if(
        mpfr_regular_p(&m)                               //
            ? ((m_actual_precision == mpfr_get_prec(&m)) //
                   ? compute_actual_prec(&m)
                   : m_actual_precision)
            : 0,
        mpfr_signbit(&m));

    if (m_old_actual_prec_sign != actual_prec_sign) {
      *m_actual_prec_sign_ptr = actual_prec_sign;
    }

    mpfr_exp_t exp = mpfr_zero_p(&m) ? 0 : mpfr_get_exp(&m);
    if (m_old_exponent != exp) {
      *m_exponent_ptr = exp;
    }
  }
};

struct impl_access {

  template <precision_t P>
  static auto mantissa_mut(mp_float_t<P>& x)
      -> mp_limb_t (&)[prec_to_nlimb(static_cast<std::uint64_t>(P))] {
    return x.m_mantissa;
  }
  template <precision_t P>
  static auto mantissa_const(mp_float_t<P> const& x) -> mp_limb_t
      const (&)[prec_to_nlimb(static_cast<std::uint64_t>(P))] {
    return x.m_mantissa;
  }

  template <precision_t P> static auto actual_prec_sign_mut(mp_float_t<P>& x) -> mpfr_prec_t& {
    return x.m_actual_prec_sign;
  }
  template <precision_t P>
  static auto actual_prec_sign_const(mp_float_t<P> const& x) -> mpfr_prec_t {
    return x.m_actual_prec_sign;
  }

  template <precision_t P> static auto exp_mut(mp_float_t<P>& x) -> mpfr_exp_t& {
    return x.m_exponent;
  }
  template <precision_t P> static auto exp_const(mp_float_t<P> const& x) -> mpfr_exp_t {
    return x.m_exponent;
  }

  template <precision_t P> static auto mpfr_cref(mp_float_t<P> const& x) -> mpfr_cref_t {
    mpfr_cref_t out{};
    mpfr_sign_t sign = (x.m_actual_prec_sign < 0) ? -1 : 1;

    mpfr_prec_t actual_prec = prec_abs(x.m_actual_prec_sign);

    if (actual_prec == 0 and x.m_exponent == 0) {
      mpfr_custom_init_set(&out.m, sign * MPFR_ZERO_KIND, x.m_exponent, 1, x.m_mantissa);
      return out;
    }
    constexpr size_t full_n_limb = prec_to_nlimb(mp_float_t<P>::precision_mpfr);
    size_t actual_n_limb = prec_to_nlimb(actual_prec);
    out.m = {
        actual_prec,
        sign,
        x.m_exponent,
        const_cast<mp_limb_t*> // NOLINT(cppcoreguidelines-pro-type-const-cast)
        (x.m_mantissa + (full_n_limb - actual_n_limb)),
    };
    return out;
  }

  template <precision_t P> static auto mpfr_setter(mp_float_t<P>& x) -> mpfr_raii_setter_t {
    return {
        mp_float_t<P>::precision_mpfr,
        static_cast<mp_limb_t*>(x.m_mantissa),
        &x.m_exponent,
        &x.m_actual_prec_sign,
    };
  }
};

HEDLEY_ALWAYS_INLINE auto mul_b_is_pow2_check(mpfr_exp_t a_exp, mpfr_exp_t b_exponent, bool div)
    -> bool {
  typename _::remove_pointer<mpfr_ptr>::type ea_{0, 0, a_exp, nullptr};

  mpfr_prec_t eb{div ? (1 - b_exponent) : b_exponent - 1};

  return mpfr_regular_p(&ea_) and           //
         (a_exp + eb) < mpfr_get_emax() and //
         (a_exp + eb) >= mpfr_get_emin();
}

HEDLEY_ALWAYS_INLINE void mul_b_is_pow2(
    mpfr_exp_t* a_exp,
    mpfr_exp_t* a_prec_sign,
    mpfr_exp_t b_exponent,
    mpfr_exp_t b_actual_prec_sign,
    bool div) {
  mpfr_prec_t eb{div ? (1 - b_exponent) : b_exponent - 1};

  *a_prec_sign = prec_negate_if(*a_prec_sign, b_actual_prec_sign < 0);
  *a_exp += eb;
}

inline auto get_rnd() -> mpfr_rnd_t {
  auto rnd = std::fegetround();
  switch (rnd) {
  case FE_TONEAREST:
    return MPFR_RNDN;
  case FE_TOWARDZERO:
    return MPFR_RNDZ;
  case FE_DOWNWARD:
    return MPFR_RNDD;
  case FE_UPWARD:
    return MPFR_RNDU;
  default:
    return MPFR_RNDN;
  }
}

template <typename T1, typename T2> struct common_type;
template <precision_t P, typename T1> struct common_type<T1, mp_float_t<P>> {
  using type = mp_float_t<P>;
};
template <precision_t P, typename T2> struct common_type<mp_float_t<P>, T2> {
  using type = mp_float_t<P>;
};
template <precision_t P1, precision_t P2> struct common_type<mp_float_t<P1>, mp_float_t<P2>> {
  using type = mp_float_t<(P1 > P2) ? P1 : P2>;
};

template <typename T> struct into_mp_float_lossless {
  using type = mp_float_t<digits2{sizeof(T) * CHAR_BIT}>;
};
template <precision_t P> struct into_mp_float_lossless<mp_float_t<P>> {
  using type = mp_float_t<P>;
};

template <typename T1, typename T2> using common_type_t = typename common_type<T1, T2>::type;

template <typename T> struct is_arithmetic { static constexpr bool value = false; };

template <typename T> void set_primitive(mpfr_raii_setter_t& out, T a) {
  is_arithmetic<T>::fnptr(&out.m, a, _::get_rnd());

  if (mpfr_get_prec(&out.m) >= static_cast<mpfr_prec_t>(sizeof(T) * CHAR_BIT)) {
    typename remove_pointer<mpfr_ptr>::type p = out.m;

    size_t full_n_limb = prec_to_nlimb(mpfr_get_prec(&out.m));
    size_t actual_n_limb = prec_to_nlimb(sizeof(T) * CHAR_BIT);

    mpfr_custom_move(
        &p,
        (static_cast<mp_limb_t*>(mpfr_custom_get_significand(&p)) + (full_n_limb - actual_n_limb)));
    mpfr_get_prec(&p) = sizeof(T) * CHAR_BIT;

    out.m_actual_precision = compute_actual_prec(&p);
  }
}

template <precision_t P> struct is_arithmetic<mp_float_t<P>> {
  static constexpr bool value = true;
  static constexpr auto* fnptr = mpfr_set_sj;
  static HEDLEY_ALWAYS_INLINE void
  set(mpfr_exp_t& m_exponent,
      mpfr_prec_t& m_actual_prec_sign,
      mpfr_prec_t precision_mpfr,
      mp_limb_t* m_mantissa,
      size_t size,
      mp_float_t<P> const& a) {

    static_cast<void>(size);

    mpfr_raii_setter_t g{
        precision_mpfr,
        m_mantissa,
        &m_exponent,
        &m_actual_prec_sign,
    };
    mpfr_cref_t x = impl_access::mpfr_cref(a);
    mpfr_set(&g.m, &x.m, _::get_rnd());
  }
};

template <> struct is_arithmetic<signed long long> {
  static constexpr bool value = true;
  static constexpr auto* fnptr = mpfr_set_sj;

  static HEDLEY_ALWAYS_INLINE void
  set(mpfr_exp_t& m_exponent,
      mpfr_prec_t& m_actual_prec_sign,
      mpfr_prec_t precision_mpfr,
      mp_limb_t* m_mantissa,
      size_t size,
      signed long long a) {

    if (a == 0) {
      m_exponent = 0;
      m_actual_prec_sign = 0;
      std::memset(m_mantissa, 0, sizeof(mp_limb_t) * size);
      return;
    }
    bool signbit = a < 0;
    auto b = static_cast<unsigned long long>(a);
    if (signbit) {
      b = -b;
    }
    bool pow_of_2 = (b & (b - 1)) == 0;
    int exponent = count_trailing_zeros(b) + 1;

    if (pow_of_2) {
      // a is a power of two
      // a = signbit * 2^(exp-1)

      auto* xp = static_cast<mp_limb_t*>(m_mantissa);

      if (size >= 1) {
        std::memset(xp, 0, sizeof(mp_limb_t) * (size - 1));
      }
      m_exponent = exponent;
      xp[size - 1] = _::pow2_mantissa_last;
      m_actual_prec_sign = prec_negate_if(1, signbit);
    } else {
      _::mpfr_raii_setter_t g{
          precision_mpfr,
          m_mantissa,
          &m_exponent,
          &m_actual_prec_sign,
      };
      _::set_primitive(g, a);
    }
  }
};

template <> struct is_arithmetic<unsigned long long> {
  static constexpr bool value = true;
  static constexpr auto* fnptr = mpfr_set_uj;

  static HEDLEY_ALWAYS_INLINE void
  set(mpfr_exp_t& m_exponent,
      mpfr_prec_t& m_actual_prec_sign,
      mpfr_prec_t precision_mpfr,
      mp_limb_t* m_mantissa,
      size_t size,
      unsigned long long a) {

    if (a == 0) {
      m_exponent = 0;
      m_actual_prec_sign = 0;
      std::memset(m_mantissa, 0, sizeof(mp_limb_t) * size);
      return;
    }
    bool pow_of_2 = (a & (a - 1)) == 0;
    int exponent = count_trailing_zeros(a) + 1;

    if (pow_of_2) {
      // a is a power of two
      // a = 2^(exp-1)

      auto* xp = static_cast<mp_limb_t*>(m_mantissa);

      if (size >= 1) {
        std::memset(xp, 0, sizeof(mp_limb_t) * (size - 1));
      }
      m_exponent = exponent;
      xp[size - 1] = _::pow2_mantissa_last;
      m_actual_prec_sign = 1;

    } else {
      _::mpfr_raii_setter_t g{
          precision_mpfr,
          m_mantissa,
          &m_exponent,
          &m_actual_prec_sign,
      };
      _::set_primitive(g, a);
    }
  };
};

template <> struct is_arithmetic<signed long> : is_arithmetic<signed long long> {};
template <> struct is_arithmetic<signed int> : is_arithmetic<signed long long> {};
template <> struct is_arithmetic<unsigned long> : is_arithmetic<unsigned long long> {};
template <> struct is_arithmetic<unsigned int> : is_arithmetic<unsigned long long> {};

template <> struct is_arithmetic<float> {
  static constexpr bool value = true;
  static constexpr auto* fnptr = mpfr_set_d;

  static HEDLEY_ALWAYS_INLINE void
  set(mpfr_exp_t& m_exponent,
      mpfr_prec_t& m_actual_prec_sign,
      mpfr_prec_t precision_mpfr,
      mp_limb_t* m_mantissa,
      size_t size,
      float a) {
    int exponent{};

    float normalized = MPFR_CXX_FABSF(MPFR_CXX_FREXPF(a, &exponent));
    if (normalized == 0.5F) {
      bool signbit = MPFR_CXX_SIGNBIT(a) != 0;

      // a is a power of two
      // a = signbit * 2^(exp-1)

      if (size >= 1) {
        std::memset(m_mantissa, 0, sizeof(mp_limb_t) * (size - 1));
      }
      m_exponent = exponent;
      m_mantissa[size - 1] = _::pow2_mantissa_last;
      m_actual_prec_sign = prec_negate_if(1, signbit);
    } else {
      _::mpfr_raii_setter_t g{
          precision_mpfr,
          m_mantissa,
          &m_exponent,
          &m_actual_prec_sign,
      };
      _::set_primitive(g, static_cast<double>(a));
    }
  }
};

template <> struct is_arithmetic<double> {
  static constexpr bool value = true;
  static constexpr auto* fnptr = mpfr_set_d;

  static HEDLEY_ALWAYS_INLINE void
  set(mpfr_exp_t& m_exponent,
      mpfr_prec_t& m_actual_prec_sign,
      mpfr_prec_t precision_mpfr,
      mp_limb_t* m_mantissa,
      size_t size,
      double a) {
    int exponent{};

    double normalized = MPFR_CXX_FABS(MPFR_CXX_FREXP(a, &exponent));
    if (normalized == 0.5) {
      bool signbit = MPFR_CXX_SIGNBIT(a) != 0;

      // a is a power of two
      // a = signbit * 2^(exp-1)

      if (size >= 1) {
        std::memset(m_mantissa, 0, sizeof(mp_limb_t) * (size - 1));
      }
      m_exponent = exponent;
      m_mantissa[size - 1] = _::pow2_mantissa_last;
      m_actual_prec_sign = prec_negate_if(1, signbit);
    } else {
      _::mpfr_raii_setter_t g{
          precision_mpfr,
          m_mantissa,
          &m_exponent,
          &m_actual_prec_sign,
      };
      _::set_primitive(g, a);
    }
  }
};

template <> struct is_arithmetic<long double> {
  static constexpr bool value = true;
  static constexpr auto* fnptr = mpfr_set_ld;
  static HEDLEY_ALWAYS_INLINE void
  set(mpfr_exp_t& m_exponent,
      mpfr_prec_t& m_actual_prec_sign,
      mpfr_prec_t precision_mpfr,
      mp_limb_t* m_mantissa,
      size_t size,
      long double a) {
    int exponent{};

    long double normalized = MPFR_CXX_FABSL(MPFR_CXX_FREXPL(a, &exponent));
    if (normalized == 0.5L) {
      bool signbit = MPFR_CXX_SIGNBIT(a) != 0;

      // a is a power of two
      // a = signbit * 2^(exp-1)

      if (size >= 1) {
        std::memset(m_mantissa, 0, sizeof(mp_limb_t) * (size - 1));
      }
      m_exponent = exponent;
      m_mantissa[size - 1] = _::pow2_mantissa_last;
      m_actual_prec_sign = prec_negate_if(1, signbit);
    } else {
      _::mpfr_raii_setter_t g{
          precision_mpfr,
          m_mantissa,
          &m_exponent,
          &m_actual_prec_sign,
      };
      _::set_primitive(g, a);
    }
  }
};

template <typename T> struct is_mp_float { static constexpr bool value = false; };
template <precision_t P> struct is_mp_float<mp_float_t<P>> { static constexpr bool value = true; };
template <precision_t P> struct is_mp_float<mp_float_t<P> const> {
  static constexpr bool value = true;
};

template <typename T1, typename T2> struct have_common_mp_type {
  static constexpr bool value = is_arithmetic<T1>::value and is_arithmetic<T2>::value and
                                (is_mp_float<T1>::value or is_mp_float<T2>::value);
};

inline void set_add(mpfr_raii_setter_t& out, mpfr_cref_t a, mpfr_cref_t b) {
  mpfr_add(&out.m, &a.m, &b.m, _::get_rnd());
}

inline void set_sub(mpfr_raii_setter_t& out, mpfr_cref_t a, mpfr_cref_t b) {
  MPFR_SIGN(&b.m) *= -1;
  set_add(out, a, b);
}

inline void set_mul(mpfr_raii_setter_t& out, mpfr_cref_t a, mpfr_cref_t b) {
  if ((mpfr_custom_get_significand(&a.m) == mpfr_custom_get_significand(&b.m)) and
      (mpfr_custom_get_exp(&a.m) == mpfr_custom_get_exp(&b.m)) and
      (mpfr_signbit(&a.m) == mpfr_signbit(&b.m)) and (mpfr_get_prec(&a.m) == mpfr_get_prec(&b.m))) {
    mpfr_sqr(&out.m, &a.m, _::get_rnd());
    return;
  }
  mpfr_mul(&out.m, &a.m, &b.m, _::get_rnd());
}

inline void set_div(mpfr_raii_setter_t& out, mpfr_cref_t a, mpfr_cref_t b) {
  mpfr_div(&out.m, &a.m, &b.m, _::get_rnd());
}

struct heap_str_t /* NOLINT(cppcoreguidelines-special-member-functions) */ {
  char* p;
  explicit heap_str_t(size_t n) : p{n > 0 ? new char[n] : nullptr} {}
  void init(size_t n) {
    MPFR_CXX_ASSERT(p == nullptr);
    p = n > 0 ? new char[n] : nullptr;
  }

  ~heap_str_t() { delete[] p; }
};

template <typename CharT, typename Traits>
auto has_flag(
    std::basic_ostream<CharT, Traits>& out, typename std::basic_ostream<CharT, Traits>::fmtflags f)
    -> bool {
  return (out.flags() & f) == f;
}

template <typename CharT, typename Traits>
void print_n(
    std::basic_ostream<CharT, Traits>& out,
    typename std::basic_ostream<CharT, Traits>::char_type c,
    size_t n) {
  constexpr size_t pack_size = 64;
  CharT buffer[pack_size];
  for (auto& e : buffer) {
    e = c;
  }
  while (n >= pack_size) {
    n -= pack_size;
    out.write(buffer, pack_size);
  }
  out.write(buffer, static_cast<std::streamsize>(n));
}

template <typename CharT, typename Traits>
void write_to_ostream(
    std::basic_ostream<CharT, Traits>& out,
    mpfr_cref_t x_,
    char* stack_buffer,
    size_t stack_bufsize) {
  using ostr = std::basic_ostream<CharT, Traits>;

  char format[128] = {};
  std::size_t pos = 0;
  format[pos++] = '%';
  format[pos++] = '.';

  bool hf = _::has_flag(out, ostr::scientific) and _::has_flag(out, ostr::fixed);
  long precision = hf ? mpfr_get_prec(&x_.m) / 4 : static_cast<long>(out.precision());
  if (precision > 0) {
    std::snprintf(format + pos, sizeof(format) - pos, "%ld", precision);
  }
  pos = std::strlen(format);
  format[pos++] = 'R';

  bool u = has_flag(out, ostr::uppercase);
  if (hf) {
    format[pos++] = u ? 'A' : 'a';
  } else if (_::has_flag(out, ostr::scientific)) {
    format[pos++] = u ? 'E' : 'e';
  } else if (_::has_flag(out, ostr::fixed)) {
    format[pos++] = u ? 'F' : 'f';
  } else {
    format[pos++] = u ? 'G' : 'g';
  }
  format[pos++] = '\0';

  bool signbit = mpfr_signbit(&x_.m);
  MPFR_SIGN(&x_.m) = 1;

  auto const size_needed = static_cast<std::size_t>(mpfr_snprintf(nullptr, 0, format, &x_.m)) + 1;

  std::size_t zero_padding = 0;

  bool use_heap = size_needed > stack_bufsize;
  _::heap_str_t heap_buffer{use_heap ? size_needed : 0};

  char* ptr = use_heap ? heap_buffer.p : stack_buffer;

  mpfr_snprintf(ptr, size_needed, format, &x_.m);
  if (not hf and _::has_flag(out, ostr::showpoint) and out.precision() > 0) {
    char* dot_ptr = std::strchr(ptr, '.');
    if (dot_ptr == nullptr) {
      zero_padding = 1 + static_cast<std::size_t>(out.precision()) - size_needed;
    }
  }

  std::size_t n_padding = 0;
  if (static_cast<std::size_t>(out.width()) >=
      size_needed - 1 + ((signbit or _::has_flag(out, ostr::showpos)) ? 1 : 0)) {

    n_padding = static_cast<std::size_t>(out.width()) - size_needed + 1 -
                ((signbit or _::has_flag(out, ostr::showpos)) ? 1 : 0) - zero_padding;
  }

  out.width(0);
  if (not _::has_flag(out, ostr::left) and not _::has_flag(out, ostr::internal)) {
    _::print_n(out, out.fill(), n_padding);
  }

  if (signbit) {
    out.put(out.widen('-'));
  } else if (_::has_flag(out, ostr::showpos)) {
    out.put(out.widen('+'));
  }

  if (_::has_flag(out, ostr::internal)) {
    _::print_n(out, out.fill(), n_padding);
  }

  out << ptr;

  if (zero_padding > 0) {
    out.put(out.widen('.'));
    _::print_n(out, '0', zero_padding - 1);
  }

  if (_::has_flag(out, ostr::left)) {
    _::print_n(out, out.fill(), n_padding);
  }
}

template <typename CharT, typename Traits, precision_t P>
inline void dump_repr(std::basic_ostream<CharT, Traits>& out, mp_float_t<P> const& x) {
  out << "repr\n";
  out << "exp       : " << impl_access::exp_const(x) << '\n';
  out << "prec|sign : " << impl_access::actual_prec_sign_const(x) << '\n';
  out << "mantissa  : ";
  for (auto e : impl_access::mantissa_const(x)) {
    out << e << ' ';
  }
  out << '\n';
  out << "value\n";
  out << x << '\n';
  out << "end\n";
}

template <precision_t P>
auto apply_unary_op(mp_float_t<P> const& x, int (*op)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t)) noexcept
    -> mp_float_t<P> {
  mp_float_t<P> out;
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(x);
    op(&g.m, &x_.m, _::get_rnd());
  }
  return out;
}

template <typename U, typename V>
auto apply_binary_op(
    U const& x, V const& y, int (*op)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t)) noexcept ->
    typename common_type<U, V>::type {
  typename common_type<U, V>::type out;

  typename _::into_mp_float_lossless<U>::type const& a{x};
  typename _::into_mp_float_lossless<V>::type const& b{y};
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_cref_t x_ = _::impl_access::mpfr_cref(a);
    _::mpfr_cref_t y_ = _::impl_access::mpfr_cref(b);
    op(&g.m, &x_.m, &y_.m, _::get_rnd());
  }
  return out;
}

template <typename T> auto to_lvalue(T&& arg) -> T& { return arg; }
template <bool Is_Const> struct into_mpfr;

template <> struct into_mpfr<true> {
  static auto get_pointer(mpfr_cref_t&& p) -> mpfr_srcptr { return &p.m; }
  template <precision_t P> static auto get_mpfr(mp_float_t<P> const& x) -> mpfr_cref_t {
    return impl_access::mpfr_cref(x);
  }
};

template <> struct into_mpfr<false> {
  static auto get_pointer(mpfr_raii_setter_t&& p) -> mpfr_ptr { return &p.m; }
  template <precision_t P> static auto get_mpfr(mp_float_t<P>& x) -> mpfr_raii_setter_t {
    return {
        static_cast<mpfr_prec_t>(P),
        static_cast<mp_limb_t*>(impl_access::mantissa_mut(x)),
        &impl_access::exp_mut(x),
        &impl_access::actual_prec_sign_mut(x),
    };
  }
};

template <typename U, typename V>
[[MPFR_CXX_NODISCARD]] auto arithmetic_op(
    U const& a,
    V const& b,
    void (*op)(_::mpfr_raii_setter_t&, _::mpfr_cref_t, _::mpfr_cref_t)) noexcept ->
    typename _::common_type<U, V>::type {

  typename _::common_type<U, V>::type out;
  typename _::into_mp_float_lossless<U>::type const& a_{a};
  typename _::into_mp_float_lossless<V>::type const& b_{b};
  {
    _::mpfr_raii_setter_t&& g = _::impl_access::mpfr_setter(out);
    _::mpfr_cref_t ac = _::impl_access::mpfr_cref(a_);
    _::mpfr_cref_t bc = _::impl_access::mpfr_cref(b_);
    op(g, ac, bc);
  }
  return out;
}

template <typename U, typename V>
[[MPFR_CXX_NODISCARD]] auto
comparison_op(U const& a, V const& b, int (*comp)(mpfr_srcptr, mpfr_srcptr)) noexcept -> bool {
  typename _::into_mp_float_lossless<U>::type const& a_{a};
  typename _::into_mp_float_lossless<V>::type const& b_{b};

  _::mpfr_cref_t ac = _::impl_access::mpfr_cref(a_);
  _::mpfr_cref_t bc = _::impl_access::mpfr_cref(b_);

  return comp(&ac.m, &bc.m) != 0;
}

} // namespace _

constexpr digits2::digits2(precision_t prec) noexcept : m_value{static_cast<std::uint64_t>(prec)} {}
constexpr digits2::operator precision_t() const noexcept {
  return static_cast<precision_t>(m_value);
}

constexpr digits10::digits10(precision_t prec) noexcept
    : m_value{_::digits2_to_10(static_cast<std::uint64_t>(prec))} {}
constexpr digits10::operator precision_t() const noexcept {
  return static_cast<precision_t>(_::digits10_to_2(m_value));
}

} // namespace mpfr

#include "mpfr/detail/epilogue.hpp"

#endif /* end of include guard MPFR_HPP_NZTOL31N */
