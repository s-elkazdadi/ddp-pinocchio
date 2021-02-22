#ifndef ENUMS_HPP_4PXFRLJ0
#define ENUMS_HPP_4PXFRLJ0

#include <climits>
#include <cstdio>
#include <cstring>
#include <cfenv>
#include <cstdint>

#include <mpfr.h>

#include "mpfr/detail/.hedley.h"
#include "mpfr/detail/prologue.hpp"

namespace mpfr {

/// The count of bits that make up the number's mantissa.\n
/// This must be a strictly positive value.
enum struct precision_t : mpfr_prec_t {};

/// Number of digits in base 2.
struct digits2 {
private:
  std::uint64_t m_value;

public:
  /// Constructs with the given number of digits.
  constexpr explicit digits2(std::uint64_t value) noexcept : m_value{value} {}
  /// Converts from precision enum.
  constexpr explicit digits2(precision_t prec) noexcept;
  /// Converts to precision enum.
  constexpr operator // NOLINT(hicpp-explicit-conversions)
      precision_t() const noexcept;
};

/// Number of digits in base 10.\n
/// The actual precision may be slightly larger since the conversion between base 2 and base 10 is
/// inexact.
struct digits10 {
private:
  std::uint64_t m_value;

public:
  /// Constructs with the given number of digits.
  constexpr explicit digits10(std::uint64_t value) noexcept : m_value{value} {}
  /// Converts from precision enum.
  constexpr explicit digits10(precision_t prec) noexcept;
  /// Converts to precision enum.
  constexpr operator // NOLINT(hicpp-explicit-conversions)
      precision_t() const noexcept;
};
} // namespace mpfr

#include "mpfr/detail/epilogue.hpp"

#endif /* end of include guard ENUMS_HPP_4PXFRLJ0 */
