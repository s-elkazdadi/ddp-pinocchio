#ifndef ZERO_HPP_BSDF0JPH
#define ZERO_HPP_BSDF0JPH

#include "ddp/detail/utils.hpp"

namespace ddp {
namespace zero {
struct zero_t {
  static auto transpose() -> zero_t { return {}; }
  static auto row(index_t) -> zero_t { return {}; }
  template <typename T>
  auto operator=(T const&) -> zero_t& = delete;

  template <typename T>
  using neg_t = decltype(-DDP_DECLVAL(T const&));
  // clang-format off
  template <typename T> friend auto operator+=(T& x, zero_t) noexcept -> T& { return x; }

  template <typename T> friend auto operator+(zero_t      , T const& x ) noexcept -> T const& { return  x; }
  template <typename T> friend auto operator+(T const& x  , zero_t     ) noexcept -> T const& { return  x; }
  template <typename T> friend auto operator-(zero_t      , T const& x ) noexcept -> neg_t<T> { return -x; }
  template <typename T> friend auto operator-(T const& x  , zero_t     ) noexcept -> T const& { return  x; }
                        friend auto operator+(zero_t      , zero_t     ) noexcept -> zero_t   { return {}; }
                        friend auto operator-(zero_t      , zero_t     ) noexcept -> zero_t   { return {}; }

  friend auto operator-(zero_t) -> zero_t { return {}; }

  template <typename T> friend auto operator*(zero_t  , T const&) noexcept -> zero_t { return {}; }
  template <typename T> friend auto operator*(T const&, zero_t  ) noexcept -> zero_t { return {}; }
                        friend auto operator*(zero_t  , zero_t  ) noexcept -> zero_t { return {}; }

  template <typename T> friend auto operator/(zero_t  , T const&) noexcept -> zero_t { return {}; }
  template <typename T> friend void operator/(T const&, zero_t  ) noexcept = delete;
                        friend void operator/(zero_t  , zero_t  ) noexcept = delete;

  // template <typename T> friend auto operator+=(T const&, zero_t) noexcept -> void {}
  // template <typename T> friend auto operator-=(T const&, zero_t) noexcept -> void {}
  // clang-format on
};

static constexpr zero_t zero{};

} // namespace zero
} // namespace ddp

#endif /* end of include guard ZERO_HPP_BSDF0JPH */
