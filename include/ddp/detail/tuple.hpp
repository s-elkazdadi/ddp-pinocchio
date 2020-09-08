#ifndef TUPLE_HPP_AUVTR2OI
#define TUPLE_HPP_AUVTR2OI

#include "utils.hpp"
#include "../indexer.hpp"
#include <tuple>

namespace ddp {
namespace detail {

// TODO c++11 ?
template <index_t N>
using make_index_sequence = std::make_integer_sequence<index_t, N>;

template <typename T, index_t I>
struct tuple_leaf {
  T m_inner;
};

template <index_t I, typename T>
auto get_mut(tuple_leaf<T, I>& leaf) -> T& {
  return leaf.m_inner;
}

template <index_t I, typename T>
auto get_const(tuple_leaf<T, I> const& leaf) -> T const& {
  return leaf.m_inner;
}

template <typename... Ts>
struct tuple {
  template <typename T>
  struct indexed_tuple;
  template <index_t... Is>
  struct indexed_tuple<std::integer_sequence<index_t, Is...>> : tuple_leaf<Ts, Is>... {
    explicit indexed_tuple(Ts... args) : tuple_leaf<Ts, Is>{static_cast<Ts&&>(args)}... {}
  };

  indexed_tuple<make_index_sequence<DDP_VSIZEOF(Ts)>> _m_impl;
  tuple(Ts... args) noexcept : _m_impl{static_cast<Ts&&>(args)...} {}
};

template <index_t I, typename... Ts>
auto get(tuple<Ts...>& tup) noexcept -> decltype(detail::get_mut<I>(tup._m_impl)) {
  return detail::get_mut<I>(tup._m_impl);
}

template <index_t I, typename... Ts>
auto get(tuple<Ts...> const& tup) noexcept -> decltype(detail::get_const<I>(tup._m_impl)) {
  return detail::get_const<I>(tup._m_impl);
}

inline constexpr auto one_of(bool const values[], index_t n) -> bool {
  return (n == 0) ? false : values[0] || one_of(values + 1, n - 1);
}

inline constexpr auto all_of(bool const values[], index_t n) -> bool {
  return (n == 0) ? true : values[0] && all_of(values + 1, n - 1);
}

} // namespace detail

namespace ranges {

template <typename... Iters>
struct zip_iterator_t {
  detail::tuple<Iters...> m_iters;

  zip_iterator_t(Iters... iters) noexcept : m_iters(DDP_MOVE(iters)...) {}

  using difference_type = std::ptrdiff_t;
  using value_type = detail::tuple<typename std::iterator_traits<Iters>::value_type...>;
  using pointer = void;
  using reference = detail::tuple<typename std::iterator_traits<Iters>::reference...>;
  using iterator_category = std::input_iterator_tag;

  template <index_t... Is>
  void increment_impl(std::integer_sequence<index_t, Is...>) noexcept {
    int const dummy[] = {((void)(++detail::get<Is>(m_iters)), 0)...};
    (void)dummy;
  }
  template <index_t... Is>
  void decrement_impl(std::integer_sequence<index_t, Is...>) noexcept {
    int const dummy[] = {((void)(--detail::get<Is>(m_iters)), 0)...};
    (void)dummy;
  }
  template <index_t... Is>
  auto deref_impl(std::integer_sequence<index_t, Is...>) noexcept -> reference {
    return reference{*detail::get<Is>(m_iters)...};
  }

  template <index_t... Is>
  static auto cmp_eq_impl(zip_iterator_t a, zip_iterator_t b, std::integer_sequence<index_t, Is...>) -> bool {
    bool const eq[] = {static_cast<bool>(detail::get<Is>(a.m_iters) == detail::get<Is>(b.m_iters))...};
    bool is_eq = detail::one_of(eq, DDP_VSIZEOF(Iters));
    if (is_eq) {
      assert(detail::all_of(eq, DDP_VSIZEOF(Iters)));
    }

    return is_eq;
  }

  using index_seq = detail::make_index_sequence<DDP_VSIZEOF(Iters)>;
  auto operator++() noexcept -> zip_iterator_t& {
    this->increment_impl(index_seq{});
    return *this;
  }
  auto operator--() noexcept -> zip_iterator_t& {
    this->decrement_impl(index_seq{});
    return *this;
  }
  friend auto operator==(zip_iterator_t a, zip_iterator_t b) -> bool {
    return zip_iterator_t::cmp_eq_impl(a, b, index_seq{});
  }
  auto operator*() noexcept -> reference { return this->deref_impl(index_seq{}); }
  DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(zip_iterator_t);
};

using std::begin;
using std::end;
template <typename... Rngs>
struct zip_range_t {
  detail::tuple<Rngs...> m_rngs;

  zip_range_t(Rngs... rngs) noexcept : m_rngs(static_cast<Rngs&&>(rngs)...) {}

  using iterator = zip_iterator_t<decltype(begin(DDP_DECLVAL(Rngs)))...>;
  using const_iterator = zip_iterator_t<decltype(begin(DDP_DECLVAL(Rngs const)))...>;

  using index_seq = detail::make_index_sequence<DDP_VSIZEOF(Rngs)>;

  template <index_t... Is>
  auto begin_mut_impl(std::integer_sequence<index_t, Is...>) noexcept -> iterator {
    return iterator{begin(detail::get<Is>(m_rngs))...};
  }
  template <index_t... Is>
  auto begin_const_impl(std::integer_sequence<index_t, Is...>) const noexcept -> const_iterator {
    return const_iterator{begin(detail::get<Is>(m_rngs))...};
  }
  template <index_t... Is>
  auto end_mut_impl(std::integer_sequence<index_t, Is...>) noexcept -> iterator {
    return iterator{end(detail::get<Is>(m_rngs))...};
  }
  template <index_t... Is>
  auto end_const_impl(std::integer_sequence<index_t, Is...>) const noexcept -> const_iterator {
    return const_iterator{end(detail::get<Is>(m_rngs))...};
  }

  friend auto begin(zip_range_t& zip) noexcept -> iterator { return zip.begin_mut_impl(index_seq{}); }
  friend auto begin(zip_range_t const& zip) noexcept -> const_iterator { return zip.begin_const_impl(index_seq{}); }
  friend auto end(zip_range_t& zip) noexcept -> iterator { return zip.end_mut_impl(index_seq{}); }
  friend auto end(zip_range_t const& zip) noexcept -> const_iterator { return zip.end_const_impl(index_seq{}); }
};

template <typename Rng>
struct reversed_range_t {
  Rng m_rng;

  using iterator = std::reverse_iterator<typename Rng::iterator>;
  using const_iterator = std::reverse_iterator<typename Rng::const_iterator>;
  using reverse_iterator = typename Rng::iterator;
  using const_reverse_iterator = typename Rng::const_iterator;

  friend auto begin(reversed_range_t& rng) noexcept -> iterator { return iterator{end(rng.m_rng)}; }
  friend auto begin(reversed_range_t const& rng) noexcept -> const_iterator { return const_iterator{end(rng.m_rng)}; }
  friend auto end(reversed_range_t& rng) noexcept -> iterator { return iterator{begin(rng.m_rng)}; }
  friend auto end(reversed_range_t const& rng) noexcept -> const_iterator { return const_iterator{begin(rng.m_rng)}; }
};

template <typename... Rngs>
auto zip(Rngs&&... rngs) -> zip_range_t<Rngs...> {
  return {static_cast<Rngs&&>(rngs)...};
}

template <typename Rng>
auto reverse(Rng&& rng) -> reversed_range_t<Rng> {
  return {static_cast<Rng&&>(rng)};
}

} // namespace ranges
} // namespace ddp

namespace std {
template <typename... Ts>
struct tuple_size<ddp::detail::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};
} // namespace std

#endif /* end of include guard TUPLE_HPP_AUVTR2OI */
