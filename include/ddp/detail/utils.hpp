#ifndef ALL_HPP_VYM0WI7T
#define ALL_HPP_VYM0WI7T

#include <cstdint>
#include <cstddef>

#ifdef NDEBUG
#endif

#include <initializer_list>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/seq/variadic_seq_to_seq.hpp>
#include <boost/preprocessor/seq/seq.hpp>
#include <omp.h>

#include <Eigen/Core>
#include <fmt/core.h>
#include <gsl-lite/gsl-lite.hpp>
#include "fancy-assert/fancy_assert.hpp"

#define DDP_DECLVAL(...) static_cast<__VA_ARGS__ (*)()>(nullptr)()
#define DDP_SIZEOF(...) static_cast<::ddp::index_t>(sizeof(__VA_ARGS__))
#define DDP_VSIZEOF(...) static_cast<::ddp::index_t>(sizeof...(__VA_ARGS__))
#define DDP_MOVE(...) static_cast<typename ::std::remove_reference<decltype(__VA_ARGS__)>::type&&>(__VA_ARGS__)

#define DDP_DECLTYPE_AUTO(...)                                                                                         \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) { return __VA_ARGS__; }                                       \
  static_assert(true, "")

#define DDP_PRECOND_DECLTYPE_AUTO(Precondition_Block, ...)                                                             \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) {                                                             \
    BOOST_PP_REMOVE_PARENS Precondition_Block return __VA_ARGS__;                                                      \
  }                                                                                                                    \
  static_assert(true, "")

/**********************************************************************************************************************/
#define DDP_IMPL_BIND(r, Tuple, Index, Identifier) auto&& Identifier = ::ddp::detail::adl_get<Index>(Tuple);

#define DDP_IMPL_BIND_ID_SEQ(CV_Auto, Identifiers, Tuple, SeqSize, TupleId)                                            \
  static_assert(                                                                                                       \
      ::std::tuple_size<                                                                                               \
          typename ::std::remove_const<typename ::std::remove_reference<decltype(Tuple)>::type>::type>::value ==       \
          SeqSize,                                                                                                     \
      "Wrong number of identifiers.");                                                                                 \
  CV_Auto TupleId = Tuple;                                                                                             \
  BOOST_PP_SEQ_FOR_EACH_I(DDP_IMPL_BIND, TupleId, Identifiers)                                                         \
  ((void)0)

#define DDP_BIND(CV_Auto, Identifiers, Tuple)                                                                          \
  DDP_IMPL_BIND_ID_SEQ(                                                                                                \
      CV_Auto,                                                                                                         \
      BOOST_PP_TUPLE_TO_SEQ(Identifiers),                                                                              \
      Tuple,                                                                                                           \
      BOOST_PP_TUPLE_SIZE(Identifiers),                                                                                \
      BOOST_PP_CAT(_dummy_tuple_variable_id_, __LINE__))
/**********************************************************************************************************************/

namespace ddp {

struct log_file_t {
  std::FILE* ptr;
  explicit log_file_t(char const* path);

private:
  struct open_file_set_t;
  static open_file_set_t open_files;
  static void add_file(char const* path);
};

struct chronometer_t {

  chronometer_t(chronometer_t const&) = delete;
  chronometer_t(chronometer_t&&) = delete;
  auto operator=(chronometer_t const&) -> chronometer_t& = delete;
  auto operator=(chronometer_t&&) -> chronometer_t& = delete;

  explicit chronometer_t(char const* message, log_file_t file = log_file_t{"/tmp/chrono.log"});
  ~chronometer_t();

private:
  std::intmax_t m_begin;
  std::intmax_t m_end;

  char const* m_message;
  log_file_t const m_file;
};

struct unsafe_t {};
struct safe_t {};

static constexpr unsafe_t unsafe;
static constexpr safe_t safe;

using index_t = std::int64_t;
using usize = std::size_t;
using u64 = std::uint64_t;

namespace detail {

template <typename T>
auto to_owned(T s) -> std::string;

template <>
inline auto to_owned(std::string s) -> std::string {
  return DDP_MOVE(s);
}

template <>
inline auto to_owned(fmt::string_view s) -> std::string {
  return {s.begin(), s.end()};
}

template <size_t I>
void get() = delete;

template <std::size_t I, typename T>
auto adl_get(T&& x) DDP_DECLTYPE_AUTO(get<I>(static_cast<T&&>(x)));

template <typename L, typename R>
using add_t = decltype(DDP_DECLVAL(L) + DDP_DECLVAL(R));

template <bool Cond>
struct conditional;

// clang-format off
template <> struct conditional<true>  { template <typename T, typename F> using type = T; };
template <> struct conditional<false> { template <typename T, typename F> using type = F; };

#define DDP_CONDITIONAL(Cond, ...) typename ::ddp::detail::conditional<(Cond)>::template type<__VA_ARGS__>
// clang-format on

template <typename... Ts>
struct tuple;

template <typename... Ts>
auto make_tuple(Ts... args) -> tuple<Ts...>;
} // namespace detail

template <index_t N>
struct fix_index;

struct dyn_index {
private:
  index_t m_value{};

public:
  static constexpr bool known_at_compile_time = false;
  static constexpr index_t value_at_compile_time = Eigen::Dynamic;

  constexpr dyn_index() : m_value{} {}
  explicit constexpr dyn_index(index_t value) : m_value{value} {}
  template <index_t N>
  explicit constexpr dyn_index(fix_index<N>) : m_value{N} {}
  constexpr auto value() const -> index_t { return m_value; }

  // clang-format off
  friend auto operator+(dyn_index m, dyn_index n) -> dyn_index { return dyn_index{m.value() + n.value()}; }
  friend auto operator-(dyn_index m, dyn_index n) -> dyn_index { return dyn_index{m.value() - n.value()}; }
  friend auto operator*(dyn_index m, dyn_index n) -> dyn_index { return dyn_index{m.value() * n.value()}; }
  // clang-format on
};

template <index_t N>
struct fix_index {
  static constexpr bool known_at_compile_time = true;
  static constexpr const index_t value_at_compile_time = N;

  constexpr fix_index() = default;
  explicit constexpr fix_index(index_t value) {
#if __cplusplus >= 201402L
    DDP_ASSERT(value == N);
#endif
  }
  explicit constexpr fix_index(dyn_index value) : fix_index{value.value()} {}

  constexpr auto value() const -> index_t { return N; }

  // clang-format off
  template <index_t M> friend auto operator+(fix_index<N>, fix_index<M>)   -> fix_index<N + M> { return {}; }
  template <index_t M> friend auto operator-(fix_index<N>, fix_index<M>)   -> fix_index<N - M> { return {}; }
  template <index_t M> friend auto operator*(fix_index<N>, fix_index<M>)   -> fix_index<N * M> { return {}; }

  friend auto operator+(dyn_index n, fix_index)   -> dyn_index { return dyn_index{n.value() + N}; }
  friend auto operator+(fix_index, dyn_index n)   -> dyn_index { return dyn_index{n.value() + N}; }
  friend auto operator-(dyn_index n, fix_index)   -> dyn_index { return dyn_index{n.value() - N}; }
  friend auto operator-(fix_index, dyn_index n)   -> dyn_index { return dyn_index{n.value() - N}; }
  friend auto operator*(dyn_index n, fix_index)   -> dyn_index { return dyn_index{n.value() * N}; }
  friend auto operator*(fix_index, dyn_index n)   -> dyn_index { return dyn_index{n.value() * N}; }
  // clang-format on
};

namespace concepts {
template <typename T>
struct index {
  static constexpr bool value = false;
};
template <index_t N>
struct index<fix_index<N>> {
  static constexpr bool value = true;
};
template <>
struct index<dyn_index> {
  static constexpr bool value = true;
};
#define DDP_INDEX_CONCEPT(T) ::ddp::concepts::index<BOOST_PP_REMOVE_PARENS(T)>::value
} // namespace concepts

namespace detail {

template <typename... Ts>
struct debug_t;
template <typename... Types, typename... Args>
auto show_types(Args&&...) -> debug_t<Types..., Args...>;

template <typename... Ts>
void unused(Ts const&...) {}

enum ternary_e { yes, no, maybe };

#define DDP_COMPARISON_CHECK(Impl_Name, Assert_Name, Compile_Time_Cond, Run_Time_Cond)                                 \
  template <typename N, typename M, typename Enable = void>                                                            \
  struct Impl_Name {                                                                                                   \
    static constexpr ternary_e value = maybe;                                                                          \
    static constexpr void assertion(N n, M m) { DDP_ASSERT(Run_Time_Cond); };                                          \
  };                                                                                                                   \
                                                                                                                       \
  template <index_t N, index_t M>                                                                                      \
  struct Impl_Name<fix_index<N>, fix_index<M>, typename std::enable_if<not(Compile_Time_Cond)>::type> {                \
    static constexpr ternary_e value = no;                                                                             \
    static constexpr void assertion(fix_index<N>, fix_index<M>) = delete;                                              \
  };                                                                                                                   \
                                                                                                                       \
  template <index_t N, index_t M>                                                                                      \
  struct Impl_Name<fix_index<N>, fix_index<M>, typename std::enable_if<(Compile_Time_Cond)>::type> {                   \
    static constexpr ternary_e value = yes;                                                                            \
    static constexpr void assertion(fix_index<N>, fix_index<M>){};                                                     \
  };                                                                                                                   \
  template <typename N, typename M>                                                                                    \
  constexpr void Assert_Name(N n, M m) {                                                                               \
    Impl_Name<N, M>::assertion(n, m); /* NOLINT(bugprone-macro-parentheses) */                                         \
  }                                                                                                                    \
  static_assert(true, "")

DDP_COMPARISON_CHECK(check_leq, assert_leq, N <= M, n.value() <= m.value());
DDP_COMPARISON_CHECK(check_geq, assert_geq, N >= M, n.value() >= m.value());
DDP_COMPARISON_CHECK(check_lt, assert_lt, N < M, n.value() < m.value());
DDP_COMPARISON_CHECK(check_gt, assert_gt, N > M, n.value() > m.value());
DDP_COMPARISON_CHECK(check_eq, assert_eq, N == M, n.value() == m.value());

#undef DDP_COMPARISON_CHECK
} // namespace detail
} // namespace ddp

#endif /* end of include guard ALL_HPP_VYM0WI7T */
