#ifndef ALL_HPP_VYM0WI7T
#define ALL_HPP_VYM0WI7T

#include <cstdint>
#include <cstddef>

#ifdef NDEBUG
#undef NDEBUG
#endif

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include "boost/preprocessor/cat.hpp"
#include "boost/preprocessor/tuple/size.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"
#include "boost/preprocessor/seq/for_each.hpp"
#include "boost/preprocessor/seq/for_each_i.hpp"
#include "boost/preprocessor/punctuation/remove_parens.hpp"
#include "boost/preprocessor/variadic/to_seq.hpp"
#include "boost/preprocessor/seq/variadic_seq_to_seq.hpp"
#include "boost/preprocessor/seq/seq.hpp"

#include <Eigen/Core>
#include <fmt/core.h>

#define DDP_DECLVAL(...) static_cast<__VA_ARGS__ (*)() noexcept>(nullptr)()
#define DDP_SIZEOF(...) static_cast<::ddp::index_t>(sizeof(__VA_ARGS__))
#define DDP_VSIZEOF(...) static_cast<::ddp::index_t>(sizeof...(__VA_ARGS__))
#define DDP_MOVE(...) static_cast<typename ::std::remove_reference<decltype(__VA_ARGS__)>::type&&>(__VA_ARGS__)

#define DDP_ASSERT(Cond, Message) (static_cast<bool>(Cond) ? (void)(0) : ::ddp::fast_fail(Message))

/**********************************************************************************************************************/
#define DDP_IMPL_BIND(r, Tuple, Index, Identifier) auto&& Identifier = get<Index>(Tuple);

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
  using ddp::detail::get;                                                                                              \
  DDP_IMPL_BIND_ID_SEQ(                                                                                                \
      CV_Auto,                                                                                                         \
      BOOST_PP_TUPLE_TO_SEQ(Identifiers),                                                                              \
      Tuple,                                                                                                           \
      BOOST_PP_TUPLE_SIZE(Identifiers),                                                                                \
      BOOST_PP_CAT(_dummy_tuple_variable_id_, __LINE__))
/**********************************************************************************************************************/

namespace ddp {

[[noreturn]] void fast_fail(fmt::string_view message) noexcept;

struct unsafe_t {};
struct safe_t {};

static constexpr unsafe_t unsafe;
static constexpr safe_t safe;

using index_t = std::int64_t;
using usize = std::size_t;
using u64 = std::uint64_t;

namespace detail {
template <typename L, typename R>
using add_t = decltype(DDP_DECLVAL(L) + DDP_DECLVAL(R));

template <bool Cond>
struct conditional;

// clang-format off
template <> struct conditional<true>  { template <typename T, typename F> using type = T; };
template <> struct conditional<false> { template <typename T, typename F> using type = F; };

#define DDP_CONDITIONAL(Cond, ...) typename ::ddp::detail::conditional<(Cond)>::template type<__VA_ARGS__>
// clang-format on
} // namespace detail

template <index_t N>
struct fix_index;

struct dyn_index {
private:
  index_t m_value;

public:
  static constexpr bool known_at_compile_time = false;
  static constexpr index_t value_at_compile_time = Eigen::Dynamic;

  explicit constexpr dyn_index(index_t value) noexcept : m_value{value} {}
  template <index_t N>
  explicit constexpr dyn_index(fix_index<N>) noexcept : m_value{N} {}
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

  constexpr fix_index() noexcept = default;
  explicit constexpr fix_index(index_t value) noexcept { assert(value == N); }
  explicit constexpr fix_index(dyn_index value) noexcept : fix_index{value.value()} {}

  constexpr auto value() const -> index_t { return N; }

  // clang-format off
  template <index_t M> friend auto operator+(fix_index<N>, fix_index<M>) noexcept -> fix_index<N + M> { return {}; }
  template <index_t M> friend auto operator-(fix_index<N>, fix_index<M>) noexcept -> fix_index<N - M> { return {}; }
  template <index_t M> friend auto operator*(fix_index<N>, fix_index<M>) noexcept -> fix_index<N * M> { return {}; }

  friend auto operator+(dyn_index n, fix_index) noexcept -> dyn_index { return dyn_index{n.value() + N}; }
  friend auto operator+(fix_index, dyn_index n) noexcept -> dyn_index { return dyn_index{n.value() + N}; }
  friend auto operator-(dyn_index n, fix_index) noexcept -> dyn_index { return dyn_index{n.value() - N}; }
  friend auto operator-(fix_index, dyn_index n) noexcept -> dyn_index { return dyn_index{n.value() - N}; }
  friend auto operator*(dyn_index n, fix_index) noexcept -> dyn_index { return dyn_index{n.value() * N}; }
  friend auto operator*(fix_index, dyn_index n) noexcept -> dyn_index { return dyn_index{n.value() * N}; }
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
    static constexpr void assertion(N n, M m) noexcept { assert(Run_Time_Cond); };                                     \
  };                                                                                                                   \
                                                                                                                       \
  template <index_t N, index_t M>                                                                                      \
  struct Impl_Name<fix_index<N>, fix_index<M>, typename std::enable_if<not(Compile_Time_Cond)>::type> {                \
    static constexpr ternary_e value = no;                                                                             \
    static constexpr void assertion(fix_index<N>, fix_index<M>) noexcept = delete;                                     \
  };                                                                                                                   \
                                                                                                                       \
  template <index_t N, index_t M>                                                                                      \
  struct Impl_Name<fix_index<N>, fix_index<M>, typename std::enable_if<(Compile_Time_Cond)>::type> {                   \
    static constexpr ternary_e value = yes;                                                                            \
    static constexpr void assertion(fix_index<N>, fix_index<M>) noexcept {};                                           \
  };                                                                                                                   \
  template <typename N, typename M>                                                                                    \
  constexpr void Assert_Name(N n, M m) noexcept {                                                                      \
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

namespace eigen {

template <typename T>
using view_t = Eigen::Map<T, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;

template <typename T, typename Rows, typename Cols, int Options, typename Max_Rows, typename Max_Cols>
using matrix_t = Eigen::Matrix<
    T,
    Rows::value_at_compile_time,
    Cols::value_at_compile_time,
    Options,
    Max_Rows::value_at_compile_time,
    Max_Cols::value_at_compile_time>;

template <typename T>
struct type_to_size
    : std::integral_constant<index_t, (T::known_at_compile_time ? Eigen::Dynamic : T::value_at_compile_time)> {};

template <index_t N>
using size_to_type = DDP_CONDITIONAL((N == Eigen::Dynamic), dyn_index, fix_index<N>);

template <
    typename T,
    typename Indexer,
    int Options =
        (Indexer::row_kind::value_at_compile_time == 1 and Indexer::col_kind::value_at_compile_time != 1)
            ? Eigen::RowMajor
            : (Indexer::col_kind::value_at_compile_time == 1 and Indexer::row_kind::value_at_compile_time != 1)
                  ? (Eigen::ColMajor)
                  : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION //
    >
using matrix_from_idx_t = Eigen::Matrix<
    T,
    Indexer::row_kind::value_at_compile_time,
    Indexer::col_kind::value_at_compile_time,
    Options,
    Indexer::max_row_kind::value_at_compile_time,
    Indexer::max_col_kind::value_at_compile_time>;

enum assign_mode_e { equal, add, sub };
namespace detail {

template <bool More_Than_Two>
struct sum_impl;

template <>
struct sum_impl<true> {
  template <typename T, typename... Ts>
  static auto run(T const& first, Ts const&... rest) noexcept
      -> decltype(first + sum_impl<(DDP_VSIZEOF(Ts) > 2)>::run(rest...)) {

    return first + sum_impl<(DDP_VSIZEOF(Ts) > 2)>::run(rest...);
  }
};

template <>
struct sum_impl<false> {
  template <typename T1, typename T2>
  static auto run(T1 const& first, T2 const& second) noexcept -> decltype(first + second) {
    return first + second;
  }
};

template <typename... Ts>
struct get_dims {
  static constexpr index_t rows[] = {Ts::RowsAtCompileTime...};
  static constexpr index_t cols[] = {Ts::ColsAtCompileTime...};
};

constexpr auto first_non_dyn_else_dyn(index_t const arr[], index_t n) -> index_t {
  return (n == 0) ? Eigen::Dynamic              //
                  : ((arr[0] != Eigen::Dynamic) //
                         ? arr[0]               //
                         : first_non_dyn_else_dyn(arr + 1, n - 1));
}

constexpr auto all_equal_to_val_or_dyn(index_t value, index_t const arr[], index_t n) -> bool {
  return (value == Eigen::Dynamic) or                       //
         (n == 0) or                                        //
         ((arr[0] == value or arr[0] == Eigen::Dynamic) and //
          all_equal_to_val_or_dyn(value, arr + 1, n - 1));
}

constexpr auto all_same_or_dyn(index_t const arr[], index_t n) -> bool {
  return all_equal_to_val_or_dyn(first_non_dyn_else_dyn(arr, n), arr, n);
}

constexpr auto all_consecutive_same_or_dyn(index_t const rows[], index_t const cols[], index_t n) -> bool {
  return (n == 0 or n == 1) or            //
         ((cols[0] == rows[1] or          //
           cols[0] == Eigen::Dynamic or   //
           rows[1] == Eigen::Dynamic) and //
          all_consecutive_same_or_dyn(rows + 1, cols + 1, n - 1));
}

constexpr auto equal_or_dyn(index_t n, index_t m) {
  return n == m or n == Eigen::Dynamic or m == Eigen::Dynamic;
}

template <assign_mode_e Mode>
struct assign_impl;

template <>
struct assign_impl<equal> {
  template <typename L, typename R>
  static void run(L& l, R const& r) {
    l = r;
  }
};
template <>
struct assign_impl<add> {
  template <typename L, typename R>
  static void run(L& l, R const& r) {
    l += r;
  }
};
template <>
struct assign_impl<sub> {
  template <typename L, typename R>
  static void run(L& l, R const& r) {
    l -= r;
  }
};

} // namespace detail

template <typename T>
using dyn_vec_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using row_kind = DDP_CONDITIONAL(T::RowsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::RowsAtCompileTime>);
template <typename T>
using col_kind = DDP_CONDITIONAL(T::ColsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::ColsAtCompileTime>);

template <
    typename... Ts,
    typename std::enable_if<                                                         //
        ((DDP_VSIZEOF(Ts) >= 2) and                                                  //
         detail::all_same_or_dyn(detail::get_dims<Ts...>::rows, DDP_VSIZEOF(Ts)) and //
         detail::all_same_or_dyn(detail::get_dims<Ts...>::cols, DDP_VSIZEOF(Ts))),   //
        int>::type = 0>
auto sum(Ts const&... args) noexcept -> decltype(detail::sum_impl<(DDP_VSIZEOF(Ts) > 2)>::run(args...)) {
  index_t const all_rows[] = {args.rows()...};
  index_t const all_cols[] = {args.cols()...};
  for (index_t i = 0; i < DDP_VSIZEOF(Ts) - 1; ++i) {
    assert(all_rows[i + 1] == all_rows[i]);
    assert(all_cols[i + 1] == all_cols[i]);
  }

  return detail::sum_impl<(DDP_VSIZEOF(Ts) > 2)>::run(args...);
}

template <
    typename... Ts,
    typename std::enable_if<                  //
        ((DDP_VSIZEOF(Ts) >= 2) and           //
         detail::all_consecutive_same_or_dyn( //
             detail::get_dims<Ts...>::rows,   //
             detail::get_dims<Ts...>::cols,   //
             DDP_VSIZEOF(Ts))),
        int>::type = 0>
auto prod(Ts const&... args) noexcept -> decltype(detail::sum_impl<(DDP_VSIZEOF(Ts) > 2)>::run(args...)) {
  index_t const all_rows[] = {args.rows()...};
  index_t const all_cols[] = {args.cols()...};
  for (index_t i = 0; i < DDP_VSIZEOF(Ts) - 1; ++i) {
    assert(all_cols[i] == all_rows[i + 1]);
  }

  return detail::sum_impl<(DDP_VSIZEOF(Ts) > 2)>::run(args...);
}

template <typename T, typename T_ = typename std::remove_reference<T>::type>
auto as_mut_view(T&& mat) noexcept                              //
    -> view_t<                                                  //
        Eigen::Matrix<                                          //
            typename T_::Scalar,                                //
            T_::RowsAtCompileTime,                              //
            T_::ColsAtCompileTime,                              //
            T_::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor, //
            T_::MaxRowsAtCompileTime,                           //
            T_::MaxColsAtCompileTime                            //
            >> {
  assert(mat.innerStride() == 1);
  return {mat.data(), mat.rows(), mat.cols(), mat.outerStride()};
}

template <
    typename T,
    typename Out_Matrix = Eigen::Matrix<
        typename T::Scalar,
        Eigen::Dynamic,
        T::ColsAtCompileTime,
        T::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,
        Eigen::Dynamic,
        T::MaxColsAtCompileTime>>
auto dyn_rows(view_t<T> v) noexcept -> view_t<DDP_CONDITIONAL(std::is_const<T>::value, Out_Matrix const, Out_Matrix)> {
  return {v.data(), v.rows(), v.cols(), v.outerStride()};
}

template <
    typename T,
    typename Out_Matrix = Eigen::Matrix<
        typename T::Scalar,
        T::RowsAtCompileTime,
        Eigen::Dynamic,
        T::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,
        T::MaxRowsAtCompileTime,
        Eigen::Dynamic>>
auto dyn_cols(view_t<T> v) noexcept -> view_t<DDP_CONDITIONAL(std::is_const<T>::value, Out_Matrix const, Out_Matrix)> {
  return {v.data(), v.rows(), v.cols(), v.outerStride()};
}

template <typename T>
auto as_const_view(T const& mat) noexcept                      //
    -> view_t<                                                 //
        Eigen::Matrix<                                         //
            typename T::Scalar,                                //
            T::RowsAtCompileTime,                              //
            T::ColsAtCompileTime,                              //
            T::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor, //
            T::MaxRowsAtCompileTime,                           //
            T::MaxColsAtCompileTime                            //
            > const> {
  assert(mat.innerStride() == 1);
  return {mat.data(), mat.rows(), mat.cols(), mat.outerStride()};
}

// clang-format off
template <typename T> auto rows(T const& mat) -> row_kind<T> { return row_kind<T>{mat.rows()}; }
template <typename T> auto cols(T const& mat) -> col_kind<T> { return col_kind<T>{mat.rows()}; }
// clang-format on

} // namespace eigen
} // namespace ddp

#endif /* end of include guard ALL_HPP_VYM0WI7T */
