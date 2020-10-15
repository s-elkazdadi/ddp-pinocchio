#ifndef ALL_HPP_VYM0WI7T
#define ALL_HPP_VYM0WI7T

#include <cstdint>
#include <cstddef>

#ifdef NDEBUG
#undef NDEBUG
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

#define DDP_DECLVAL(...) static_cast<__VA_ARGS__ (*)() noexcept>(nullptr)()
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

#include "ddp/detail/assertions.hpp"

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

namespace gsl {
template <typename T>
using owner = T;
} // namespace gsl

namespace ddp {

[[noreturn]] void fast_fail(fmt::string_view message) noexcept;
void print_msg(fmt::string_view message) noexcept;

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
  auto operator=(chronometer_t &&) -> chronometer_t& = delete;

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

  constexpr dyn_index() noexcept : m_value{} {}
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
  explicit constexpr fix_index(index_t value) noexcept {
#if __cplusplus >= 201402L
    DDP_ASSERT(value == N);
#endif
  }
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
    static constexpr void assertion(N n, M m) noexcept { DDP_ASSERT(Run_Time_Cond); };                                 \
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

template <
    typename T,
    typename Rows,
    typename Cols,
    int Options = Eigen::ColMajor,
    typename Max_Rows = Rows,
    typename Max_Cols = Cols>
using matrix_t = Eigen::Matrix<
    T,
    Rows::value_at_compile_time,
    Cols::value_at_compile_time,
    Options,
    Max_Rows::value_at_compile_time,
    Max_Cols::value_at_compile_time>;

template <typename T, typename Rows, typename Cols, int Options, typename Max_Rows, typename Max_Cols>
struct matrix_view {
  using type = view_t<Eigen::Matrix<
      T,
      Rows::value_at_compile_time,
      Cols::value_at_compile_time,
      Options,
      Max_Rows::value_at_compile_time,
      Max_Cols::value_at_compile_time>>;
};

template <typename T, typename Rows, typename Cols, int Options, typename Max_Rows, typename Max_Cols>
struct matrix_view<T const, Rows, Cols, Options, Max_Rows, Max_Cols> {
  using type = view_t<Eigen::Matrix<
      T,
      Rows::value_at_compile_time,
      Cols::value_at_compile_time,
      Options,
      Max_Rows::value_at_compile_time,
      Max_Cols::value_at_compile_time> const>;
};

template <
    typename T,
    typename Rows,
    typename Cols,
    int Options = Eigen::ColMajor,
    typename Max_Rows = Rows,
    typename Max_Cols = Cols>
using matrix_view_t = typename matrix_view<T, Rows, Cols, Options, Max_Rows, Max_Cols>::type;

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

inline auto n_suffix(index_t n) -> fmt::string_view {
  DDP_ASSERT(n >= 0);
  if ((n % 100) / 10 == 1) {
    return "th";
  }
  switch (n % 10) {
  case 1:
    return "st";
  case 2:
    return "nd";
  case 3:
    return "rd";
  default:
    return "th";
  }
}

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
    DDP_ASSERT_MSG_ALL_OF(
        (fmt::format("{}{} and {}{} operands have mismatching rows", i, n_suffix(i), i + 1, n_suffix(i + 1)),
         all_rows[i + 1] == all_rows[i]),
        (fmt::format("{}{} and {}{} operands have mismatching cols", i, n_suffix(i), i + 1, n_suffix(i + 1)),
         all_cols[i + 1] == all_cols[i]));
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
    DDP_ASSERT_MSG(
        fmt::format(
            "columns of {}{} operand do not match rows of {}{} operand",
            i,
            n_suffix(i),
            i + 1,
            n_suffix(i + 1)),
        all_cols[i] == all_rows[i + 1]);
  }

  return detail::sum_impl<(DDP_VSIZEOF(Ts) > 2)>::run(args...);
}

#define DDP_EIGEN_STORAGE_ORDER(...)                                                                                   \
  (__VA_ARGS__::ColsAtCompileTime ==                                                                                   \
   1) /**************************************************************************************************************/ \
      ? Eigen::ColMajor                                                                                                \
      : (__VA_ARGS__::RowsAtCompileTime ==                                                                             \
                 1 /******************************************************************************************/        \
             ? Eigen::RowMajor                                                                                         \
             : __VA_ARGS__::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor)

template <typename T>
auto as_mut_view(T&& mat) noexcept                                                          //
    -> view_t<                                                                              //
        Eigen::Matrix<                                                                      //
            typename std::remove_reference<T>::type::Scalar,                                //
            std::remove_reference<T>::type::RowsAtCompileTime,                              //
            std::remove_reference<T>::type::ColsAtCompileTime,                              //
            std::remove_reference<T>::type::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor, //
            std::remove_reference<T>::type::MaxRowsAtCompileTime,                           //
            std::remove_reference<T>::type::MaxColsAtCompileTime                            //
            >> {
  DDP_ASSERT(mat.innerStride() == 1);
  return {mat.data(), mat.rows(), mat.cols(), mat.outerStride()};
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
  DDP_ASSERT(mat.innerStride() == 1);
  return {mat.data(), mat.rows(), mat.cols(), mat.outerStride()};
}

template <
    typename T,
    typename Rows,
    typename Cols = fix_index<1>,
    int Options = Rows::value_at_compile_time != 1
                      ? Eigen::ColMajor
                      : Cols::value_at_compile_time == 1 ? Eigen::ColMajor : Eigen::RowMajor,
    typename Max_Rows = Rows,
    typename Max_Cols = Cols>
auto make_matrix(Rows rows, Cols cols = {}, Max_Rows = {}, Max_Cols = {}) //
    -> Eigen::Matrix<                                                     //
        T,                                                                //
        Rows::value_at_compile_time,                                      //
        Cols::value_at_compile_time,                                      //
        Options,                                                          //
        Max_Rows::value_at_compile_time,                                  //
        Max_Cols::value_at_compile_time                                   //
        > {
  DDP_ASSERT(rows.value() > 0);
  DDP_ASSERT(cols.value() > 0);
  Eigen::Matrix<                       //
      T,                               //
      Rows::value_at_compile_time,     //
      Cols::value_at_compile_time,     //
      Options,                         //
      Max_Rows::value_at_compile_time, //
      Max_Cols::value_at_compile_time  //
      >
      retval{rows.value(), cols.value()};
  retval.setZero();
  return retval;
}

template <typename T>
auto rows_c(T const& mat)
    -> DDP_CONDITIONAL(T::RowsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::RowsAtCompileTime>) {
  return DDP_CONDITIONAL(T::RowsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::RowsAtCompileTime>){
      mat.rows()};
}

template <typename T>
auto cols_c(T const& mat)
    -> DDP_CONDITIONAL(T::RowsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::RowsAtCompileTime>) {
  return DDP_CONDITIONAL(T::RowsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::RowsAtCompileTime>){
      mat.cols()};
}

template <index_t N, index_t I>
struct eigen_diff {
  static constexpr index_t value = (N == Eigen::Dynamic or I == Eigen::Dynamic) ? Eigen::Dynamic : N - I;
};

template <typename T, typename Idx>
auto split_at_row(T const& mat, Idx idx) DDP_PRECOND_DECLTYPE_AUTO(
    ({
      DDP_DEBUG_ASSERT_MSG_ALL_OF(
          ("row index must be within bounds", idx.value() >= 0),
          ("row index must be within bounds", idx.value() <= mat.rows()));
    }),
    ddp::detail::make_tuple(
        eigen::as_const_view(mat.template topRows<Idx::value_at_compile_time>(idx.value())),
        eigen::as_const_view(
            mat.template bottomRows<eigen_diff<T::RowsAtCompileTime, Idx::value_at_compile_time>::value>(
                mat.rows() - idx.value()))));

template <typename T, typename Idx>
auto split_at_row_mut(T&& mat, Idx idx) DDP_PRECOND_DECLTYPE_AUTO(
    ({
      DDP_DEBUG_ASSERT_MSG_ALL_OF(
          ("row index must be within bounds", idx.value() >= 0),
          ("row index must be within bounds", idx.value() <= mat.rows()));
    }),
    ddp::detail::make_tuple(
        eigen::as_mut_view(mat.template topRows<Idx::value_at_compile_time>(idx.value())),
        eigen::as_mut_view(
            mat.template bottomRows<
                eigen_diff<std::remove_reference<T>::type::RowsAtCompileTime, Idx::value_at_compile_time>::value>(
                mat.rows() - idx.value()))));

template <typename T, typename Idx>
auto split_at_col(T const& mat, Idx idx) DDP_PRECOND_DECLTYPE_AUTO(
    ({
      DDP_DEBUG_ASSERT_MSG_ALL_OF(
          ("col index must be within bounds", idx.value() >= 0),
          ("col index must be within bounds", idx.value() <= mat.cols()));
    }),
    ddp::detail::make_tuple(
        eigen::as_const_view(mat.template leftCols<Idx::value_at_compile_time>(idx.value())),
        eigen::as_const_view(
            mat.template rightCols<eigen_diff<T::ColsAtCompileTime, Idx::value_at_compile_time>::value>(
                mat.cols() - idx.value()))));

template <typename T, typename Idx>
auto split_at_col_mut(T&& mat, Idx idx) DDP_PRECOND_DECLTYPE_AUTO(
    ({
      DDP_DEBUG_ASSERT_MSG_ALL_OF(
          ("col index must be within bounds", idx.value() >= 0),
          ("col index must be within bounds", idx.value() <= mat.cols()));
    }),
    ddp::detail::make_tuple(
        eigen::as_mut_view(mat.template leftCols<Idx::value_at_compile_time>(idx.value())),
        eigen::as_mut_view(
            mat.template rightCols<
                eigen_diff<std::remove_reference<T>::type::ColsAtCompileTime, Idx::value_at_compile_time>::value>(
                mat.cols() - idx.value()))));

template <typename T, typename Row_Idx, typename Col_Idx>
auto split_at(T const& mat, Row_Idx row_idx, Col_Idx col_idx) DDP_PRECOND_DECLTYPE_AUTO(
    ({
      DDP_DEBUG_ASSERT_MSG_ALL_OF(
          ("row index must be within bounds", row_idx.value() >= 0),
          ("row index must be within bounds", row_idx.value() <= mat.rows()),
          ("col index must be within bounds", col_idx.value() >= 0),
          ("col index must be within bounds", col_idx.value() <= mat.cols()));
    }),
    ddp::detail::make_tuple(
        eigen::as_const_view(mat.template topLeftCorner<     //
                             Row_Idx::value_at_compile_time, //
                             Col_Idx::value_at_compile_time  //
                             >(row_idx.value(), col_idx.value())),

        eigen::as_const_view(mat.template topRightCorner<                                            //
                             Row_Idx::value_at_compile_time,                                         //
                             eigen_diff<T::ColsAtCompileTime, Col_Idx::value_at_compile_time>::value //
                             >(row_idx.value(), mat.cols() - col_idx.value())),

        eigen::as_const_view(mat.template bottomLeftCorner<                                           //
                             eigen_diff<T::RowsAtCompileTime, Row_Idx::value_at_compile_time>::value, //
                             Col_Idx::value_at_compile_time                                           //
                             >(mat.rows() - row_idx.value(), col_idx.value())),

        eigen::as_const_view(mat.template bottomRightCorner<                                          //
                             eigen_diff<T::RowsAtCompileTime, Row_Idx::value_at_compile_time>::value, //
                             eigen_diff<T::ColsAtCompileTime, Col_Idx::value_at_compile_time>::value  //
                             >(mat.rows() - row_idx.value(), mat.cols() - col_idx.value()))

            ));

template <typename T, typename Row_Idx, typename Col_Idx>
auto split_at_mut(T&& mat, Row_Idx row_idx, Col_Idx col_idx) DDP_PRECOND_DECLTYPE_AUTO(
    ({
      DDP_DEBUG_ASSERT_MSG_ALL_OF(
          ("row index must be within bounds", row_idx.value() >= 0),
          ("row index must be within bounds", row_idx.value() <= mat.rows()),
          ("col index must be within bounds", col_idx.value() >= 0),
          ("col index must be within bounds", col_idx.value() <= mat.cols()));
    }),
    ddp::detail::make_tuple(
        eigen::as_mut_view(mat.template topLeftCorner<     //
                           Row_Idx::value_at_compile_time, //
                           Col_Idx::value_at_compile_time  //
                           >(row_idx.value(), col_idx.value())),

        eigen::as_mut_view(
            mat.template topRightCorner<                                                                             //
                Row_Idx::value_at_compile_time,                                                                      //
                eigen_diff<std::remove_reference<T>::type::ColsAtCompileTime, Col_Idx::value_at_compile_time>::value //
                >(row_idx.value(), mat.cols() - col_idx.value())),

        eigen::as_mut_view(
            mat.template bottomLeftCorner<                                                                            //
                eigen_diff<std::remove_reference<T>::type::RowsAtCompileTime, Row_Idx::value_at_compile_time>::value, //
                Col_Idx::value_at_compile_time                                                                        //
                >(mat.rows() - row_idx.value(), col_idx.value())),

        eigen::as_mut_view(
            mat.template bottomRightCorner<                                                                           //
                eigen_diff<std::remove_reference<T>::type::RowsAtCompileTime, Row_Idx::value_at_compile_time>::value, //
                eigen_diff<std::remove_reference<T>::type::ColsAtCompileTime, Col_Idx::value_at_compile_time>::value  //
                >(mat.rows() - row_idx.value(), mat.cols() - col_idx.value()))

            ));

// clang-format off
template <typename T> auto rows(T const& mat) -> row_kind<T> { return row_kind<T>{mat.rows()}; }
template <typename T> auto cols(T const& mat) -> col_kind<T> { return col_kind<T>{mat.rows()}; }
// clang-format on

} // namespace eigen
} // namespace ddp

#endif /* end of include guard ALL_HPP_VYM0WI7T */
