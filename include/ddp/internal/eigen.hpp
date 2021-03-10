#ifndef DDP_PINOCCHIO_EIGEN_HPP_BDOKZRTGS
#define DDP_PINOCCHIO_EIGEN_HPP_BDOKZRTGS

#include <Eigen/Core>
#include "ddp/internal/utils.hpp"
#include <sys/ioctl.h> //ioctl() and TIOCGWINSZ
#include <unistd.h>    // for STDOUT_FILENO

namespace ddp {
namespace eigen {

enum struct kind { colvec, colmat, rowvec, rowmat };

namespace internal {

template <typename T>
struct with_formatter {
  T const& val;
  fmt::formatter<T>& fmt;
};

template <typename T, typename OutIt>
auto format_impl(
    fmt::basic_format_context<OutIt, char>& fc,
    veg::i64 rows,
    veg::i64 cols,
    veg::fn_ref<T(veg::i64, veg::i64)> getter,
    fmt::formatter<T>& fmt) -> decltype(fc.out()) {
  auto out = fc.out();

  bool const colmat = cols == 1;
  bool const multi = (cols > 1) && (rows > 1);

  out = fmt::format_to(
      out,
      "[{}|{:>2}×{:<2}] {}",
      (rows == 1)   ? "row"
      : (cols == 1) ? "col"
                    : "mat",
      rows,
      cols,
      multi ? "{\n" : "{");

  using veg::i64;
  i64 line_len = multi ? 0
                       : veg::narrow<i64>(fmt::formatted_size(
                             "[xxx|{:>2}×{:<2}] x", rows, cols));

  char const* row_sep = "";
  char const* row_begin = colmat ? "" : "{";
  char const* row_end = colmat ? "" : "}";

  struct winsize size {};
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
  auto win_cols = size.ws_col;

  for (veg::i64 i = 0; i < rows; ++i) {
    char const* col_sep = "";

    out = fmt::format_to(out, "{}", row_sep);
    line_len += veg::narrow<i64>(fmt::formatted_size("{}", row_sep));

    if (line_len >= win_cols - 10) {
      *out++ = '\n';
      *out++ = ' ';
      line_len = 1;
    }
    out = fmt::format_to(out, "{}", row_begin);
    line_len += veg::narrow<i64>(fmt::formatted_size("{}", row_begin));

    for (veg::i64 j = 0; j < cols; ++j) {
      line_len += veg::narrow<i64>(fmt::formatted_size("{}", col_sep));
      out = fmt::format_to(out, "{}", col_sep);
      fc.advance_to(out);

      T v = getter(i, j);

      auto len = veg::narrow<i64>(
          fmt::formatted_size("{}", with_formatter<T>{v, fmt}));
      line_len += len;

      if (line_len >= win_cols - 10) {
        fmt::print("{}\n", "break");
        *out++ = '\n';
        *out++ = ' ';
        line_len = len + 1;
      }
      out = fmt::format_to(out, "{}", with_formatter<T>{v, fmt});

      col_sep = ", ";
    }
    line_len += veg::narrow<i64>(fmt::formatted_size("{}", row_end));
    out = fmt::format_to(out, "{}", row_end);
    row_sep = multi ? ",\n" : ", ";

    line_len = multi ? 0 : line_len;
  }
  out = fmt::format_to(out, "{}", multi ? "\n}" : "}");
  return out;
}

extern template auto ddp::eigen::internal::format_impl(
    fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
    veg::i64 rows,
    veg::i64 cols,
    veg::fn_ref<float(veg::i64, veg::i64)> getter,
    fmt::formatter<float>& fmt) -> decltype(fc.out());

extern template auto ddp::eigen::internal::format_impl(
    fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
    veg::i64 rows,
    veg::i64 cols,
    veg::fn_ref<double(veg::i64, veg::i64)> getter,
    fmt::formatter<double>& fmt) -> decltype(fc.out());

extern template auto ddp::eigen::internal::format_impl(
    fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
    veg::i64 rows,
    veg::i64 cols,
    veg::fn_ref<long double(veg::i64, veg::i64)> getter,
    fmt::formatter<long double>& fmt) -> decltype(fc.out());

template <typename T>
constexpr auto fail(T* p = nullptr) -> T {
  return *p;
}

template <typename T, kind K>
struct to_matrix;

template <typename T>
struct to_matrix<T, kind::colmat> {
  using type = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
};
template <typename T>
struct to_matrix<T, kind::colvec> {
  using type = Eigen::Matrix<T, -1, 1, Eigen::ColMajor>;
};
template <typename T>
struct to_matrix<T, kind::rowmat> {
  using type = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
};
template <typename T>
struct to_matrix<T, kind::rowvec> {
  using type = Eigen::Matrix<T, 1, -1, Eigen::RowMajor>;
};

template <bool B, typename T>
using add_const_if = veg::meta::conditional_t<B, T const, T>;

template <typename T>
struct kind_of
    : std::integral_constant<
          kind,
          !T::IsRowMajor
              ? (T::ColsAtCompileTime == 1 ? kind::colvec : kind::colmat)
              : (T::RowsAtCompileTime == 1 ? kind::rowvec : kind::rowmat)> {};

} // namespace internal

template <typename T, kind K>
using matrix = typename internal::to_matrix<T, K>::type;

template <typename T, kind K>
using view = Eigen::Map<
    internal::add_const_if<
        std::is_const<T>::value,
        typename internal::to_matrix<std::remove_cv_t<T>, K>::type>,
    Eigen::Unaligned,
    Eigen::OuterStride<-1>>;

static_assert(internal::kind_of<Eigen::VectorXd>::value == kind::colvec, "");
static_assert(internal::kind_of<Eigen::MatrixXd>::value == kind::colmat, "");

namespace internal {
template <template <typename, kind> class Tpl, kind K>
struct dyn_cast_impl;

template <kind K>
struct dyn_cast_impl<matrix, K> {
  template <typename T>
  static auto apply(T& mat) -> matrix<typename T::Scalar, K> {
    if (K == kind::colvec) {
      VEG_ASSERT(mat.cols() == 1);
    }
    return mat;
  }
};

template <>
struct dyn_cast_impl<view, kind::colvec> {
  template <typename T>
  static auto apply(T& mat)
      -> view<std::remove_pointer_t<decltype(mat.data())>, kind::colvec> {
    static_assert(!T::IsRowMajor, "");

    VEG_ASSERT_ALL_OF((mat.innerStride() == 1), (mat.cols() == 1));
    return {mat.data(), mat.rows(), 1, mat.outerStride()};
  }
};

template <>
struct dyn_cast_impl<view, kind::rowvec> {
  template <typename T>
  static auto apply(T& mat)
      -> view<std::remove_pointer_t<decltype(mat.data())>, kind::rowvec> {
    static_assert(T::IsRowMajor, "");

    VEG_ASSERT_ALL_OF((mat.innerStride() == 1), (mat.rows() == 1));
    return {mat.data(), 1, mat.cols(), mat.outerStride()};
  }
};

template <>
struct dyn_cast_impl<view, kind::colmat> {
  template <typename T>
  static auto apply(T& mat)
      -> view<std::remove_pointer_t<decltype(mat.data())>, kind::colmat> {
    static_assert(!T::IsRowMajor, "");

    VEG_ASSERT(mat.innerStride() == 1);
    return {mat.data(), mat.rows(), mat.cols(), mat.outerStride()};
  }
};
} // namespace internal

template <template <typename, kind> class Tpl, kind K, typename T>
auto dyn_cast(T&& mat) {
  return internal::dyn_cast_impl<Tpl, K>::apply(mat);
}

template <typename T>
auto as_mut(T&& mat) {
  return internal::dyn_cast_impl<
      view,
      internal::kind_of<meta::remove_cvref_t<T>>::value>::apply(mat);
}

template <typename T>
auto as_const(T const& mat) {
  return internal::dyn_cast_impl<view, internal::kind_of<T>::value>::apply(mat);
}

template <
    typename T,
    kind K = internal::kind_of<meta::remove_cvref_t<T>>::value,
    typename View =
        view<std::remove_pointer_t<decltype(__VEG_DECLVAL(T).data())>, K>>

VEG_NODISCARD auto split_at_row(T&& mat, i64 row) -> tuple<View, View> {
  static_assert(
      internal::kind_of<meta::remove_cvref_t<T>>::value != kind::rowvec, "");
  VEG_ASSERT_ALL_OF((row >= 0), (row <= mat.rows()));
  return {
      elems,
      View{mat.data(), row, mat.cols(), mat.outerStride()},
      View{
          veg::mem::addressof(mat.coeffRef(row, 0)),
          mat.rows() - row,
          mat.cols(),
          mat.outerStride()},
  };
}

template <
    typename T,
    kind K = internal::kind_of<meta::remove_cvref_t<T>>::value,
    typename View =
        view<std::remove_pointer_t<decltype(__VEG_DECLVAL(T).data())>, K>>

VEG_NODISCARD auto split_at_col(T&& mat, i64 col) -> tuple<View, View> {
  static_assert(
      internal::kind_of<meta::remove_cvref_t<T>>::value != kind::colvec, "");
  VEG_ASSERT_ALL_OF((col >= 0), (col <= mat.cols()));
  return {
      elems,
      View{mat.data(), mat.rows(), col, mat.outerStride()},
      View{
          veg::mem::addressof(mat.coeffRef(0, col)),
          mat.rows(),
          mat.cols() - col,
          mat.outerStride()},
  };
}

template <
    typename T,
    typename View = view<
        std::remove_pointer_t<decltype(__VEG_DECLVAL(T).data())>,
        kind::colmat>>

VEG_NODISCARD auto split_at(T&& mat, i64 row, i64 col)
    -> tuple<View, View, View, View> {
  static_assert(
      internal::kind_of<meta::remove_cvref_t<T>>::value == kind::colmat, "");
  VEG_ASSERT_ALL_OF( //
      (row >= 0),
      (row <= mat.rows()),
      (col >= 0),
      (col <= mat.cols()));
  auto const os = mat.outerStride();
  return {
      elems,
      View{mat.data(), row, col, os},
      View{
          veg::mem::addressof(mat.coeffRef(0, col)), row, mat.cols() - col, os},
      View{
          veg::mem::addressof(mat.coeffRef(row, 0)), mat.rows() - row, col, os},
      View{
          veg::mem::addressof(mat.coeffRef(row, col)),
          mat.rows() - row,
          mat.cols() - col,
          os},
  };
}

namespace internal {

inline auto aliases_impl(
    i64 size,
    void const* ptr1_,
    i64 inner_stride1,
    i64 outer_stride1,
    i64 inner_size1,
    i64 outer_size1,
    void const* ptr2_,
    i64 inner_stride2,
    i64 outer_stride2,
    i64 inner_size2,
    i64 outer_size2) -> bool {

  if (inner_size1 == 0 || //
      outer_size1 == 0 || //
      inner_size2 == 0 || //
      outer_size2 == 0) {
    return false;
  }

  using std::uintptr_t;
  auto begin1 = uintptr_t(ptr1_);
  auto begin2 = uintptr_t(ptr2_);

  auto before_end1 = uintptr_t(
      static_cast<char const*>(ptr1_) +
      size * (inner_stride1 * (inner_size1 - 1) +
              outer_stride1 * (outer_size1 - 1)));

  auto before_end2 = uintptr_t(
      static_cast<char const*>(ptr2_) +
      size * (inner_stride2 * (inner_size2 - 1) +
              outer_stride2 * (outer_size2 - 1)));

  return !(
      ((begin1 < begin2) && (before_end1 < begin2)) || //
      ((begin2 < begin1) && (before_end2 < begin1)));
}

template <typename T, typename U>
auto aliases_impl_2(T const& t, U const& u) -> bool {
  static_assert(
      std::is_same<typename T::Scalar, typename U::Scalar>::value,
      "no type mixing please v_v");
  return internal::aliases_impl(
      sizeof(typename T::Scalar),
      t.data(),
      t.innerStride(),
      t.outerStride(),
      t.innerSize(),
      t.outerSize(),
      u.data(),
      u.innerStride(),
      u.outerStride(),
      u.innerSize(),
      u.outerSize());
}

} // namespace internal

template <typename T, typename... Ts>
auto aliases(T const& t, Ts const&... ts) -> bool {
  return veg::meta::any_of({internal::aliases_impl_2(t, ts)...});
}

template <typename T>
auto slice_to_vec(T& s) -> view<
    meta::remove_pointer_t<decltype(__VEG_DECLVAL(T&).data())>,
    kind::colvec> {
  auto size = veg::narrow<i64>(s.size());
  return {s.data(), size, 1, size};
}
template <typename T>
auto slice_to_mat(T& s, i64 nrows, i64 ncols) -> view<
    meta::remove_pointer_t<decltype(__VEG_DECLVAL(T&).data())>,
    kind::colmat> {
  VEG_DEBUG_ASSERT(nrows * ncols == s.size());
  return {s.data(), nrows, ncols, nrows};
}

#if 1

template <typename Out, typename In>
void assign(Out&& out, In const& rhs) {
  out = eigen::as_const(rhs);
}
template <typename Out, typename InL, typename InR>
void add_to(Out&& out, InL const& lhs, InR const& rhs) {
  out = eigen::as_const(lhs).operator+(eigen::as_const(rhs));
}
template <typename Out, typename InL, typename InR>
void sub_to(Out&& out, InL const& lhs, InR const& rhs) {
  out = eigen::as_const(lhs).operator-(eigen::as_const(rhs));
}
template <typename Out, typename InL, typename InR>
void mul_scalar_add_to(Out&& out, InL const& lhs, InR const& rhs) {
  out.noalias() += eigen::as_const(rhs).operator*(lhs);
}
template <typename Out, typename InL, typename InR>
void mul_scalar_to(Out&& out, InL const& lhs, InR const& rhs) {
  out = eigen::as_const(rhs).operator*(lhs);
}
template <typename Out, typename InL, typename InR>
void mul_add_to_noalias(Out&& out, InL const& lhs, InR const& rhs) {
  out.noalias() += eigen::as_const(lhs).operator*(eigen::as_const(rhs));
}
template <typename Out, typename InL, typename InR>
void tmul_add_to_noalias(Out&& out, InL const& lhs, InR const& rhs) {
  out.noalias() +=
      eigen::as_const(lhs.transpose()).operator*(eigen::as_const(rhs));
}
template <typename In>
auto dot(In const& lhs, In const& rhs) -> typename In::Scalar {
  return (eigen::as_const(lhs.transpose()).operator*(eigen::as_const(rhs)))[0];
}
template <typename Out>
void add_identity(
    Out&& out, typename meta::remove_cvref_t<Out>::Scalar const& factor = 1) {
  i64 const small_dim = out.rows() < out.cols() ? out.rows() : out.cols();
  for (i64 i = 0; i < small_dim; ++i) {
    out(i, i) += factor;
  }
}

#else

template <typename Out, typename In>
void assign(Out&& out, In const& rhs);
template <typename Out, typename InL, typename InR>
void add_to(Out&& out, InL const& lhs, InR const& rhs);
template <typename Out, typename InL, typename InR>
void sub_to(Out&& out, InL const& lhs, InR const& rhs);
template <typename Out, typename InL, typename InR>
void mul_scalar_add_to(Out&& out, InL const& lhs, InR const& rhs);
template <typename Out, typename InL, typename InR>
void mul_scalar_to(Out&& out, InL const& lhs, InR const& rhs);
template <typename Out, typename InL, typename InR>
void mul_add_to_noalias(Out&& out, InL const& lhs, InR const& rhs);
template <typename Out, typename InL, typename InR>
void tmul_add_to_noalias(Out&& out, InL const& lhs, InR const& rhs);
template <typename Out>
void add_identity(
    Out&& out, typename meta::remove_cvref_t<Out>::Scalar const& factor = 1);
template <typename In>
auto dot(In const& lhs, In const& rhs) -> typename In::Scalar;

#endif

} // namespace eigen

namespace {
constexpr auto const& colmat =
    std::integral_constant<eigen::kind, eigen::kind::colmat>::value;
constexpr auto const& colvec =
    std::integral_constant<eigen::kind, eigen::kind::colvec>::value;
constexpr auto const& rowmat =
    std::integral_constant<eigen::kind, eigen::kind::rowmat>::value;
constexpr auto const& rowvec =
    std::integral_constant<eigen::kind, eigen::kind::rowvec>::value;
} // namespace
using eigen::matrix;
using eigen::view;

} // namespace ddp

template <typename T, typename CharT>
struct fmt::formatter<ddp::eigen::internal::with_formatter<T>, CharT> {
  auto parse(fmt::basic_format_parse_context<CharT>& pc) { return pc.begin(); }
  template <typename OutIt>
  auto format(
      ddp::eigen::internal::with_formatter<T> val,
      fmt::basic_format_context<OutIt, CharT>& fc) {
    return val.fmt.format(val.val, fc);
  }
};

template <typename Matrix>
struct fmt::formatter<
    Matrix,
    char,
    veg::meta::enable_if_t<
        std::is_base_of<Eigen::MatrixBase<Matrix>, Matrix>::value>>
    : fmt::formatter<typename Matrix::Scalar> {

  using scalar = veg::meta::remove_cv_t<typename Matrix::Scalar>;

  template <typename OutIt>
  auto format(Matrix const& mat, fmt::basic_format_context<OutIt, char>& fc)
      -> decltype(fc.out()) {
    return ddp::eigen::internal::format_impl<scalar>(
        fc,
        mat.rows(),
        mat.cols(),
        [&](veg::i64 i, veg::i64 j) { return mat(i, j); },
        *this);
  }
};

#endif /* end of include guard DDP_PINOCCHIO_EIGEN_HPP_BDOKZRTGS */
