#ifndef DDP_PINOCCHIO_EIGEN_HPP_4XSN07RYS
#define DDP_PINOCCHIO_EIGEN_HPP_4XSN07RYS

#include "utils.hpp"
#include <Eigen/Core>

namespace ddp {
namespace eigen {
template <typename T>
using dyn_vec_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <int Rows, int Cols>
struct default_options : std::integral_constant<
                             int,
                             ((Rows == 1 and Cols != 1) //
                                  ? Eigen::RowMajor
                                  : (Cols == 1 and Rows != 1) //
                                        ? (Eigen::ColMajor)
                                        : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION)> {};

template <typename T>
using view_t = Eigen::Map<T, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;

template <
    typename T,
    typename Rows,
    typename Cols,
    typename Max_Rows = Rows,
    typename Max_Cols = Cols,
    int Options = default_options<Rows::value_at_compile_time, Cols::value_at_compile_time>::value>
using matrix_t = Eigen::Matrix<
    T,
    Rows::value_at_compile_time,
    Cols::value_at_compile_time,
    Options,
    Max_Rows::value_at_compile_time,
    Max_Cols::value_at_compile_time>;

template <
    typename T,
    typename Rows,
    typename Cols,
    typename Max_Rows,
    typename Max_Cols,
    int Options = default_options<Rows::value_at_compile_time, Cols::value_at_compile_time>::value>
struct matrix_view {
  using type = view_t<matrix_t<T, Rows, Cols, Max_Rows, Max_Cols, Options>>;
};

template <typename T, typename Rows, typename Cols, int Options, typename Max_Rows, typename Max_Cols>
struct matrix_view<T const, Rows, Cols, Max_Rows, Max_Cols, Options> {
  using type = view_t<matrix_t<T, Rows, Cols, Max_Rows, Max_Cols, Options> const>;
};

template <
    typename T,
    typename Rows,
    typename Cols,
    typename Max_Rows = Rows,
    typename Max_Cols = Cols,
    int Options = default_options<Rows::value_at_compile_time, Cols::value_at_compile_time>::value>
using matrix_view_t = typename matrix_view<T, Rows, Cols, Max_Rows, Max_Cols, Options>::type;

template <typename T>
struct type_to_size
    : std::integral_constant<index_t, (T::known_at_compile_time ? Eigen::Dynamic : T::value_at_compile_time)> {};

template <index_t N>
using size_to_type = DDP_CONDITIONAL((N == Eigen::Dynamic), dyn_index, fix_index<N>);

template <
    typename T,
    typename Indexer,
    int Options =
        default_options<Indexer::row_kind::value_at_compile_time, Indexer::col_kind::value_at_compile_time>::value>
using matrix_from_idx_t = Eigen::Matrix<
    T,
    Indexer::row_kind::value_at_compile_time,
    Indexer::col_kind::value_at_compile_time,
    Options,
    Indexer::max_row_kind::value_at_compile_time,
    Indexer::max_col_kind::value_at_compile_time>;

template <typename T>
using row_kind = DDP_CONDITIONAL(T::RowsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::RowsAtCompileTime>);
template <typename T>
using col_kind = DDP_CONDITIONAL(T::ColsAtCompileTime == Eigen::Dynamic, dyn_index, fix_index<T::ColsAtCompileTime>);

template <typename T>
auto as_mut_view(T&& mat)                                                                   //
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
auto as_const_view(T const& mat)                               //
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
    typename Max_Rows = Rows,
    typename Max_Cols = Cols,
    int Options = default_options<Rows::value_at_compile_time, Cols::value_at_compile_time>::value>
auto make_matrix(Rows rows, Cols cols = {}, Max_Rows /*unused*/ = {}, Max_Cols /*unused*/ = {}) //
    -> matrix_t<T, Rows, Cols, Max_Rows, Max_Cols, Options> {
  DDP_ASSERT(rows.value() >= 0);
  DDP_ASSERT(cols.value() >= 0);
  matrix_t<T, Rows, Cols, Max_Rows, Max_Cols, Options> retval{rows.value(), cols.value()};
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

template <typename T>
struct into_view_t {
  T* data;
  index_t rows;
  index_t cols;
  index_t outer_stride;
  bool is_row_major;

  template <typename View>
  operator/* NOLINT(hicpp-explicit-conversions) */ View() const {
    if (View::RowsAtCompileTime != Eigen::Dynamic) {
      DDP_ASSERT(rows == View::RowsAtCompileTime);
    }
    if (View::ColsAtCompileTime != Eigen::Dynamic) {
      DDP_ASSERT(cols == View::ColsAtCompileTime);
    }
    if (View::MaxRowsAtCompileTime != Eigen::Dynamic) {
      DDP_ASSERT(rows <= View::MaxRowsAtCompileTime);
    }
    if (View::MaxColsAtCompileTime != Eigen::Dynamic) {
      DDP_ASSERT(cols <= View::MaxColsAtCompileTime);
    }
    DDP_ASSERT(is_row_major == View::IsRowMajor);
    return {data, rows, cols, outer_stride};
  }
};

template <typename T>
auto into_view(view_t<T> mat)
    -> into_view_t<DDP_CONDITIONAL(std::is_const<T>::value, typename T::Scalar const, typename T::Scalar)> {
  return {
      mat.data(),
      mat.rows(),
      mat.cols(),
      mat.outerStride(),
      T::IsRowMajor,
  };
}

namespace detail {

template <typename T>
struct inplace_llt_result_t;

template <typename T>
auto llt_inplace_impl(matrix_view_t<T, dyn_index, dyn_index> out) -> inplace_llt_result_t<T>;
template <typename T>
void llt_solve_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    inplace_llt_result_t<T> const& llt,
    matrix_view_t<T const, dyn_index, dyn_index> b);
template <typename T>
void llt_solve_impl_rowmajor(
    matrix_view_t<T, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> out,
    inplace_llt_result_t<T> const& llt,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> b);

template <typename T>
struct inplace_llt_result_t {
  ~inplace_llt_result_t() = default;
  inplace_llt_result_t(inplace_llt_result_t const&) noexcept;
  inplace_llt_result_t(inplace_llt_result_t&& other) noexcept
      : inplace_llt_result_t(static_cast<inplace_llt_result_t const&>(other)){};
  void operator=(inplace_llt_result_t const&) = delete;
  void operator=(inplace_llt_result_t&&) = delete;
  auto success() const -> bool { return m_success; }

private:
  inplace_llt_result_t() = default;
  friend auto detail::llt_inplace_impl<>(matrix_view_t<T, dyn_index, dyn_index> out) -> inplace_llt_result_t;
  friend void detail::llt_solve_impl<>(
      matrix_view_t<T, dyn_index, dyn_index> out,
      inplace_llt_result_t const& llt,
      matrix_view_t<T const, dyn_index, dyn_index> b);
  friend void llt_solve_impl_rowmajor<>(
      matrix_view_t<T, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> out,
      inplace_llt_result_t<T> const& llt,
      matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> b);

  using llt_type = Eigen::LLT<Eigen::Ref<Eigen::Matrix<T, -1, -1>>>;
  auto inner() const -> llt_type const& { return *reinterpret_cast<llt_type const*>(buf); }

  alignas(std::max_align_t) unsigned char buf[64 + sizeof(T)] = {};
  bool m_success = {};
};

template <typename T>
void add_multiply_k_mat_mat_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    T const& k,
    matrix_view_t<T const, dyn_index, dyn_index> left,
    matrix_view_t<T const, dyn_index, dyn_index> right);

template <typename T>
void add_multiply_k_tmat_mat_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    T const& k,
    matrix_view_t<T const, dyn_index, dyn_index> left,
    matrix_view_t<T const, dyn_index, dyn_index> right);

template <typename T>
void add_multiply_k_mat_tmat_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    T const& k,
    matrix_view_t<T const, dyn_index, dyn_index> left,
    matrix_view_t<T const, dyn_index, dyn_index> right);

} // namespace detail

template <typename In_Out>
auto llt_inplace(In_Out&& mat) -> detail::inplace_llt_result_t<typename std::remove_reference<In_Out>::type::Scalar> {
  return detail::llt_inplace_impl<                         //
      typename std::remove_reference<In_Out>::type::Scalar //
      >((into_view)((as_mut_view)(mat)));
}

template <typename T, typename In, typename Out>
void llt_solve(Out&& out, detail::inplace_llt_result_t<T> const& llt, In const& b) {
  static_assert(std::remove_reference<Out>::type::IsRowMajor == In::IsRowMajor, "");
  if (In::IsRowMajor) {
    return detail::llt_solve_impl_rowmajor<T>((into_view)((as_mut_view)(out)), llt, (into_view)((as_const_view)(b)));
  }
  return detail::llt_solve_impl<T>((into_view)((as_mut_view)(out)), llt, (into_view)((as_const_view)(b)));
}

template <typename Out, typename Left, typename Right>
void noalia_add_scal_mat_mat(Out&& out, typename Left::Scalar const& k, Left const& l, Right const& r) {
  static_assert(not Left::IsRowMajor, "");
  static_assert(not Right::IsRowMajor, "");
  detail::add_multiply_k_mat_mat_impl<typename Left::Scalar>(
      (into_view)((as_mut_view)(out)),
      k,
      (into_view)((as_const_view)(l)),
      (into_view)((as_const_view)(r)));
}

template <typename Out, typename Left, typename Right>
void noalia_add_scal_tmat_mat(Out&& out, typename Left::Scalar const& k, Left const& l, Right const& r) {
  static_assert(not Left::IsRowMajor, "");
  static_assert(not Right::IsRowMajor, "");
  detail::add_multiply_k_tmat_mat_impl<typename Left::Scalar>(
      (into_view)((as_mut_view)(out)),
      k,
      (into_view)((as_const_view)(l)),
      (into_view)((as_const_view)(r)));
}

template <typename Out, typename Left, typename Right>
void noalia_add_scal_mat_tmat(Out&& out, typename Left::Scalar const& k, Left const& l, Right const& r) {
  static_assert(not Left::IsRowMajor, "");
  static_assert(not Right::IsRowMajor, "");
  detail::add_multiply_k_mat_tmat_impl<typename Left::Scalar>(
      (into_view)((as_mut_view)(out)),
      k,
      (into_view)((as_const_view)(l)),
      (into_view)((as_const_view)(r)));
}

} // namespace eigen
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_EIGEN_HPP_4XSN07RYS */
