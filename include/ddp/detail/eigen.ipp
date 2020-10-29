#ifndef DDP_PINOCCHIO_EIGEN_IPP_RBS7TPFLS
#define DDP_PINOCCHIO_EIGEN_IPP_RBS7TPFLS

#include <Eigen/Cholesky>
#include "ddp/detail/eigen.hpp"
#include <new>

namespace ddp {
namespace eigen {
namespace detail {

template <typename T>
inplace_llt_result_t<T>::inplace_llt_result_t(inplace_llt_result_t const& other) noexcept {
  using llt = Eigen::LLT<Eigen::Ref<Eigen::Matrix<T, -1, -1>>>;
  new (buf) llt{other.inner()};
  m_success = other.m_success;
}

template <typename T>
auto llt_inplace_impl(matrix_view_t<T, dyn_index, dyn_index> out) -> inplace_llt_result_t<T> {

  using llt_type = typename inplace_llt_result_t<T>::llt_type;

  inplace_llt_result_t<T> inplace{};
  new (inplace.buf) llt_type{out};
  inplace.m_success = (inplace.inner().info() == Eigen::Success);
  return inplace;

  static_assert(alignof(llt_type) <= alignof(decltype(inplace)), "");
  static_assert(sizeof(llt_type) <= sizeof(decltype(inplace.buf)), "");
}

template <typename T>
void llt_solve_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    inplace_llt_result_t<T> const& llt,
    matrix_view_t<T const, dyn_index, dyn_index> b) {
  out = llt.inner().solve(b);
}

template <typename T>
void llt_solve_impl_rowmajor(
    matrix_view_t<T, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> out,
    inplace_llt_result_t<T> const& llt,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> b) {
  out = llt.inner().solve(b);
}

template <typename T>
void add_multiply_k_mat_mat_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    T const& k,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> left,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> right) {
  out.noalias() += k * left * right;
}

template <typename T>
void add_multiply_k_tmat_mat_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    T const& k,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> left,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> right) {
  out.noalias() += k * left.transpose() * right;
}

template <typename T>
void add_multiply_k_mat_tmat_impl(
    matrix_view_t<T, dyn_index, dyn_index> out,
    T const& k,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> left,
    matrix_view_t<T const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> right) {
  out.noalias() += k * left * right.transpose();
}

} // namespace detail
} // namespace eigen
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_EIGEN_IPP_RBS7TPFLS */
