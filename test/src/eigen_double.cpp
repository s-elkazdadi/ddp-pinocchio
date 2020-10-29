#include "ddp/detail/eigen.ipp"

namespace ddp {
namespace eigen {
namespace detail {

template struct inplace_llt_result_t<double>;

template auto llt_inplace_impl(matrix_view_t<double, dyn_index, dyn_index> out) -> inplace_llt_result_t<double>;

template void llt_solve_impl(
    matrix_view_t<double, dyn_index, dyn_index> out,
    inplace_llt_result_t<double> const& llt,
    matrix_view_t<double const, dyn_index, dyn_index> b);
template void llt_solve_impl_rowmajor(
    matrix_view_t<double, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> out,
    inplace_llt_result_t<double> const& llt,
    matrix_view_t<double const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> b);

template void add_multiply_k_mat_mat_impl(
    matrix_view_t<double, dyn_index, dyn_index> out,
    double const& k,
    matrix_view_t<double const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> left,
    matrix_view_t<double const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> right);

template void add_multiply_k_tmat_mat_impl(
    matrix_view_t<double, dyn_index, dyn_index> out,
    double const& k,
    matrix_view_t<double const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> left,
    matrix_view_t<double const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> right);

template void add_multiply_k_mat_tmat_impl(
    matrix_view_t<double, dyn_index, dyn_index> out,
    double const& k,
    matrix_view_t<double const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::ColMajor> left,
    matrix_view_t<double const, dyn_index, dyn_index, dyn_index, dyn_index, Eigen::RowMajor> right);

} // namespace detail
} // namespace eigen
} // namespace ddp
