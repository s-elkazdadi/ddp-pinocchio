#ifndef SECOND_ORDER_FINITE_DIFF_HPP_5JNUIJG6
#define SECOND_ORDER_FINITE_DIFF_HPP_5JNUIJG6

#include "ddp/detail/tensor.hpp"

namespace ddp {

template <
    typename Scalar,
    typename X,
    typename X_Max,
    typename DX,
    typename DX_Max,
    typename U,
    typename U_Max,
    typename DU,
    typename DU_Max,
    typename Out,
    typename Out_Max,
    typename DOut,
    typename DOut_Max>
struct dimensions_t {
  using _1 = fix_index<1>;

  using x_t = X;
  using x_max_t = X_Max;
  using dx_t = DX;
  using dx_max_t = DX_Max;
  using u_t = U;
  using u_max_t = U_Max;
  using du_t = DU;
  using du_max_t = DU_Max;
  using out_t = Out;
  using out_max_t = Out_Max;
  using dout_t = DOut;
  using dout_max_t = DOut_Max;

  using x_mut = eigen::matrix_view_t<Scalar, X, _1, X_Max, _1>;
  using x_const = eigen::matrix_view_t<Scalar const, X, _1, X_Max, _1>;
  using dx_mut = eigen::matrix_view_t<Scalar, DX, _1, DX_Max, _1>;
  using dx_const = eigen::matrix_view_t<Scalar const, DX, _1, DX_Max, _1>;

  using u_mut = eigen::matrix_view_t<Scalar, U, _1, U_Max, _1>;
  using u_const = eigen::matrix_view_t<Scalar const, U, _1, U_Max, _1>;
  using du_mut = eigen::matrix_view_t<Scalar, DU, _1, DU_Max, _1>;
  using du_const = eigen::matrix_view_t<Scalar const, DU, _1, DU_Max, _1>;

  using out_mut = eigen::matrix_view_t<Scalar, Out, _1, Out_Max, _1>;
  using out_const = eigen::matrix_view_t<Scalar const, Out, _1, Out_Max, _1>;
  using dout_const = eigen::matrix_view_t<Scalar const, DOut, _1, DOut_Max, _1>;
  using dout_mut = eigen::matrix_view_t<Scalar, DOut, _1, DOut_Max, _1>;

  using xt_mut = eigen::matrix_view_t<Scalar, _1, DX, _1, DX_Max>;
  using ut_mut = eigen::matrix_view_t<Scalar, _1, DU, _1, DU_Max>;
  using xx_mut = eigen::matrix_view_t<Scalar, DX, DX, DX_Max, DX_Max>;
  using ux_mut = eigen::matrix_view_t<Scalar, DU, DX, DU_Max, DX_Max>;
  using uu_mut = eigen::matrix_view_t<Scalar, DU, DU, DU_Max, DU_Max>;

  using out_x_mut = eigen::matrix_view_t<Scalar, DOut, DX, DOut_Max, DX_Max>;
  using out_u_mut = eigen::matrix_view_t<Scalar, DOut, DU, DOut_Max, DU_Max>;
  using out_xx_mut = tensor::tensor_view_t<Scalar, DOut, DX, DX>;
  using out_ux_mut = tensor::tensor_view_t<Scalar, DOut, DU, DX>;
  using out_uu_mut = tensor::tensor_view_t<Scalar, DOut, DU, DU>;
};

template <typename Scalar, typename X, typename DX, typename U, typename DU, typename Out, typename DOut>
using dimensions_from_idx_t = dimensions_t<
    Scalar,
    typename X::row_kind,
    typename X::max_row_kind,
    typename DX::row_kind,
    typename DX::max_row_kind,
    typename U::row_kind,
    typename U::max_row_kind,
    typename DU::row_kind,
    typename DU::max_row_kind,
    typename Out::row_kind,
    typename Out::max_row_kind,
    typename DOut::row_kind,
    typename DOut::max_row_kind>;

template <typename Dims>
using x_mut = typename Dims::x_mut;
template <typename Dims>
using x_const = typename Dims::x_const;
template <typename Dims>
using dx_const = typename Dims::dx_const;
template <typename Dims>
using dx_mut = typename Dims::dx_mut;

template <typename Dims>
using u_mut = typename Dims::u_mut;
template <typename Dims>
using u_const = typename Dims::u_const;
template <typename Dims>
using du_const = typename Dims::du_const;

template <typename Dims>
using out_mut = typename Dims::out_mut;
template <typename Dims>
using out_const = typename Dims::out_const;
template <typename Dims>
using dout_mut = typename Dims::dout_mut;

template <typename Dims>
using xt_mut = typename Dims::xt_mut;
template <typename Dims>
using ut_mut = typename Dims::ut_mut;
template <typename Dims>
using xx_mut = typename Dims::xx_mut;
template <typename Dims>
using ux_mut = typename Dims::ux_mut;
template <typename Dims>
using uu_mut = typename Dims::uu_mut;

template <typename Dims>
using out_x_mut = typename Dims::out_x_mut;
template <typename Dims>
using out_u_mut = typename Dims::out_u_mut;
template <typename Dims>
using out_xx_mut = typename Dims::out_xx_mut;
template <typename Dims>
using out_ux_mut = typename Dims::out_ux_mut;
template <typename Dims>
using out_uu_mut = typename Dims::out_uu_mut;

template <typename Fn>
struct finite_diff_hessian_compute {
  using scalar_t = typename Fn::scalar_t;
  using dims = typename Fn::dims;
  using key = typename Fn::key;

  auto second_order_deriv_1(
      out_xx_mut<dims> out_xx,
      out_ux_mut<dims> out_ux,
      out_uu_mut<dims> out_uu,
      out_x_mut<dims> out_x,
      out_u_mut<dims> out_u,
      out_mut<dims> out,
      index_t t,
      x_const<dims> x,
      u_const<dims> u,
      key k) const -> key {

    DDP_ASSERT_MSG_ALL_OF(
        ("non commutative groups are not supported", out_x.rows() == out.rows()),
        ("non commutative groups are not supported", out_x.cols() == x.rows()),
        ("non commutative groups are not supported", out_u.cols() == u.rows()));

    auto no = out_x.rows();
    auto nx = out_x.cols();
    auto nu = out_u.cols();

    k = fn.first_order_deriv(out_x, out_u, out, t, x, u, DDP_MOVE(k));
    // compute second derivatives
    {
      auto fx_ = out_x.eval();
      auto fu_ = out_u.eval();

      auto _out_ = out.eval();
      auto _x_ = x.eval();
      auto _u_ = u.eval();

      auto _dx = dx_const<dims>::Zero(nx).eval();
      auto _du = du_const<dims>::Zero(nu).eval();

      auto out_ = eigen::as_mut_view(_out_);
      auto x_ = eigen::as_mut_view(_x_);
      auto u_ = eigen::as_mut_view(_u_);
      auto dx = eigen::as_mut_view(_dx);
      auto du = eigen::as_mut_view(_du);

      using std::sqrt;
      scalar_t eps = sqrt(std::numeric_limits<scalar_t>::epsilon());

      for (index_t i = 0; i < nx + nu; ++i) {
        bool at_x = (i < nx);
        index_t idx = at_x ? i : (i - nx);

        scalar_t& in_var = at_x ? dx[idx] : du[idx];

        in_var = eps;

        fn.integrate_x(x_, x, eigen::as_const_view(dx));
        fn.integrate_u(u_, t, u, eigen::as_const_view(du));

        k = fn.first_order_deriv(
            eigen::as_mut_view(fx_),
            eigen::as_mut_view(fu_),
            out_,
            t,
            eigen::as_const_view(x_),
            eigen::as_const_view(u_),
            DDP_MOVE(k));

        if (at_x) {

          for (index_t out_i = 0; out_i < no; ++out_i) {
            for (index_t j = 0; j < nx; ++j) {
              out_xx(out_i, j, idx) = (fx_(out_i, j) - out_x(out_i, j)) / eps;
            }
            for (index_t j = 0; j < nu; ++j) {
              out_ux(out_i, j, idx) = (fu_(out_i, j) - out_u(out_i, j)) / eps;
            }
          }

        } else {
          for (index_t out_i = 0; out_i < no; ++out_i) {
            for (index_t j = 0; j < nu; ++j) {
              out_uu(out_i, j, idx) = (fu_(out_i, j) - out_u(out_i, j)) / eps;
            }
          }
        }

        in_var = 0;
      }
    }
    return k;
  }

  auto second_order_deriv_2(
      out_xx_mut<dims> out_xx,
      out_ux_mut<dims> out_ux,
      out_uu_mut<dims> out_uu,
      out_x_mut<dims> out_x,
      out_u_mut<dims> out_u,
      out_mut<dims> out,
      index_t t,
      x_const<dims> x,
      u_const<dims> u,
      key k) const -> key {

    auto no = out_x.rows();
    auto nx = out_x.cols();
    auto nu = out_u.cols();

    k = fn.first_order_deriv(out_x, out_u, out, t, x, u, DDP_MOVE(k));
    // compute second derivatives
    // dx.T H dx = 2 * ((f(x + dx) - f(x)) - J dx)
    {
      auto f0 = eigen::as_const_view(out);

      auto _f1 = decltype(out.eval())::Zero(out.rows()).eval();
      auto _df = decltype(out_x.col(0).eval())::Zero(out_x.rows()).eval();

      auto _x1 = x.eval();
      auto _u1 = u.eval();

      auto f1 = eigen::as_mut_view(_f1);
      auto df = eigen::as_mut_view(_df);
      auto x1 = eigen::as_mut_view(_x1);
      auto u1 = eigen::as_mut_view(_u1);

      auto dx = dx_const<dims>::Zero(nx).eval();
      auto du = du_const<dims>::Zero(nu).eval();

      using std::sqrt;
      scalar_t eps = sqrt(sqrt(std::numeric_limits<scalar_t>::epsilon()));
      scalar_t eps2 = eps * eps;

      // compute diagonal of hessian
      for (index_t i = 0; i < nx + nu; ++i) {
        bool at_x = (i < nx);
        index_t idx = at_x ? i : (i - nx);

        scalar_t& in_var = at_x ? dx[idx] : du[idx];
        auto f_col = at_x ? eigen::as_const_view(out_x.col(idx)) : eigen::as_const_view(out_u.col(idx));
        auto tensor = at_x ? out_xx.as_dynamic() : out_uu.as_dynamic();

        in_var = eps;

        fn.integrate_x(x1, x, eigen::as_const_view(dx));
        fn.integrate_u(u1, t, u, eigen::as_const_view(du));

        k = fn.eval_to(f1, t, eigen::as_const_view(x1), eigen::as_const_view(u1), DDP_MOVE(k));

        // f(x + dx) - f(x)
        fn.difference_out(df, f0, eigen::as_const_view(f1));

        // (f(x + dx) - f(x)) - J dx
        // dx = eps * e_i => J dx = eps * J.col(i)
        df -= eps * f_col;

        // 2 * ((f(x + dx) - f(x)) - J dx)
        df *= 2;

        for (index_t out_i = 0; out_i < no; ++out_i) {
          tensor(out_i, idx, idx) = df[out_i] / eps2;
        }

        in_var = 0;
      }

      // compute non diagonal part
      // ei H ej = ((ei + ej) H (ei + ej) - ei H ei - ej H ej) / 2
      for (index_t i = 0; i < nx + nu; ++i) {
        bool at_x_1 = (i < nx);
        index_t idx_1 = at_x_1 ? i : (i - nx);

        scalar_t& in_var_1 = at_x_1 ? dx[idx_1] : du[idx_1];
        auto f_col_1 = at_x_1 //
                           ? eigen::as_const_view(out_x.col(idx_1))
                           : eigen::as_const_view(out_u.col(idx_1));

        auto tensor_1 = at_x_1 //
                            ? out_xx.as_dynamic()
                            : out_uu.as_dynamic();

        in_var_1 = eps;

        for (index_t j = i + 1; j < nx + nu; ++j) {
          bool at_x_2 = (j < nx);
          index_t idx_2 = at_x_2 ? j : (j - nx);

          scalar_t& in_var_2 = at_x_2 ? dx[idx_2] : du[idx_2];
          auto f_col_2 = at_x_2 //
                             ? eigen::as_const_view(out_x.col(idx_2))
                             : eigen::as_const_view(out_u.col(idx_2));

          auto tensor_2 = at_x_2 //
                              ? out_xx.as_dynamic()
                              : out_uu.as_dynamic();

          auto tensor = at_x_1 ? (at_x_2 //
                                      ? out_xx.as_dynamic()
                                      : out_ux.as_dynamic())
                               : (at_x_2 //
                                      ? (DDP_ASSERT(false), out_uu.as_dynamic())
                                      : out_uu.as_dynamic());

          in_var_2 = eps;

          fn.integrate_x(x1, x, eigen::as_const_view(dx));
          fn.integrate_u(u1, t, u, eigen::as_const_view(du));

          k = fn.eval_to(f1, t, eigen::as_const_view(x1), eigen::as_const_view(u1), DDP_MOVE(k));

          // f(x + dx) - f(x)
          fn.difference_out(df, f0, eigen::as_const_view(f1));

          // (f(x + dx) - f(x)) - J dx
          // dx = eps * e_i => J dx = eps * J.col(i)
          df -= eps * f_col_1;
          df -= eps * f_col_2;

          // 2 * ((f(x + dx) - f(x)) - J dx)
          df *= 2;

          // [v1, f1] == f(x + eps * ei + eps * ej) - f(x)

          for (index_t out_i = 0; out_i < no; ++out_i) {
            tensor(out_i, idx_2, idx_1) =              //
                0.5 * (df[out_i] / eps2                //
                       - tensor_1(out_i, idx_1, idx_1) //
                       - tensor_2(out_i, idx_2, idx_2));

            if (at_x_1 == at_x_2) {
              tensor(out_i, idx_1, idx_2) = tensor(out_i, idx_2, idx_1);
            }
          }

          in_var_2 = 0;
        }

        in_var_1 = 0;
      }
    }
    return k;
  }

  auto second_order_deriv(
      out_xx_mut<dims> out_xx,
      out_ux_mut<dims> out_ux,
      out_uu_mut<dims> out_uu,
      out_x_mut<dims> out_x,
      out_u_mut<dims> out_u,
      out_mut<dims> out,
      index_t t,
      x_const<dims> x,
      u_const<dims> u,
      key k) const -> key {
    auto odim = out_xx.outdim().value();
    auto xdim = out_xx.indiml().value();
    auto udim = out_ux.indiml().value();

    DDP_ASSERT(out_xx.outdim().value() == odim);
    DDP_ASSERT(out_xx.indiml().value() == xdim);
    DDP_ASSERT(out_xx.indimr().value() == xdim);

    DDP_ASSERT(out_ux.outdim().value() == odim);
    DDP_ASSERT(out_ux.indiml().value() == udim);
    DDP_ASSERT(out_ux.indimr().value() == xdim);

    DDP_ASSERT(out_uu.outdim().value() == odim);
    DDP_ASSERT(out_uu.indiml().value() == udim);
    DDP_ASSERT(out_uu.indimr().value() == udim);

    DDP_ASSERT(out_x.rows() == odim);
    DDP_ASSERT(out_x.cols() == xdim);

    DDP_ASSERT(out_u.rows() == odim);
    DDP_ASSERT(out_u.cols() == udim);

    if (second_order_finite_diff) {
      return second_order_deriv_2(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
    } else {
      return second_order_deriv_1(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
    }
  }

  Fn const& fn;
  bool second_order_finite_diff = true;
};

} // namespace ddp

#endif /* end of include guard SECOND_ORDER_FINITE_DIFF_HPP_5JNUIJG6 */
