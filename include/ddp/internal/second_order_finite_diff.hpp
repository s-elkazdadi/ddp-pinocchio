#ifndef DDP_PINOCCHIO_SECOND_ORDER_FINITE_DIFF_HPP_LNCEGZXSS
#define DDP_PINOCCHIO_SECOND_ORDER_FINITE_DIFF_HPP_LNCEGZXSS

#include "ddp/internal/tensor.hpp"

namespace ddp {

template <typename Fn, typename Key, typename T = typename Fn::scalar>
auto second_order_deriv_1(
    Fn const& fn,
    tensor_view<T> out_xx,
    tensor_view<T> out_ux,
    tensor_view<T> out_uu,
    view<T, colmat> out_x,
    view<T, colmat> out_u,
    view<T, colvec> out,
    i64 t,
    view<T const, colvec> x,
    view<T const, colvec> u,
    Key k) -> Key {

  auto odim = out_xx.outdim();
  auto xdim = out_xx.indiml();
  auto udim = out_ux.indiml();

  VEG_ASSERT(out_xx.outdim() == odim);
  VEG_ASSERT(out_xx.indiml() == xdim);
  VEG_ASSERT(out_xx.indimr() == xdim);

  VEG_ASSERT(out_ux.outdim() == odim);
  VEG_ASSERT(out_ux.indiml() == udim);
  VEG_ASSERT(out_ux.indimr() == xdim);

  VEG_ASSERT(out_uu.outdim() == odim);
  VEG_ASSERT(out_uu.indiml() == udim);
  VEG_ASSERT(out_uu.indimr() == udim);

  VEG_ASSERT(out_x.rows() == odim);
  VEG_ASSERT(out_x.cols() == xdim);

  VEG_ASSERT(out_u.rows() == odim);
  VEG_ASSERT(out_u.cols() == udim);
  VEG_ASSERT_ALL_OF_ELSE(
      ("non commutative groups are not supported", out_x.rows() == out.rows()),
      ("non commutative groups are not supported", out_x.cols() == x.rows()),
      ("non commutative groups are not supported", out_u.cols() == u.rows()));

  auto no = out_x.rows();
  auto nx = out_x.cols();
  auto nu = out_u.cols();

  k = fn.d_eval_to(out_x, out_u, out, t, x, u, VEG_FWD(k));
  // compute second derivatives
  {
    auto fx_ = out_x.eval();
    auto fu_ = out_u.eval();

    auto _out_ = out.eval();
    auto _x_ = x.eval();
    auto _u_ = u.eval();

    auto _dx = view<T, colvec>::Zero(nx).eval();
    auto _du = view<T, colvec>::Zero(nu).eval();

    auto out_ = eigen::dyn_cast<view, colvec>(_out_);
    auto x_ = eigen::dyn_cast<view, colvec>(_x_);
    auto u_ = eigen::dyn_cast<view, colvec>(_u_);
    auto dx = eigen::dyn_cast<view, colvec>(_dx);
    auto du = eigen::dyn_cast<view, colvec>(_du);

    using std::sqrt;
    T eps = sqrt(std::numeric_limits<T>::epsilon());

    for (i64 i = 0; i < nx + nu; ++i) {
      bool at_x = (i < nx);
      i64 idx = at_x ? i : (i - nx);

      T& in_var = at_x ? dx[idx] : du[idx];

      in_var = eps;

      fn.state_space().integrate(x_, t, x, eigen::as_const(dx));
      fn.control_space().integrate(u_, t, u, eigen::as_const(du));

      k = fn.d_eval_to(
          eigen::dyn_cast<view, colmat>(fx_),
          eigen::dyn_cast<view, colmat>(fu_),
          out_,
          t,
          eigen::as_const(x_),
          eigen::as_const(u_),
          VEG_FWD(k));

      if (at_x) {

        for (i64 out_i = 0; out_i < no; ++out_i) {
          for (i64 j = 0; j < nx; ++j) {
            out_xx(out_i, j, idx) = (fx_(out_i, j) - out_x(out_i, j)) / eps;
          }
          for (i64 j = 0; j < nu; ++j) {
            out_ux(out_i, j, idx) = (fu_(out_i, j) - out_u(out_i, j)) / eps;
          }
        }

      } else {
        for (i64 out_i = 0; out_i < no; ++out_i) {
          for (i64 j = 0; j < nu; ++j) {
            out_uu(out_i, j, idx) = (fu_(out_i, j) - out_u(out_i, j)) / eps;
          }
        }
      }

      in_var = 0;
    }
  }
  return k;
}

template <typename Fn>
auto second_order_deriv_2_req(Fn const& fn) -> mem_req {
  mem_req init = fn.d_eval_to_req();

  mem_req max_ = mem_req::sum_of(

      {mem_req::max_of({
           fn.state_space().integrate_req(),
           fn.control_space().integrate_req(),
           fn.output_space().difference_req(),
           fn.eval_to_req(),
       }),

       {veg::tag<typename Fn::scalar>,
        (fn.output_space().max_dim()     // f1
         + fn.output_space().max_ddim()  // df
         + fn.state_space().max_dim()    // x1
         + fn.control_space().max_dim()  // u1
         + fn.state_space().max_ddim()   // dx
         + fn.control_space().max_ddim() // du
         )}});

  return mem_req::max_of({init, max_});
}

template <typename Fn, typename T = typename Fn::scalar>
auto second_order_deriv_2(
    Fn const& fn,
    tensor_view<T> out_xx,
    tensor_view<T> out_ux,
    tensor_view<T> out_uu,
    view<T, colmat> out_x,
    view<T, colmat> out_u,
    view<T, colvec> out,
    i64 t,
    view<T const, colvec> x,
    view<T const, colvec> u,
    typename Fn::key k,
    veg::dynamic_stack_view stack) -> typename Fn::key {

  auto no = out_x.rows();
  auto nx = out_x.cols();
  auto nu = out_u.cols();

  k = fn.d_eval_to(out_x, out_u, out, t, x, u, VEG_FWD(k), stack);
  // compute second derivatives
  // dx.T H dx = 2 * ((f(x + dx) - f(x)) - J dx)
  {
    auto f0 = eigen::as_const(out);

    auto _f1 = stack.make_new(veg::tag<T>, out.rows()).unwrap();
    auto _df = stack.make_new(veg::tag<T>, out_x.rows()).unwrap();

    auto _x1 = stack.make_new_for_overwrite(veg::tag<T>, x.rows()).unwrap();
    auto _u1 = stack.make_new_for_overwrite(veg::tag<T>, u.rows()).unwrap();

    auto _dx = stack.make_new(veg::tag<T>, nx).unwrap();
    auto _du = stack.make_new(veg::tag<T>, nu).unwrap();

    auto f1 = eigen::slice_to_vec(_f1);
    auto df = eigen::slice_to_vec(_df);
    auto x1 = eigen::slice_to_vec(_x1);
    auto u1 = eigen::slice_to_vec(_u1);
    x1 = x;
    u1 = u;

    auto dx = eigen::slice_to_vec(_dx);
    auto du = eigen::slice_to_vec(_du);

    using std::sqrt;
    T eps = sqrt(sqrt(std::numeric_limits<T>::epsilon()));
    T eps2 = eps * eps;

    // compute diagonal of hessian
    for (i64 i = 0; i < nx + nu; ++i) {
      bool at_x = (i < nx);
      i64 idx = at_x ? i : (i - nx);

      T& in_var = at_x ? dx[idx] : du[idx];
      auto f_col = at_x ? eigen::as_const(out_x.col(idx))
                        : eigen::as_const(out_u.col(idx));
      auto tensor = at_x ? out_xx : out_uu;

      in_var = eps;

      fn.state_space().integrate(x1, t, x, eigen::as_const(dx), stack);
      fn.control_space().integrate(u1, t, u, eigen::as_const(du), stack);

      k = fn.eval_to(
          f1, t, eigen::as_const(x1), eigen::as_const(u1), VEG_FWD(k), stack);

      // f(x + dx) - f(x)
      fn.output_space().difference(df, t, f0, eigen::as_const(f1), stack);

      // (f(x + dx) - f(x)) - J dx
      // dx = eps * e_i => J dx = eps * J.col(i)
      df -= eps * f_col;

      // 2 * ((f(x + dx) - f(x)) - J dx)
      df *= 2;

      for (i64 out_i = 0; out_i < no; ++out_i) {
        tensor(out_i, idx, idx) = df[out_i] / eps2;
      }

      in_var = 0;
    }

    // compute non diagonal part
    // ei H ej = ((ei + ej) H (ei + ej) - ei H ei - ej H ej) / 2
    for (i64 i = 0; i < nx + nu; ++i) {
      bool at_x_1 = (i < nx);
      i64 idx_1 = at_x_1 ? i : (i - nx);

      T& in_var_1 = at_x_1 ? dx[idx_1] : du[idx_1];
      auto f_col_1 = at_x_1 //
                         ? eigen::as_const(out_x.col(idx_1))
                         : eigen::as_const(out_u.col(idx_1));

      auto tensor_1 = at_x_1 //
                          ? out_xx
                          : out_uu;

      in_var_1 = eps;

      for (i64 j = i + 1; j < nx + nu; ++j) {
        bool at_x_2 = (j < nx);
        i64 idx_2 = at_x_2 ? j : (j - nx);

        T& in_var_2 = at_x_2 ? dx[idx_2] : du[idx_2];
        auto f_col_2 = at_x_2 ? eigen::as_const(out_x.col(idx_2))
                              : eigen::as_const(out_u.col(idx_2));

        auto tensor_2 = at_x_2 ? out_xx : out_uu;

        auto tensor = at_x_1 ? (at_x_2 ? out_xx : out_ux)
                             : (at_x_2 ? (VEG_ASSERT(false), out_uu) : out_uu);

        in_var_2 = eps;

        fn.state_space().integrate(x1, t, x, eigen::as_const(dx), stack);
        fn.control_space().integrate(u1, t, u, eigen::as_const(du), stack);

        k = fn.eval_to(
            f1, t, eigen::as_const(x1), eigen::as_const(u1), VEG_FWD(k), stack);

        // f(x + dx) - f(x)
        fn.output_space().difference(df, t, f0, eigen::as_const(f1), stack);

        // (f(x + dx) - f(x)) - J dx
        // dx = eps * e_i => J dx = eps * J.col(i)
        df -= eps * f_col_1;
        df -= eps * f_col_2;

        // 2 * ((f(x + dx) - f(x)) - J dx)
        df *= 2;

        // [v1, f1] == f(x + eps * ei + eps * ej) - f(x)

        for (i64 out_i = 0; out_i < no; ++out_i) {
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
} // namespace ddp

#endif /* end of include guard                                                 \
          DDP_PINOCCHIO_SECOND_ORDER_FINITE_DIFF_HPP_LNCEGZXSS */
