#ifndef PROBLEM_HPP_SQUCCNMX
#define PROBLEM_HPP_SQUCCNMX

#include "ddp/detail/utils.hpp"
#include "ddp/detail/tensor.hpp"
#include "ddp/trajectory.hpp"
#include "ddp/ddp.hpp"
#include <array>

namespace ddp {

template <typename Out, typename XX, typename UX, typename UU, typename DX, typename DU>
void add_second_order_term(Out&& out, XX const& xx, UX const& ux, UU const& uu, DX const& dx, DU const& du) {
  DDP_ASSERT_MSG_ALL_OF(
      ("", out.cols() == 1),
      ("", out.rows() == xx.outdim().value()),
      ("", out.rows() == ux.outdim().value()),
      ("", out.rows() == uu.outdim().value()),
      ("", dx.rows() == xx.indiml().value()),
      ("", dx.rows() == xx.indimr().value()),
      ("", dx.rows() == ux.indimr().value()),
      ("", du.rows() == uu.indiml().value()),
      ("", du.rows() == uu.indimr().value()),
      ("", du.rows() == ux.indiml().value()));

  for (index_t j = 0; j < dx.rows(); ++j) {
    for (index_t i = 0; i < dx.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * dx(i) * xx(k, i, j) * dx(j);
      }
    }
  }

  for (index_t j = 0; j < du.rows(); ++j) {
    for (index_t i = 0; i < du.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * du(i) * uu(k, i, j) * du(j);
      }
    }
  }

  for (index_t j = 0; j < dx.rows(); ++j) {
    for (index_t i = 0; i < du.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += du(i) * ux(k, i, j) * dx(j);
      }
    }
  }
}

template <typename Fn>
struct finite_diff_hessian_compute {
  using scalar_t = typename Fn::scalar_t;
  using x_const = typename Fn::x_const;
  using dx_const = typename Fn::dx_const;
  using u_const = typename Fn::u_const;
  using du_const = typename Fn::du_const;

  using out_mut = typename Fn::out_mut;

  using out_x_mut = typename Fn::out_x_mut;
  using out_u_mut = typename Fn::out_u_mut;

  using out_xx_mut = typename Fn::out_xx_mut;
  using out_ux_mut = typename Fn::out_ux_mut;
  using out_uu_mut = typename Fn::out_uu_mut;

  using key = typename Fn::key;

  auto second_order_deriv_1(
      out_xx_mut out_xx,
      out_ux_mut out_ux,
      out_uu_mut out_uu,
      out_x_mut out_x,
      out_u_mut out_u,
      out_mut out,
      index_t t,
      x_const x,
      u_const u,
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

      auto _dx = dx_const::Zero(nx).eval();
      auto _du = du_const::Zero(nu).eval();

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
      out_xx_mut out_xx,
      out_ux_mut out_ux,
      out_uu_mut out_uu,
      out_x_mut out_x,
      out_u_mut out_u,
      out_mut out,
      index_t t,
      x_const x,
      u_const u,
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

      auto dx = dx_const::Zero(nx).eval();
      auto du = du_const::Zero(nu).eval();

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
      out_xx_mut out_xx,
      out_ux_mut out_ux,
      out_uu_mut out_uu,
      out_x_mut out_x,
      out_u_mut out_u,
      out_mut out,
      index_t t,
      x_const x,
      u_const u,
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

template <typename Model>
struct dynamics_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;

  using state_kind = decltype(
      static_cast<model_t const*>(nullptr)->configuration_dim_c() +
      static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using dstate_kind = decltype(
      static_cast<model_t const*>(nullptr)->tangent_dim_c() + static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using control_kind = decltype(static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using dcontrol_kind = decltype(static_cast<model_t const*>(nullptr)->tangent_dim_c());

  using state_indexer_t = indexing::regular_indexer_t<state_kind>;
  using dstate_indexer_t = indexing::regular_indexer_t<dstate_kind>;

  using _1 = fix_index<1>;

  using x_mut = eigen::matrix_view_t<scalar_t, state_kind, _1>;
  using x_const = eigen::matrix_view_t<scalar_t const, state_kind, _1>;
  using dx_const = eigen::matrix_view_t<scalar_t const, dstate_kind, _1>;
  using u_mut = eigen::matrix_view_t<scalar_t, control_kind, _1>;
  using u_const = eigen::matrix_view_t<scalar_t const, control_kind, _1>;
  using du_const = eigen::matrix_view_t<scalar_t const, dcontrol_kind, _1>;

  using out_const = eigen::matrix_view_t<scalar_t const, state_kind, _1>;
  using out_mut = eigen::matrix_view_t<scalar_t, state_kind, _1>;
  using dout_mut = eigen::matrix_view_t<scalar_t, dstate_kind, _1>;

  using out_x_mut = eigen::matrix_view_t<scalar_t, dstate_kind, dstate_kind>;
  using out_u_mut = eigen::matrix_view_t<scalar_t, dstate_kind, dcontrol_kind>;

  using out_xx_mut = tensor::tensor_view_t<scalar_t, dstate_kind, dstate_kind, dstate_kind>;
  using out_ux_mut = tensor::tensor_view_t<scalar_t, dstate_kind, dcontrol_kind, dstate_kind>;
  using out_uu_mut = tensor::tensor_view_t<scalar_t, dstate_kind, dcontrol_kind, dcontrol_kind>;

  using key = typename model_t::key;

  auto state_dim() const -> state_kind { return m_model.configuration_dim_c() + m_model.tangent_dim_c(); }
  auto dstate_dim() const noexcept -> dstate_kind { return m_model.tangent_dim_c() + m_model.tangent_dim_c(); }
  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.configuration_dim_c() + m_model.tangent_dim_c());
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.tangent_dim_c() + m_model.tangent_dim_c());
  }
  auto control_dim(index_t /*unused*/) const noexcept -> control_kind { return m_model.tangent_dim_c(); }
  auto dcontrol_dim(index_t /*unused*/) const noexcept -> dcontrol_kind { return m_model.tangent_dim_c(); }

  auto acquire_workspace() const noexcept -> key { return m_model.acquire_workspace(); }

  void neutral_configuration(x_mut out) const {
    DDP_ASSERT_MSG("out vector does not have the correct size", out.size() == state_dim().value());
    auto nq = m_model.configuration_dim_c();
    DDP_BIND(auto, (out_q, out_v), eigen::split_at_row_mut(out, nq));
    m_model.neutral_configuration(out_q);
    out_v.setZero();
  }

  void integrate_x(x_mut out, x_const x, dx_const dx) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (q_out, v_out), eigen::split_at_row_mut(out, nq));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (dq, dv), eigen::split_at_row(dx, nv));

    m_model.integrate(q_out, q, dq);
    v_out = v + dv;
  }

  void d_integrate_x(out_x_mut out, x_const x, dx_const dx) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (out_qq, out_qv, out_vq, out_vv), eigen::split_at_mut(out, nv, nv));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (dq, dv), eigen::split_at_row(dx, nv));

    m_model.d_integrate_dq(out_qq, q, dq);
    out_qv.setZero();
    out_vq.setZero();
    out_vv.setIdentity();
  }
  void d_integrate_x_dx(out_x_mut out, x_const x, dx_const dx) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (out_qq, out_qv, out_vq, out_vv), eigen::split_at_mut(out, nv, nv));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (dq, dv), eigen::split_at_row(dx, nv));

    m_model.d_integrate_dv(out_qq, q, dq);
    out_qv.setZero();
    out_vq.setZero();
    out_vv.setIdentity();
  }

  void integrate_u(u_mut out, index_t /*unused*/, u_const u, u_const du) const { out = u + du; }
  void difference_out(dout_mut out, out_const start, out_const finish) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (out_q, out_v), eigen::split_at_row_mut(out, nv));
    DDP_BIND(auto, (start_q, start_v), eigen::split_at_row(start, nq));
    DDP_BIND(auto, (finish_q, finish_v), eigen::split_at_row(finish, nq));

    m_model.difference(out_q, start_q, finish_q);
    out_v = finish_v - start_v;
  }
  void d_difference_out_dfinish(out_x_mut out, x_const start, x_const finish) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();

    DDP_ASSERT(out.rows() == (nv + nv).value());
    DDP_ASSERT(out.cols() == (nv + nv).value());
    DDP_ASSERT(start.size() == (nq + nv).value());
    DDP_ASSERT(finish.size() == (nq + nv).value());

    DDP_BIND(auto, (out_top_left, out_top_right, out_bot_left, out_bot_right), eigen::split_at_mut(out, nv, nv));
    DDP_BIND(auto, (start_q, start_v), eigen::split_at_row(start, nq));
    DDP_BIND(auto, (finish_q, finish_v), eigen::split_at_row(finish, nq));
    (void)start_v;
    (void)finish_v;

    m_model.d_difference_dq_finish(out_top_left, start_q, finish_q);

    out_bot_right.setIdentity();
    out_top_right.setZero();
    out_bot_left.setZero();
  }

  auto eval_to(out_mut x_out, index_t t, x_const x, u_const u, key k) const -> key {
    (void)t;
    auto nq = m_model.configuration_dim_c();

    DDP_BIND(auto, (q_out, v_out), eigen::split_at_row_mut(x_out, nq));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));

    // v_out = dt * v_in
    v_out = dt * v;

    // q_out = q_in + v_out
    //       = q_in + dt * v_in
    m_model.integrate(q_out, q, eigen::as_const_view(v_out));

    // v_out = acc
    k = m_model.dynamics_aba(v_out, q, v, u, DDP_MOVE(k));

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    v_out = v + v_out * dt;
    return k;
  }

  auto first_order_deriv( //
      out_x_mut fx,       //
      out_u_mut fu,       //
      out_mut f,          //
      index_t t,          //
      x_const x,          //
      u_const u,          //
      key k               //
  ) const -> key {
    k = eval_to(f, t, x, u, DDP_MOVE(k));

    (void)t;
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();

    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));

    auto tmp = detail::get<1>(eigen::split_at_row_mut(fx.col(0), nv));
    // v_out = dt * v_in;
    tmp = dt * v;
    auto dt_v = eigen::as_const_view(tmp);

    DDP_BIND(auto, (fx_top_left, fx_top_right, fx_bot_left, fx_bot_right), eigen::split_at_mut(fx, nv, nv));

    // q_out = q_in + dt * v_in
    m_model.d_integrate_dq(fx_top_left, q, dt_v);
    m_model.d_integrate_dv(fx_top_right, q, dt_v);
    fx_top_right *= dt;

    // v_out = acc
    DDP_BIND(auto, (fu_top, fu_bot), eigen::split_at_row_mut(fu, nv));

    fu_top.setZero();
    k = m_model.d_dynamics_aba(fx_bot_left, fx_bot_right, fu_bot, q, v, u, DDP_MOVE(k));

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    fx_bot_left *= dt;
    fx_bot_right *= dt;
    fx_bot_right += decltype(fx_bot_right.eval())::Identity(nv.value(), nv.value());
    fu_bot *= dt;
    return k;
  }

  auto second_order_deriv( //
      out_xx_mut fxx,      //
      out_ux_mut fux,      //
      out_uu_mut fuu,      //
      out_x_mut fx,        //
      out_u_mut fu,        //
      out_mut f,           //
      index_t t,           //
      x_const x,           //
      u_const u,           //
      key k                //
  ) const -> key {
    return finite_diff_hessian_compute<dynamics_t>{*this, m_second_order_finite_diff}
        .second_order_deriv(fxx, fux, fuu, fx, fu, f, t, x, u, DDP_MOVE(k));
  }

  auto second_order_finite_diff() const noexcept { return m_second_order_finite_diff; }

  auto name() const noexcept -> fmt::string_view { return m_model.model_name(); }

  model_t const& m_model;
  scalar_t dt;
  bool m_second_order_finite_diff = true;
};

template <typename Constraint>
struct constraint_advance_time_t {
  using scalar_t = typename Constraint::scalar_t;
  using dynamics_t = typename Constraint::dynamics_t;
  using eq_indexer_t = indexing::shift_time_idx_t<typename Constraint::eq_indexer_t>;

  using x_mut = typename Constraint::x_mut;
  using u_mut = typename Constraint::u_mut;
  using x_const = typename Constraint::x_const;
  using dx_const = typename Constraint::dx_const;
  using u_const = typename Constraint::u_const;
  using du_const = typename Constraint::du_const;

  using out_const = typename Constraint::out_const;
  using out_mut = typename Constraint::out_mut;
  using dout_mut = typename Constraint::dout_mut;

  using out_x_mut = typename Constraint::out_x_mut;
  using out_u_mut = typename Constraint::out_u_mut;

  using out_xx_mut = typename Constraint::out_xx_mut;
  using out_ux_mut = typename Constraint::out_ux_mut;
  using out_uu_mut = typename Constraint::out_uu_mut;

  using key = typename Constraint::key;

  using _1 = fix_index<1>;

  auto eq_idx() const -> eq_indexer_t { return {m_constraint.eq_idx(), 1}; }
  auto eq_dim(index_t t) const -> typename eq_indexer_t::row_kind { return m_constraint.eq_dim(t + 1); }

  void integrate_x(x_mut out, x_const x, dx_const dx) const { m_constraint.integrate_x(out, x, dx); }
  void integrate_u(u_mut out, index_t t, u_const u, u_const du) const { m_constraint.integrate_u(out, t, u, du); }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    m_constraint.difference_out(out, start, finish);
  }

  auto eval_to(out_mut out, index_t t, x_const x, u_const u, key k) const -> key {
    auto x_n = eigen::make_matrix<scalar_t>(m_dynamics.state_dim(), _1{}); // x_{t+1}
    k = m_dynamics.eval_to(eigen::as_mut_view(x_n), t, x, u, DDP_MOVE(k));
    k = m_constraint.eval_to(out, t + 1, eigen::as_const_view(x_n), u, DDP_MOVE(k));
    return k;
  }

  auto first_order_deriv( //
      out_x_mut out_x,    //
      out_u_mut out_u,    //
      out_mut out,        //
      index_t t,          //
      x_const x,          //
      u_const u,          //
      key k) const -> key {

    if (out.rows() == 0) {
      return k;
    }

    auto nx = m_dynamics.state_dim();
    auto nx_ = m_dynamics.dstate_dim();

    auto _x_n = eigen::make_matrix<scalar_t>(nx, _1{}); // x_{t+1}
    auto x_n = eigen::as_mut_view(_x_n);

    auto _fx_n = eigen::make_matrix<scalar_t>(nx_, nx_);
    auto _fu_n = eigen::make_matrix<scalar_t>(nx_, m_dynamics.dcontrol_dim(t));
    auto fx_n = eigen::as_mut_view(_fx_n);
    auto fu_n = eigen::as_mut_view(_fu_n);

    k = m_dynamics.first_order_deriv(fx_n, fu_n, x_n, t, x, u, DDP_MOVE(k));

    auto _eq_n_x = eigen::make_matrix<scalar_t>(eigen::rows_c(out), nx_);
    auto _eq_n_u = eigen::make_matrix<scalar_t>(eigen::rows_c(out), m_dynamics.dcontrol_dim(t + 1));
    auto eq_n_x = eigen::as_mut_view(_eq_n_x);
    auto eq_n_u = eigen::as_mut_view(_eq_n_u);

    k = m_constraint.first_order_deriv(eq_n_x, eq_n_u, out, t + 1, eigen::as_const_view(x_n), u, DDP_MOVE(k));
    DDP_ASSERT_MSG("constraint depends on control", eq_n_u.isConstant(0));

    out_x.noalias() = eq_n_x * fx_n;
    out_u.noalias() = eq_n_x * fu_n;

    return k;
  }

  auto second_order_deriv( //
      out_xx_mut out_xx,   //
      out_ux_mut out_ux,   //
      out_uu_mut out_uu,   //
      out_x_mut out_x,     //
      out_u_mut out_u,     //
      out_mut out,
      index_t t, //
      x_const x, //
      u_const u, //
      key k      //
  ) const -> key {
    return finite_diff_hessian_compute<constraint_advance_time_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  Constraint m_constraint;
  dynamics_t m_dynamics;
};

template <typename Constraint>
auto constraint_advance_time(Constraint c) -> constraint_advance_time_t<Constraint> {
  return {c, c.m_dynamics};
}

template <typename Model, typename Constraint_Target_View>
struct spatial_constraint_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using state_kind = typename dynamics_t::state_kind;
  using dstate_kind = typename dynamics_t::dstate_kind;
  using control_kind = decltype(static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using eq_indexer_t = decltype(std::declval<Constraint_Target_View const&>().eq_idx());

  static constexpr index_t out_eigen_c =
      std::remove_reference<decltype(std::declval<Constraint_Target_View const&>()[0])>::type::RowsAtCompileTime;
  using out_kind = DDP_CONDITIONAL(out_eigen_c == -1, dyn_index, fix_index<out_eigen_c>);

  using _1 = fix_index<1>;

  using x_mut = typename dynamics_t::x_mut;
  using u_mut = typename dynamics_t::u_mut;
  using x_const = typename dynamics_t::x_const;
  using dx_const = typename dynamics_t::dx_const;
  using u_const = typename dynamics_t::u_const;
  using du_const = typename dynamics_t::du_const;

  using out_const = eigen::matrix_view_t<scalar_t const, out_kind, _1>;
  using out_mut = eigen::matrix_view_t<scalar_t, out_kind, _1>;
  using dout_mut = eigen::matrix_view_t<scalar_t, out_kind, _1>;

  using out_x_mut = eigen::matrix_view_t<scalar_t, out_kind, dstate_kind>;
  using out_u_mut = eigen::matrix_view_t<scalar_t, out_kind, control_kind>;

  using out_xx_mut = tensor::tensor_view_t<scalar_t, out_kind, dstate_kind, dstate_kind>;
  using out_ux_mut = tensor::tensor_view_t<scalar_t, out_kind, control_kind, dstate_kind>;
  using out_uu_mut = tensor::tensor_view_t<scalar_t, out_kind, control_kind, control_kind>;

  using key = typename dynamics_t::key;

  auto eq_idx() const -> eq_indexer_t { return m_constraint_target_view.eq_idx(); }
  auto eq_dim(index_t t) const -> typename eq_indexer_t::row_kind { return eq_idx().rows(t); }

  void integrate_x(x_mut out, x_const x, dx_const dx) const { m_dynamics.integrate_x(out, x, dx); }
  void integrate_u(u_mut out, index_t t, u_const u, u_const du) const { m_dynamics.integrate_u(out, t, u, du); }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
        ("", out.rows() == start.rows()), //
        ("", out.rows() == finish.rows()));
    out = finish - start;
  }

  auto eval_to(out_mut out, index_t t, x_const x, u_const u, key k) const -> key {
    auto target = eigen::as_const_view(m_constraint_target_view[t]);
    if (target.rows() == 0) {
      return;
    }

    (void)u;
    auto nq = m_dynamics.m_model.configuration_dim_c();

    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (out_3, out_0), eigen::split_at_row_mut(out, fix_index<3>{}));
    (void)v;

    k = m_dynamics.m_model.frame_coordinates(out_3, m_frame_id, q, DDP_MOVE(k));
    out -= target;
    return k;
  }

  auto first_order_deriv( //
      out_x_mut out_x,    //
      out_u_mut out_u,    //
      out_mut out,        //
      index_t t,          //
      x_const x,          //
      u_const u,          //
      key k               //
  ) const -> key {
    (void)u;
    auto target = eigen::as_const_view(m_constraint_target_view[t]);
    DDP_ASSERT_MSG(fmt::format("at t = {}", t), target.rows() == out.rows());
    if (target.rows() == 0) {
      return;
    }

    auto const& m_model = m_dynamics.m_model;

    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();

    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (out_3, out_0), eigen::split_at_row_mut(out_x, fix_index<3>{}));
    DDP_ASSERT(out_0.rows() == 0);
    DDP_BIND(auto, (out_3q, out_3v), eigen::split_at_col_mut(out_3, nv));

    (void)v;

    k = m_dynamics.m_model.frame_coordinates(out_3, m_frame_id, q, DDP_MOVE(k));
    out -= target;

    k = m_dynamics.m_model.d_frame_coordinates(out_3q, m_frame_id, q, DDP_MOVE(k));
    out_3v.setZero();
    out_u.setZero();
    return k;
  }

  auto second_order_deriv( //
      out_xx_mut out_xx,   //
      out_ux_mut out_ux,   //
      out_uu_mut out_uu,   //
      out_x_mut out_x,     //
      out_u_mut out_u,     //
      out_mut out,
      index_t t, //
      x_const x, //
      u_const u, //
      key k      //
  ) const -> key {
    return finite_diff_hessian_compute<spatial_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
  index_t m_frame_id;
};

template <typename Model, typename Constraint_Target_View>
struct config_constraint_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using state_kind = typename dynamics_t::state_kind;
  using dstate_kind = typename dynamics_t::dstate_kind;
  using control_kind = decltype(static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using eq_indexer_t = decltype(std::declval<Constraint_Target_View const&>().eq_idx());

  static constexpr index_t out_eigen_c =
      std::remove_reference<decltype(std::declval<Constraint_Target_View const&>()[0])>::type::RowsAtCompileTime;
  using out_kind = DDP_CONDITIONAL(out_eigen_c == -1, dyn_index, fix_index<out_eigen_c>);

  using _1 = fix_index<1>;

  using x_mut = typename dynamics_t::x_mut;
  using u_mut = typename dynamics_t::u_mut;
  using x_const = typename dynamics_t::x_const;
  using dx_const = typename dynamics_t::dx_const;
  using u_const = typename dynamics_t::u_const;
  using du_const = typename dynamics_t::du_const;

  using out_const = eigen::matrix_view_t<scalar_t const, out_kind, _1>;
  using out_mut = eigen::matrix_view_t<scalar_t, out_kind, _1>;
  using dout_mut = eigen::matrix_view_t<scalar_t, out_kind, _1>;

  using out_x_mut = eigen::matrix_view_t<scalar_t, out_kind, dstate_kind>;
  using out_u_mut = eigen::matrix_view_t<scalar_t, out_kind, control_kind>;

  using out_xx_mut = tensor::tensor_view_t<scalar_t, out_kind, dstate_kind, dstate_kind>;
  using out_ux_mut = tensor::tensor_view_t<scalar_t, out_kind, control_kind, dstate_kind>;
  using out_uu_mut = tensor::tensor_view_t<scalar_t, out_kind, control_kind, control_kind>;

  using key = typename dynamics_t::key;

  auto eq_idx() const -> eq_indexer_t { return m_constraint_target_view.eq_idx(); }
  auto eq_dim(index_t t) const -> typename eq_indexer_t::row_kind { return eq_idx().rows(t); }

  void integrate_x(x_mut out, x_const x, dx_const dx) const { m_dynamics.integrate_x(out, x, dx); }
  void integrate_u(u_mut out, index_t t, u_const u, u_const du) const { m_dynamics.integrate_u(out, t, u, du); }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
        ("", out.rows() == start.rows()), //
        ("", out.rows() == finish.rows()));
    out = finish - start;
  }

  auto eval_to(out_mut out, index_t t, x_const x, u_const u, key k) const -> key {
    auto target = eigen::as_const_view(m_constraint_target_view[t]);
    if (target.rows() == 0) {
      return k;
    }

    (void)u;
    auto nq = m_dynamics.m_model.configuration_dim_c();

    difference_out(
        out,
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x, nq)) // configuration part of x_nn
    );
    return k;
  }

  auto first_order_deriv( //
      out_x_mut out_x,    //
      out_u_mut out_u,    //
      out_mut out,        //
      index_t t,          //
      x_const x,          //
      u_const u,          //
      key k               //
  ) const -> key {
    (void)u;
    auto target = eigen::as_const_view(m_constraint_target_view[t]);
    DDP_ASSERT_MSG(fmt::format("at t = {}", t), target.rows() == out.rows());
    if (target.rows() == 0) {
      return k;
    }

    auto const& m_model = m_dynamics.m_model;

    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();

    difference_out(
        out,
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x, nq)) // configuration part of x_nn
    );

    auto d_diff = eigen::make_matrix<scalar_t>(nv, nv);
    m_dynamics.m_model.d_difference_dq_finish(
        eigen::as_mut_view(d_diff),
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x, nq)) // configuration part of x_nn
    );

    DDP_BIND(auto, (out_xq, out_xv), eigen::split_at_col_mut(out_x, nv));
    out_xq.noalias() = d_diff;
    out_xv.setZero();
    out_u.setZero();
    return k;
  }

  auto second_order_deriv( //
      out_xx_mut out_xx,   //
      out_ux_mut out_ux,   //
      out_uu_mut out_uu,   //
      out_x_mut out_x,     //
      out_u_mut out_u,     //
      out_mut out,
      index_t t, //
      x_const x, //
      u_const u, //
      key k      //
  ) const -> key {
    return finite_diff_hessian_compute<config_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
};

template <typename Dynamics, typename Constraint_Target_View>
auto config_constraint(Dynamics d, Constraint_Target_View v)
    -> config_constraint_t<typename Dynamics::model_t, Constraint_Target_View> {
  return {d, v};
}

template <typename Dynamics, typename Eq>
struct problem_t {
  using dynamics_t = Dynamics;
  using scalar_t = typename dynamics_t::scalar_t;

  using constraint_t = Eq;

  using state_kind = typename dynamics_t::state_kind;
  using dstate_kind = typename dynamics_t::dstate_kind;
  using control_kind = typename dynamics_t::control_kind;
  using dcontrol_kind = typename dynamics_t::dcontrol_kind;

  using _1 = fix_index<1>;

  using state_indexer_t = indexing::regular_indexer_t<state_kind>;
  using dstate_indexer_t = indexing::regular_indexer_t<dstate_kind>;
  using control_indexer_t = indexing::regular_indexer_t<control_kind>;
  using dcontrol_indexer_t = indexing::regular_indexer_t<dcontrol_kind>;
  using eq_indexer_t = typename constraint_t::eq_indexer_t;
  using eq_kind = typename eq_indexer_t::row_kind;

  using x_mut = typename dynamics_t::x_mut;
  using u_mut = typename dynamics_t::u_mut;
  using x_const = typename dynamics_t::x_const;
  using dx_const = typename dynamics_t::dx_const;
  using u_const = typename dynamics_t::u_const;

  using f_const = typename dynamics_t::out_const;
  using f_mut = typename dynamics_t::out_mut;
  using df_mut = typename dynamics_t::dout_mut;

  using fx_mut = typename dynamics_t::out_x_mut;
  using fu_mut = typename dynamics_t::out_u_mut;

  using fxx_mut = typename dynamics_t::out_xx_mut;
  using fux_mut = typename dynamics_t::out_ux_mut;
  using fuu_mut = typename dynamics_t::out_uu_mut;

  using eq_mut = typename constraint_t::out_mut;

  using eq_x_mut = typename constraint_t::out_x_mut;
  using eq_u_mut = typename constraint_t::out_u_mut;

  using eq_xx_mut = typename constraint_t::out_xx_mut;
  using eq_ux_mut = typename constraint_t::out_ux_mut;
  using eq_uu_mut = typename constraint_t::out_uu_mut;

  using key = typename dynamics_t::key;

  auto dynamics() const noexcept -> dynamics_t const& { return m_dynamics; }
  auto constraint() const noexcept -> constraint_t const& { return m_constraint; }

  auto state_dim() const -> state_kind { return m_dynamics.state_dim(); }
  auto dstate_dim() const noexcept -> dstate_kind { return m_dynamics.dstate_dim(); }

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return m_dynamics.state_indexer(begin, end);
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return m_dynamics.dstate_indexer(begin, end);
  }

  void neutral_configuration(x_mut out) const { m_dynamics.neutral_configuration(out); }

  auto lf(x_const x) const -> scalar_t {
    (void)x;
    (void)this;
    return 0;
  }
  auto l(index_t t, x_const x, u_const u) const -> scalar_t {
    (void)t;
    (void)x;
    (void)this;
    return 0.5 * c * u.squaredNorm();
  }

  void difference(f_mut out, f_const start, f_const finish) const { m_dynamics.difference_out(out, start, finish); }
  void d_difference_dfinish(fx_mut out, f_const start, f_const finish) const {
    m_dynamics.d_difference_out_dfinish(out, start, finish);
  }

  auto eval_f_to(x_mut x_out, index_t t, x_const x, u_const u, key k = {}) const -> key {
    if (!k) {
      k = m_dynamics.acquire_workspace();
    }
    return m_dynamics.eval_to(x_out, t, x, u, DDP_MOVE(k));
  }
  auto eval_eq_to(eq_mut eq_out, index_t t, x_const x, u_const u, key k = {}) const -> key {
    if (!k) {
      k = m_dynamics.acquire_workspace();
    }
    return m_constraint.eval_to(eq_out, t, x, u, DDP_MOVE(k));
  }

  using derivative_storage_t =
      ddp::derivative_storage_t<scalar_t, control_indexer_t, eq_indexer_t, state_indexer_t, dstate_indexer_t>;
  using trajectory_t = trajectory::trajectory_t<scalar_t, state_indexer_t, control_indexer_t>;

  auto compute_derivatives(derivative_storage_t& derivs, trajectory_t const& traj) const {

    derivs.lfx.setZero();
    derivs.lfxx.setZero();

    // clang-format off
    auto rng = ranges::zip(
          derivs.lx, derivs.lu, derivs.lxx, derivs.lux, derivs.luu,
          derivs.f_val, derivs.fx, derivs.fu, derivs.fxx, derivs.fux, derivs.fuu,
          derivs.eq_val, derivs.eq_x, derivs.eq_u, derivs.eq_xx, derivs.eq_ux, derivs.eq_uu,
          traj);
    // clang-format on

    using iter = typename decltype(rng)::iterator;

    constexpr index_t max_threads = 16;

    index_t n_threads = std::min(max_threads, index_t{omp_get_num_procs()});
    std::array<iter, max_threads> begin_arr;
    std::array<iter, max_threads> end_arr;

    {
      index_t horizon = 1 + (*(--end(traj))).current_index() - (*begin(traj)).current_index();
      index_t r = horizon % n_threads;

      iter it = begin(rng);
      for (index_t i = 0; i < n_threads; ++i) {
        index_t n_iterations = horizon / n_threads + ((i < r) ? 1 : 0);

        begin_arr[i] = it;
        for (index_t j = 0; j < n_iterations; ++j) {
          ++it;
        }
        end_arr[i] = it;
      }
      DDP_ASSERT((it == end(rng)));
    }

#pragma omp parallel num_threads(n_threads)
    {
      index_t thread_id = omp_get_thread_num();
      DDP_ASSERT(thread_id >= 0);
      DDP_ASSERT(thread_id < n_threads);

      auto k = dynamics().acquire_workspace();

      for (iter it = begin_arr[thread_id], it_end = end_arr[thread_id]; it != it_end; ++it) {
        auto zipped = *it;
        // clang-format off
      DDP_BIND(auto&&, (
          lx, lu, lxx, lux, luu,
          f_v, fx, fu, fxx, fux, fuu,
          eq_v, eq_x, eq_u, eq_xx, eq_ux, eq_uu,
          xu), zipped);
        // clang-format on

        DDP_ASSERT(not xu.x().hasNaN());
        DDP_ASSERT(not xu.x_next().hasNaN());
        DDP_ASSERT(not xu.u().hasNaN());

        auto t = xu.current_index();
        auto x = xu.x();
        auto u = xu.u();

        lx.get().setZero();
        lxx.get().setZero();
        lux.get().setZero();
        lu.get() = c * xu.u().transpose();
        luu.get().setIdentity();
        luu.get() *= c;

        {
          auto msg = fmt::format("  computing f  derivatives from thread {}", thread_id);
          ddp::chronometer_t timer(msg.c_str());
          k = dynamics().second_order_deriv(
              fxx.get(),
              fux.get(),
              fuu.get(),
              fx.get(),
              fu.get(),
              f_v.get(),
              t,
              x,
              u,
              DDP_MOVE(k));
        }
        {
          auto msg = fmt::format("  computing eq derivatives from thread {}", thread_id);
          ddp::chronometer_t timer(msg.c_str());
          k = constraint().second_order_deriv(
              eq_xx.get(),
              eq_ux.get(),
              eq_uu.get(),
              eq_x.get(),
              eq_u.get(),
              eq_v.get(),
              t,
              x,
              u,
              DDP_MOVE(k));
        }

#if not defined(NDEBUG)
        {
          using vec_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
          DDP_ASSERT(not eq_v.get().hasNaN());
          DDP_ASSERT(not eq_x.get().hasNaN());
          DDP_ASSERT(not eq_u.get().hasNaN());
          DDP_ASSERT(not eq_xx.get().has_nan());
          DDP_ASSERT(not eq_ux.get().has_nan());
          DDP_ASSERT(not eq_uu.get().has_nan());

          DDP_ASSERT(not fx.get().hasNaN());
          DDP_ASSERT(not fu.get().hasNaN());
          DDP_ASSERT(not fxx.get().has_nan());
          DDP_ASSERT(not fux.get().has_nan());
          DDP_ASSERT(not fuu.get().has_nan());

          auto ne = eq_v.get().rows();

          scalar_t l;
          scalar_t l_;
          auto f = f_v.get().eval();
          auto f_ = f_v.get().eval();
          auto eq = eq_v.get().eval();
          auto eq_ = eq_v.get().eval();

          scalar_t _dl1;
          auto _df1 = eigen::make_matrix<scalar_t>(dynamics().dstate_dim());
          auto _deq1 = eq_v.get().eval();

          scalar_t _ddl1;
          auto _ddf1 = eigen::make_matrix<scalar_t>(dynamics().dstate_dim());
          vec_t _ddeq1{ne};

          scalar_t _dddl1;
          auto _dddf1 = eigen::make_matrix<scalar_t>(dynamics().dstate_dim());
          vec_t _dddeq1{ne};

          scalar_t _dl2;
          auto _df2 = eigen::make_matrix<scalar_t>(dynamics().dstate_dim());
          auto _deq2 = eq_v.get().eval();

          scalar_t _ddl2;
          auto _ddf2 = eigen::make_matrix<scalar_t>(dynamics().dstate_dim());
          vec_t _ddeq2{ne};

          scalar_t _dddl2;
          auto _dddf2 = eigen::make_matrix<scalar_t>(dynamics().dstate_dim());
          vec_t _dddeq2{ne};

          auto* dl = &_dl1;
          auto* df = &_df1;
          auto* deq = &_deq1;

          auto* ddl = &_ddl1;
          auto* ddf = &_ddf1;
          auto* ddeq = &_ddeq1;

          auto* dddl = &_dddl1;
          auto* dddf = &_dddf1;
          auto* dddeq = &_dddeq1;

          auto dx_ = eigen::make_matrix<scalar_t>(dynamics().dstate_dim());
          dx_.setRandom();

          auto du_ = eigen::make_matrix<scalar_t>(dynamics().dcontrol_dim(t));
          du_.setRandom();

          auto finite_diff_err = [&](scalar_t eps_x, scalar_t eps_u) {
            auto dx = (dx_.operator*(eps_x)).eval();
            auto du = (du_.operator*(eps_u)).eval();

            auto x_ = x.eval();
            auto u_ = u.eval();

            dynamics().integrate_x(eigen::as_mut_view(x_), eigen::as_const_view(x), eigen::as_const_view(dx));
            dynamics().integrate_u(eigen::as_mut_view(u_), t, eigen::as_const_view(u), eigen::as_const_view(du));

            l = this->l(t, x, u);
            k = eval_f_to(eigen::as_mut_view(f), t, x, u, DDP_MOVE(k));
            k = eval_eq_to(eigen::as_mut_view(eq), t, x, u, DDP_MOVE(k));

            l_ = this->l(t, eigen::as_const_view(x_), eigen::as_const_view(u_));
            k = eval_f_to(eigen::as_mut_view(f_), t, eigen::as_const_view(x_), eigen::as_const_view(u_), DDP_MOVE(k));
            k = eval_eq_to(eigen::as_mut_view(eq_), t, eigen::as_const_view(x_), eigen::as_const_view(u_), DDP_MOVE(k));

            *dl = l_ - l;
            dynamics().difference_out(eigen::as_mut_view(*df), eigen::as_const_view(f), eigen::as_const_view(f_));
            constraint().difference_out(eigen::as_mut_view(*deq), eigen::as_const_view(eq), eigen::as_const_view(eq_));

            *ddl = *dl - (lx.get() * dx + lu.get() * du).value();
            *ddf = *df - (fx.get() * dx + fu.get() * du);
            *ddeq = *deq - (eq_x.get() * dx + eq_u.get() * du);

            *dddl = *ddl - (0.5 * dx.transpose() * lxx.get() * dx   //
                            + 0.5 * du.transpose() * luu.get() * du //
                            + du.transpose() * lux.get() * dx)
                               .value();
            *dddf = -*ddf;
            *dddeq = -*ddeq;

            ddp::add_second_order_term(*dddf, fxx.get(), fux.get(), fuu.get(), dx, du);
            ddp::add_second_order_term(*dddeq, eq_xx.get(), eq_ux.get(), eq_uu.get(), dx, du);

            *dddf = -*dddf;
            *dddeq = -*dddeq;
          };

          scalar_t eps_x = 1e-30;
          scalar_t eps_u = 1e-30;

          scalar_t eps_factor = 0.1;

          finite_diff_err(eps_x, eps_u);

          dl = &_dl2;
          df = &_df2;
          deq = &_deq2;

          ddl = &_ddl2;
          ddf = &_ddf2;
          ddeq = &_ddeq2;

          dddl = &_dddl2;
          dddf = &_dddf2;
          dddeq = &_dddeq2;

          finite_diff_err(eps_x * eps_factor, eps_u * eps_factor);

          auto dl1 = _dl1;
          auto dl2 = _dl2;
          auto ddl1 = _ddl1;
          auto ddl2 = _ddl2;
          auto dddl1 = _dddl1;
          auto dddl2 = _dddl2;

          auto df1 = _df1.array();
          auto df2 = _df2.array();
          auto ddf1 = _ddf1.array();
          auto ddf2 = _ddf2.array();
          auto dddf1 = _dddf1.array();
          auto dddf2 = _dddf2.array();

          auto deq1 = _deq1.array();
          auto deq2 = _deq2.array();
          auto ddeq1 = _ddeq1.array();
          auto ddeq2 = _ddeq2.array();
          auto dddeq1 = _dddeq1.array();
          auto dddeq2 = _dddeq2.array();

          using std::fabs;
          using std::pow;

          auto eps = pow(std::numeric_limits<scalar_t>::epsilon(), 0.9);

          DDP_EXPECT_MSG_ALL_OF(
              (fmt::format("{}\n{}", dl1, dl2), (fabs(dl1) <= eps or fabs(dl2) <= 2 * eps_factor * fabs(dl1))),
              (fmt::format("{}\n{}", ddl1, ddl2),
               (fabs(ddl1) <= eps or fabs(ddl2) <= 2 * pow(eps_factor, 2) * fabs(ddl1))),
              (fmt::format("{}\n{}", dddl1, dddl2),
               (fabs(dddl1) <= eps or fabs(dddl2) <= 2 * pow(eps_factor, 3) * fabs(dddl1))));

          for (index_t i = 0; i < df1.size(); ++i) {
            DDP_EXPECT_MSG_ALL_OF(
                (fmt::format("{}\n{}", df1[i], df2[i]),
                 (fabs(df1[i]) <= eps or fabs(df2[i]) <= 2 * eps_factor * fabs(df1[i]))),
                (fmt::format("{}\n{}", ddf1[i], ddf2[i]),
                 (fabs(ddf1[i]) <= eps or fabs(ddf2[i]) <= 2 * pow(eps_factor, 2) * fabs(ddf1[i]))),
                (fmt::format("{}\n{}", dddf1[i], dddf2[i]),
                 (fabs(dddf1[i]) <= eps or fabs(dddf2[i]) <= 2 * pow(eps_factor, 3) * fabs(dddf1[i]))));
          }

          for (index_t i = 0; i < deq1.size(); ++i) {
            DDP_EXPECT_MSG_ALL_OF(
                (fmt::format("{}\n{}", deq1[i], deq2[i]),
                 (fabs(deq1[i]) <= eps or fabs(deq2[i]) <= 2 * eps_factor * fabs(deq1[i]))),
                (fmt::format("{}\n{}", ddeq1[i], ddeq2[i]),
                 (fabs(ddeq1[i]) <= eps or fabs(ddeq2[i]) <= 2 * pow(eps_factor, 2) * fabs(ddeq1[i]))),
                (fmt::format("{}\n{}", dddeq1[i], dddeq2[i]),
                 (fabs(dddeq1[i]) <= eps or fabs(dddeq2[i]) <= 2 * pow(eps_factor, 3) * fabs(dddeq1[i]))));
          }
        }
#endif
      }
    }
  }

  auto name() const noexcept -> fmt::string_view { return m_dynamics.name(); }

  index_t m_begin{};
  index_t m_end{};
  scalar_t c = 1e2;
  dynamics_t m_dynamics;
  constraint_t m_constraint;
};

template <typename Problem, typename Slack_Idx>
struct multiple_shooting_t {
  using problem_t = Problem;
  using slack_indexer_t = Slack_Idx;
  using scalar_t = typename problem_t::scalar_t;

  using orig_dynamics_t = typename problem_t::dynamics_t;
  using orig_constraint_t = typename problem_t::constraint_t;

  using _1 = fix_index<1>;

  using state_indexer_t = typename problem_t::state_indexer_t;
  using dstate_indexer_t = typename problem_t::dstate_indexer_t;
  using control_indexer_t = indexing::row_concat_indexer_t<typename problem_t::control_indexer_t, Slack_Idx>;
  using dcontrol_indexer_t = indexing::row_concat_indexer_t<typename problem_t::dcontrol_indexer_t, Slack_Idx>;
  using eq_indexer_t = indexing::row_concat_indexer_t<typename problem_t::eq_indexer_t, Slack_Idx>;

  using state_kind = typename state_indexer_t::row_kind;
  using dstate_kind = typename dstate_indexer_t::row_kind;
  using control_kind = typename control_indexer_t::row_kind;
  using dcontrol_kind = typename dcontrol_indexer_t::row_kind;
  using eq_kind = typename eq_indexer_t::row_kind;

  using x_mut = typename problem_t::x_mut;
  using x_const = typename problem_t::x_const;
  using dx_const = typename problem_t::dx_const;
  using u_mut = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, control_indexer_t> const>;
  using u_const = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, control_indexer_t>>;
  using du_const = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, dcontrol_indexer_t>>;

  using f_const = typename problem_t::out_const;
  using f_mut = typename problem_t::out_mut;
  using df_mut = typename problem_t::dout_mut;

  using fx_mut = typename problem_t::out_x_mut;
  using fu_mut = eigen::
      matrix_from_idx_t<scalar_t, typename indexing::outer_prod_result<dstate_indexer_t, control_indexer_t>::type>;

  using fxx_mut = typename problem_t::out_xx_mut;
  using fux_mut = tensor::tensor_view_t<scalar_t, dstate_kind, dcontrol_kind, dstate_kind>;
  using fuu_mut = tensor::tensor_view_t<scalar_t, dstate_kind, dcontrol_kind, dcontrol_kind>;

  using eq_mut = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, eq_indexer_t>>;

  using eq_x_mut =
      eigen::matrix_from_idx_t<scalar_t, typename indexing::outer_prod_result<eq_indexer_t, dstate_indexer_t>::type>;
  using eq_u_mut =
      eigen::matrix_from_idx_t<scalar_t, typename indexing::outer_prod_result<eq_indexer_t, control_indexer_t>::type>;

  using eq_xx_mut = tensor::tensor_view_t<scalar_t, eq_kind, dstate_kind, dstate_kind>;
  using eq_ux_mut = tensor::tensor_view_t<scalar_t, eq_kind, dcontrol_kind, dstate_kind>;
  using eq_uu_mut = tensor::tensor_view_t<scalar_t, eq_kind, dcontrol_kind, dcontrol_kind>;

  using key = typename problem_t::key;

  struct dynamics_t {
    multiple_shooting_t const& m_parent;

    using scalar_t = multiple_shooting_t::scalar_t;
    using x_const = multiple_shooting_t::x_const;
    using dx_const = multiple_shooting_t::dx_const;
    using u_const = multiple_shooting_t::u_const;
    using du_const = multiple_shooting_t::du_const;

    using out_mut = multiple_shooting_t::f_mut;

    using out_x_mut = multiple_shooting_t::fx_mut;
    using out_u_mut = multiple_shooting_t::fu_mut;

    using out_xx_mut = multiple_shooting_t::fxx_mut;
    using out_ux_mut = multiple_shooting_t::fux_mut;
    using out_uu_mut = multiple_shooting_t::fuu_mut;

    using key = typename multiple_shooting_t::key;

    auto orig() const noexcept -> orig_dynamics_t { return m_parent.m_prob.dynamics(); }

    auto state_dim() const DDP_DECLTYPE_AUTO(m_parent.state_dim());
    auto dstate_dim() const DDP_DECLTYPE_AUTO(m_parent.dstate_dim());
    auto state_indexer(index_t begin, index_t end) const DDP_DECLTYPE_AUTO(m_parent.state_indexer(begin, end));
    auto dstate_indexer(index_t begin, index_t end) const DDP_DECLTYPE_AUTO(m_parent.dstate_indexer(begin, end));
    auto control_dim(index_t t) const DDP_DECLTYPE_AUTO(m_parent.control_dim(t));
    auto dcontrol_dim(index_t t) const DDP_DECLTYPE_AUTO(m_parent.dcontrol_dim(t));

    auto acquire_workspace() const noexcept -> key { return orig().dynamics().acquire_workspace(); }

    void neutral_configuration(x_mut out) const { orig().neutral_configuration(out); }
    void difference_out(x_mut out, x_const start, x_const finish) const { orig().difference_out(out, start, finish); }
    void d_difference_out_dfinish(out_x_mut out, x_const start, x_const finish) const {
      orig().d_difference_out_dfinish(out, start, finish);
    }

    auto eval_to(out_mut x_out, index_t t, x_const x, u_const u, key k) const -> key {
      return m_parent.eval_f_to(x_out, t, x, u, DDP_MOVE(k));
    }

    auto first_order_deriv( //
        out_x_mut fx,       //
        out_u_mut fu,       //
        out_mut f,          //
        index_t t,          //
        x_const x,          //
        u_const u,          //
        key k               //
    ) const -> key {
      DDP_BIND(auto, (fu_orig, fu_slack), eigen::split_at_col_mut(fu, orig().dcontrol_dim(t)));
      DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, orig().control_dim(t)));

      k = orig().first_order_deriv(fx, fu_orig, f, t, x, u_orig, DDP_MOVE(k));
      if (u_slack.rows() == 0) {
        return k;
      }

      DDP_BIND(auto, (fu_slack_x, fu_slack_0), eigen::split_at_col_mut(fu_slack, dstate_dim()));
      DDP_BIND(auto, (u_slack_x, u_slack_0), eigen::split_at_row(u_slack, state_dim()));
      DDP_ASSERT(fu_slack_0.cols() == 0);
      DDP_ASSERT(u_slack_0.rows() == 0);

      auto _tmp0 = f.eval();
      auto _tmp1 = fx.eval();
      auto _tmp2 = fx.eval();
      auto tmp1 = eigen::as_mut_view(_tmp1);

      k = orig().first_order_deriv(fx, fu_orig, f, t, x, u_orig, DDP_MOVE(k));
      orig().d_integrate_x(tmp1, eigen::as_const_view(f), u_slack_x);
      orig().d_integrate_x_dx(fu_slack_x, eigen::as_const_view(f), u_slack_x);

      fx = tmp1 * fx;
      fu_orig = tmp1 * fu_orig;

      return eval_to(f, t, x, u, DDP_MOVE(k));
    }

    auto second_order_deriv( //
        out_xx_mut fxx,      //
        out_ux_mut fux,      //
        out_uu_mut fuu,      //
        out_x_mut fx,        //
        out_u_mut fu,        //
        out_mut f,           //
        index_t t,           //
        x_const x,           //
        u_const u,           //
        key k                //
    ) const -> key {
      return finite_diff_hessian_compute<dynamics_t>{*this, orig().second_order_finite_diff()}
          .second_order_deriv(fxx, fux, fuu, fx, fu, f, t, x, u, DDP_MOVE(k));
    }
    auto second_order_finite_diff() const noexcept -> bool { return orig().second_order_finite_diff(); }

    void integrate_x(x_mut out, x_const x, dx_const dx) const { orig().integrate_x(out, x, dx); }
    void integrate_u(u_mut out, index_t t, u_const u, u_const du) const {
      DDP_BIND(auto, (out_orig, out_slack), eigen::split_at_row_mut(out, orig().control_dim(t)));
      DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, orig().control_dim(t)));
      DDP_BIND(auto, (du_orig, du_slack), eigen::split_at_row(du, orig().dcontrol_dim(t)));
      orig().integrate_u(out_orig, t, u_orig, du_orig);
      out_slack = u_slack + du_slack;
    }

    void d_integrate_x(out_x_mut, x_const x, dx_const dx) const;
    void d_integrate_x_dx(out_x_mut, x_const x, dx_const dx) const;
  };
  struct constraint_t {
    multiple_shooting_t const& m_parent;

    using scalar_t = multiple_shooting_t::scalar_t;
    using x_const = multiple_shooting_t::x_const;
    using dx_const = multiple_shooting_t::dx_const;
    using u_const = multiple_shooting_t::u_const;
    using du_const = multiple_shooting_t::du_const;

    using out_mut = multiple_shooting_t::f_mut;

    using out_x_mut = multiple_shooting_t::fx_mut;
    using out_u_mut = multiple_shooting_t::fu_mut;

    using out_xx_mut = multiple_shooting_t::fxx_mut;
    using out_ux_mut = multiple_shooting_t::fux_mut;
    using out_uu_mut = multiple_shooting_t::fuu_mut;

    using key = typename multiple_shooting_t::key;

    auto orig() const noexcept -> orig_constraint_t { return m_parent.m_prob.constraint(); }

    auto eq_idx() const -> eq_indexer_t { return indexing::row_concat(orig().eq_idx(), m_parent.m_slack_idx.clone()); }
    auto eq_dim(index_t t) const -> typename eq_indexer_t::row_kind {
      return eq_idx().rows(t) + m_parent.m_slack_idx.rows(t);
    }

    void integrate_x(x_mut out, x_const x, dx_const dx) const { m_parent.dynamics().integrate_x(out, x, dx); }
    void integrate_u(u_mut out, index_t t, u_const u, u_const du) const {
      m_parent.dynamics().integrate_u(out, t, u, du);
    }
    template <typename Out, typename In>
    void difference_out(Out out, In start, In finish) const {
      DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
          ("", out.rows() == start.rows()), //
          ("", out.rows() == finish.rows()));
      out = finish - start;
    }

    auto first_order_deriv( //
        out_x_mut out_x,    //
        out_u_mut out_u,    //
        out_mut out,        //
        index_t t,          //
        x_const x,          //
        u_const u,          //
        key k               //
    ) const -> key {
      auto ne = orig().eq_dim(t);
      auto nx = m_parent.dstate_dim();

      DDP_BIND(auto, (out_x_orig, out_x_slack), eigen::split_at_mut(out_x, ne));
      DDP_BIND(
          auto,
          (out_u_orig_orig, out_u_orig_slack, out_u_slack_orig, out_u_slack_slack),
          eigen::split_at_mut(out_u, ne, nx));

      DDP_BIND(auto, (out_orig, out_slack), eigen::split_at_mut(out, ne));

      DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, nx));

      out_u_orig_slack.setZero();
      out_u_slack_orig.setZero();
      out_u_slack_slack.setIdentity();
      return orig().first_order_deriv(out_x_orig, out_u_orig_orig, out_orig, t, x, u_orig, DDP_MOVE(k));
    }

    auto eval_to(out_mut out, index_t t, x_const x, u_const u, key k) const -> key {
      return m_parent.eval_eq_to(out, t, x, u, DDP_MOVE(k));
    }
  };

  auto dynamics() const noexcept -> dynamics_t { return {*this}; }
  auto constraint() const noexcept -> constraint_t { return {*this}; }

  auto state_dim() const -> state_kind { return m_prob.state_dim(); }
  auto dstate_dim() const noexcept -> dstate_kind { return m_prob.dstate_dim(); }

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t { return m_prob.state_indexer(begin, end); }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return m_prob.dstate_indexer(begin, end);
  }

  auto control_dim(index_t t) const noexcept -> control_kind { return m_prob.control_dim(t) + m_slack_idx.rows(t); }
  auto dcontrol_dim(index_t t) const noexcept -> dcontrol_kind { return m_prob.dcontrol_dim(t) + m_slack_idx.rows(t); }

  void neutral_configuration(x_mut out) const { m_prob.neutral_configuration(out); }

  auto lf(x_const x) const -> scalar_t { return m_prob.lf(x); }
  auto l(index_t t, x_const x, u_const u) const -> scalar_t { return m_prob.l(t, x, u); }
  void difference(f_mut out, f_const start, f_const finish) const { m_prob.difference(out, start, finish); }
  void d_difference_dfinish(fx_mut out, f_const start, f_const finish) const {
    m_prob.d_difference_dfinish(out, start, finish);
  }

  auto eval_f_to(x_mut x_out, index_t t, x_const x, u_const u, key k = {}) const -> key {
    DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, m_prob.control_dim(t)));

    if (u_slack.rows() == 0) {
      k = m_prob.dynamics().eval_to(x_out, t, x, u_orig, DDP_MOVE(k));
    } else {
      auto _tmp = x_out.eval();
      auto tmp = eigen::as_mut_view(_tmp);
      k = m_prob.dynamics().eval_to(tmp, t, x, u_orig, DDP_MOVE(k));
      DDP_BIND(auto, (u_slack_x, u_slack_0), eigen::split_at_row(u_slack, dstate_dim()));
      DDP_ASSERT(u_slack_0.rows() == 0);
      dynamics().integrate_x(x_out, eigen::as_const_view(tmp), u_slack_x);
    }
    return k;
  }

  auto eval_eq_to(eq_mut eq_out, index_t t, x_const x, u_const u, key k = {}) const -> key {
    DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, m_prob.control_dim(t)));
    DDP_BIND(auto, (out_orig, out_slack), eigen::split_at_row(eq_out, m_prob.constraint().eq_dim(t)));

    k = m_prob.dynamics().eval_to(out_orig, t, x, u_orig, DDP_MOVE(k));
    if (u_slack.rows() > 0) {
      out_slack = u_slack;
    }
    return k;
  }

  auto name() const -> std::string { return std::string{"multi_shooting_"} + m_prob.name(); }
  problem_t m_prob;
  slack_indexer_t m_slack_idx;
};

} // namespace ddp

#endif /* end of include guard PROBLEM_HPP_SQUCCNMX */
