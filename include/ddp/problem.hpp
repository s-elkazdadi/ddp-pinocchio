#ifndef PROBLEM_HPP_SQUCCNMX
#define PROBLEM_HPP_SQUCCNMX

#include "ddp/detail/utils.hpp"
#include "ddp/detail/tensor.hpp"
#include "ddp/trajectory.hpp"
#include "ddp/ddp.hpp"

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

  void second_order_deriv_1(
      out_xx_mut out_xx,
      out_ux_mut out_ux,
      out_uu_mut out_uu,
      out_x_mut out_x,
      out_u_mut out_u,
      out_mut out,
      index_t t,
      x_const x,
      u_const u) const {

    DDP_ASSERT_MSG_ALL_OF(
        ("non commutative groups are not supported", out_x.rows() == out.rows()),
        ("non commutative groups are not supported", out_x.cols() == x.rows()),
        ("non commutative groups are not supported", out_u.cols() == u.rows()));

    auto no = out_x.rows();
    auto nx = out_x.cols();
    auto nu = out_u.cols();

    fn.first_order_deriv(out_x, out_u, out, t, x, u);
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
        fn.integrate_u(u_, u, eigen::as_const_view(du));

        fn.first_order_deriv(
            eigen::as_mut_view(fx_),
            eigen::as_mut_view(fu_),
            out_,
            t,
            eigen::as_const_view(x_),
            eigen::as_const_view(u_));

        if (at_x) {

          for (index_t k = 0; k < no; ++k) {
            for (index_t j = 0; j < nx; ++j) {
              out_xx(k, j, idx) = (fx_(k, j) - out_x(k, j)) / eps;
            }
            for (index_t j = 0; j < nu; ++j) {
              out_ux(k, j, idx) = (fu_(k, j) - out_u(k, j)) / eps;
            }
          }

        } else {
          for (index_t k = 0; k < no; ++k) {
            for (index_t j = 0; j < nu; ++j) {
              out_uu(k, j, idx) = (fu_(k, j) - out_u(k, j)) / eps;
            }
          }
        }

        in_var = 0;
      }
    }
  }

  void second_order_deriv_2(
      out_xx_mut out_xx,
      out_ux_mut out_ux,
      out_uu_mut out_uu,
      out_x_mut out_x,
      out_u_mut out_u,
      out_mut out,
      index_t t,
      x_const x,
      u_const u) const {

    auto no = out_x.rows();
    auto nx = out_x.cols();
    auto nu = out_u.cols();

    fn.first_order_deriv(out_x, out_u, out, t, x, u);
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
        fn.integrate_u(u1, u, eigen::as_const_view(du));

        fn.eval_to(f1, t, eigen::as_const_view(x1), eigen::as_const_view(u1));

        // f(x + dx) - f(x)
        fn.difference_out(df, f0, eigen::as_const_view(f1));

        // (f(x + dx) - f(x)) - J dx
        // dx = eps * e_i => J dx = eps * J.col(i)
        df -= eps * f_col;

        // 2 * ((f(x + dx) - f(x)) - J dx)
        df *= 2;

        for (index_t k = 0; k < no; ++k) {
          tensor(k, idx, idx) = df[k] / eps2;
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
          fn.integrate_u(u1, u, eigen::as_const_view(du));

          fn.eval_to(f1, t, eigen::as_const_view(x1), eigen::as_const_view(u1));

          // f(x + dx) - f(x)
          fn.difference_out(df, f0, eigen::as_const_view(f1));

          // (f(x + dx) - f(x)) - J dx
          // dx = eps * e_i => J dx = eps * J.col(i)
          df -= eps * f_col_1;
          df -= eps * f_col_2;

          // 2 * ((f(x + dx) - f(x)) - J dx)
          df *= 2;

          // [v1, f1] == f(x + eps * ei + eps * ej) - f(x)

          for (index_t k = 0; k < no; ++k) {
            tensor(k, idx_2, idx_1) =              //
                0.5 * (df[k] / eps2                //
                       - tensor_1(k, idx_1, idx_1) //
                       - tensor_2(k, idx_2, idx_2));

            if (at_x_1 == at_x_2) {
              tensor(k, idx_1, idx_2) = tensor(k, idx_2, idx_1);
            }
          }

          in_var_2 = 0;
        }

        in_var_1 = 0;
      }
    }
  }

  void second_order_deriv(
      out_xx_mut out_xx,
      out_ux_mut out_ux,
      out_uu_mut out_uu,
      out_x_mut out_x,
      out_u_mut out_u,
      out_mut out,
      index_t t,
      x_const x,
      u_const u) const {
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
      return second_order_deriv_2(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u);
    } else {
      return second_order_deriv_1(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u);
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
  using du_const = eigen::matrix_view_t<scalar_t const, control_kind, _1>;

  using out_const = eigen::matrix_view_t<scalar_t const, state_kind, _1>;
  using out_mut = eigen::matrix_view_t<scalar_t, state_kind, _1>;
  using dout_mut = eigen::matrix_view_t<scalar_t, dstate_kind, _1>;

  using out_x_mut = eigen::matrix_view_t<scalar_t, dstate_kind, dstate_kind>;
  using out_u_mut = eigen::matrix_view_t<scalar_t, dstate_kind, control_kind>;

  using out_xx_mut = tensor::tensor_view_t<scalar_t, dstate_kind, dstate_kind, dstate_kind>;
  using out_ux_mut = tensor::tensor_view_t<scalar_t, dstate_kind, control_kind, dstate_kind>;
  using out_uu_mut = tensor::tensor_view_t<scalar_t, dstate_kind, control_kind, control_kind>;

  auto state_dim() const -> state_kind { return m_model.configuration_dim_c() + m_model.tangent_dim_c(); }
  auto dstate_dim() const noexcept -> dstate_kind { return m_model.tangent_dim_c() + m_model.tangent_dim_c(); }
  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.configuration_dim_c() + m_model.tangent_dim_c());
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.tangent_dim_c() + m_model.tangent_dim_c());
  }
  auto control_dim(index_t) const noexcept -> control_kind { return m_model.tangent_dim_c(); }
  auto dcontrol_dim(index_t) const noexcept -> dcontrol_kind { return m_model.tangent_dim_c(); }

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
  void integrate_u(u_mut out, u_const u, u_const du) const { out = u + du; }
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

  void eval_to(out_mut x_out, index_t t, x_const x, u_const u) const {
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
    m_model.dynamics_aba(v_out, q, v, u);

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    v_out = v + v_out * dt;
  }

  void first_order_deriv( //
      out_x_mut fx,       //
      out_u_mut fu,       //
      out_mut f,          //
      index_t t,          //
      x_const x,          //
      u_const u           //
  ) const {
    eval_to(f, t, x, u);

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
    m_model.d_dynamics_aba(fx_bot_left, fx_bot_right, fu_bot, q, v, u);

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    fx_bot_left *= dt;
    fx_bot_right *= dt;
    fx_bot_right += decltype(fx_bot_right.eval())::Identity(nv.value(), nv.value());
    fu_bot *= dt;
  }

  void second_order_deriv( //
      out_xx_mut fxx,      //
      out_ux_mut fux,      //
      out_uu_mut fuu,      //
      out_x_mut fx,        //
      out_u_mut fu,        //
      out_mut f,           //
      index_t t,           //
      x_const x,           //
      u_const u            //
  ) const {
    finite_diff_hessian_compute<dynamics_t>{*this, second_order_finite_diff}
        .second_order_deriv(fxx, fux, fuu, fx, fu, f, t, x, u);
  }

  auto name() const noexcept -> fmt::string_view { return m_model.model_name(); }

  model_t const& m_model;
  scalar_t dt;
  bool second_order_finite_diff = true;
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

  using _1 = fix_index<1>;

  auto eq_idx() const -> eq_indexer_t { return {m_constraint.eq_idx(), 1}; }
  auto eq_dim(index_t t) const -> typename eq_indexer_t::row_kind { return m_constraint.eq_dim(t + 1); }

  void integrate_x(x_mut out, x_const x, dx_const dx) const { m_constraint.integrate_x(out, x, dx); }
  void integrate_u(u_mut out, u_const u, u_const du) const { m_constraint.integrate_u(out, u, du); }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    m_constraint.difference_out(out, start, finish);
  }

  void eval_to(out_mut out, index_t t, x_const x, u_const u) const {
    auto x_n = eigen::make_matrix<scalar_t>(m_dynamics.state_dim(), _1{}); // x_{t+1}
    m_dynamics.eval_to(eigen::as_mut_view(x_n), t, x, u);
    m_constraint.eval_to(out, t + 1, eigen::as_const_view(x_n), u);
  }

  void first_order_deriv( //
      out_x_mut out_x,    //
      out_u_mut out_u,    //
      out_mut out,        //
      index_t t,          //
      x_const x,          //
      u_const u           //
  ) const {

    if (out.rows() == 0) {
      return;
    }

    auto nx = m_dynamics.state_dim();
    auto nx_ = m_dynamics.dstate_dim();

    auto _x_n = eigen::make_matrix<scalar_t>(nx, _1{}); // x_{t+1}
    auto x_n = eigen::as_mut_view(_x_n);

    auto _fx_n = eigen::make_matrix<scalar_t>(nx_, nx_);
    auto _fu_n = eigen::make_matrix<scalar_t>(nx_, m_dynamics.dcontrol_dim(t));
    auto fx_n = eigen::as_mut_view(_fx_n);
    auto fu_n = eigen::as_mut_view(_fu_n);

    m_dynamics.first_order_deriv(fx_n, fu_n, x_n, t, x, u);

    auto _eq_n_x = eigen::make_matrix<scalar_t>(eigen::rows_c(out), nx_);
    auto _eq_n_u = eigen::make_matrix<scalar_t>(eigen::rows_c(out), m_dynamics.dcontrol_dim(t + 1));
    auto eq_n_x = eigen::as_mut_view(_eq_n_x);
    auto eq_n_u = eigen::as_mut_view(_eq_n_u);

    m_constraint.first_order_deriv(eq_n_x, eq_n_u, out, t + 1, eigen::as_const_view(x_n), u);
    DDP_ASSERT_MSG("constraint depends on control", eq_n_u.isConstant(0));

    out_x.noalias() = eq_n_x * fx_n;
    out_u.noalias() = eq_n_x * fu_n;
  }

  void second_order_deriv( //
      out_xx_mut out_xx,   //
      out_ux_mut out_ux,   //
      out_uu_mut out_uu,   //
      out_x_mut out_x,     //
      out_u_mut out_u,     //
      out_mut out,
      index_t t, //
      x_const x, //
      u_const u  //
  ) const {
    finite_diff_hessian_compute<constraint_advance_time_t>{*this, m_dynamics.second_order_finite_diff}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u);
  }

  Constraint m_constraint;
  dynamics_t m_dynamics;
};

template <typename Constraint>
auto constraint_advance_time(Constraint c) -> constraint_advance_time_t<Constraint> {
  return {c, c.m_dynamics};
}

template <typename Model, typename Constraint_Target_View>
struct config_constraint_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using state_kind = typename dynamics_t::state_kind;
  using dstate_kind = typename dynamics_t::dstate_kind;
  using control_kind = decltype(static_cast<model_t const*>(nullptr)->tangent_dim_c());

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

  void integrate_x(x_mut out, x_const x, dx_const dx) const { m_dynamics.integrate_x(out, x, dx); }
  void integrate_u(u_mut out, u_const u, u_const du) const { m_dynamics.integrate_u(out, u, du); }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
        ("", out.rows() == start.rows()), //
        ("", out.rows() == finish.rows()));
    out = finish - start;
  }

  void eval_to(out_mut out, index_t t, x_const x, u_const u) const {
    auto target = eigen::as_const_view(m_constraint_target_view[t + 2]);
    if (target.rows() == 0) {
      return;
    }

    auto nq = m_dynamics.m_model.configuration_dim_c();
    auto nv = m_dynamics.m_model.tangent_dim_c();

    auto x_n = eigen::make_matrix<scalar_t>(nq + nv, _1{});  // x_{t+1}
    auto x_nn = eigen::make_matrix<scalar_t>(nq + nv, _1{}); // x_{t+2}
    m_dynamics.eval_to(eigen::as_mut_view(x_n), t, x, u);
    m_dynamics.eval_to(eigen::as_mut_view(x_nn), t + 1, eigen::as_const_view(x_n), u);
    difference_out(
        out,
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x_nn, nq)) // configuration part of x_nn
    );
  }

  void first_order_deriv( //
      out_x_mut out_x,    //
      out_u_mut out_u,    //
      out_mut out,        //
      index_t t,          //
      x_const x,          //
      u_const u           //
  ) const {
    auto target = eigen::as_const_view(m_constraint_target_view[t + 2]);
    DDP_ASSERT_MSG(fmt::format("at t = {}", t), target.rows() == out.rows());
    if (target.rows() == 0) {
      return;
    }

    auto const& m_model = m_dynamics.m_model;

    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();

    auto _x_n = eigen::make_matrix<scalar_t>(nq + nv, _1{}); // x_{t+1}
    auto x_n = eigen::as_mut_view(_x_n);
    auto _x_nn = eigen::make_matrix<scalar_t>(nq + nv, _1{}); // x_{t+2}
    auto x_nn = eigen::as_mut_view(_x_nn);

    auto _fx_n = eigen::make_matrix<scalar_t>(nv + nv, nv + nv);
    auto fx_n = eigen::as_mut_view(_fx_n);
    auto _fu_n = eigen::make_matrix<scalar_t>(nv + nv, nv);
    auto fu_n = eigen::as_mut_view(_fu_n);
    auto _fx_nn = eigen::make_matrix<scalar_t>(nv + nv, nv + nv);
    auto fx_nn = eigen::as_mut_view(_fx_nn);
    auto _fu_nn = eigen::make_matrix<scalar_t>(nv + nv, nv);
    auto fu_nn = eigen::as_mut_view(_fu_nn);

    m_dynamics.first_order_deriv(fx_n, fu_n, x_n, t, x, u);
    m_dynamics.first_order_deriv(fx_nn, fu_nn, x_nn, t, eigen::as_const_view(x_n), u);

    difference_out(
        out,
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x_nn, nq)) // configuration part of x_nn
    );

    auto d_diff = eigen::make_matrix<scalar_t>(nv, nv);
    m_dynamics.m_model.d_difference_dq_finish(
        eigen::as_mut_view(d_diff),
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x_nn, nq)) // configuration part of x_nn
    );

    out_x.noalias() = d_diff * (fx_nn * fx_n).template topRows<decltype(nv)::value_at_compile_time>(nv.value());
    out_u.noalias() = d_diff * (fx_nn * fu_n).template topRows<decltype(nv)::value_at_compile_time>(nv.value());
  }

  void second_order_deriv( //
      out_xx_mut out_xx,   //
      out_ux_mut out_ux,   //
      out_uu_mut out_uu,   //
      out_x_mut out_x,     //
      out_u_mut out_u,     //
      out_mut out,
      index_t t, //
      x_const x, //
      u_const u  //
  ) const {
    finite_diff_hessian_compute<config_constraint_t>{*this, m_dynamics.second_order_finite_diff}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u);
  }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
};

template <typename Model, typename Constraint_Target_View>
struct config_constraint_t0 {
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

  auto eq_idx() const -> eq_indexer_t { return m_constraint_target_view.eq_idx(); }
  auto eq_dim(index_t t) const -> typename eq_indexer_t::row_kind { return eq_idx().rows(t); }

  void integrate_x(x_mut out, x_const x, dx_const dx) const { m_dynamics.integrate_x(out, x, dx); }
  void integrate_u(u_mut out, u_const u, u_const du) const { m_dynamics.integrate_u(out, u, du); }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
        ("", out.rows() == start.rows()), //
        ("", out.rows() == finish.rows()));
    out = finish - start;
  }

  void eval_to(out_mut out, index_t t, x_const x, u_const u) const {
    auto target = eigen::as_const_view(m_constraint_target_view[t + 2]);
    if (target.rows() == 0) {
      return;
    }

    (void)u;
    auto nq = m_dynamics.m_model.configuration_dim_c();

    difference_out(
        out,
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x, nq)) // configuration part of x_nn
    );
  }

  void first_order_deriv( //
      out_x_mut out_x,    //
      out_u_mut out_u,    //
      out_mut out,        //
      index_t t,          //
      x_const x,          //
      u_const u           //
  ) const {
    (void)u;
    auto target = eigen::as_const_view(m_constraint_target_view[t + 2]);
    DDP_ASSERT_MSG(fmt::format("at t = {}", t), target.rows() == out.rows());
    if (target.rows() == 0) {
      return;
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
  }

  void second_order_deriv( //
      out_xx_mut out_xx,   //
      out_ux_mut out_ux,   //
      out_uu_mut out_uu,   //
      out_x_mut out_x,     //
      out_u_mut out_u,     //
      out_mut out,
      index_t t, //
      x_const x, //
      u_const u  //
  ) const {
    finite_diff_hessian_compute<config_constraint_t0>{*this, m_dynamics.second_order_finite_diff}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u);
  }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
};

template <typename Dynamics, typename Constraint_Target_View>
auto config_constraint(Dynamics d, Constraint_Target_View v)
    -> config_constraint_t<typename Dynamics::model_t, Constraint_Target_View> {
  return {d, v};
}

template <typename Model, typename Constraint_Target_Range>
struct problem_t {
  using model_t = Model;
  using scalar_t = typename model_t::scalar_t;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using constraint_target_view_t = Constraint_Target_Range const&;
  using constraint_t = config_constraint_t<model_t, constraint_target_view_t>;

  using state_kind = typename dynamics_t::state_kind;
  using dstate_kind = typename dynamics_t::dstate_kind;
  using control_kind = typename dynamics_t::control_kind;
  using dcontrol_kind = typename dynamics_t::dcontrol_kind;

  using _1 = fix_index<1>;

  using state_indexer_t = indexing::regular_indexer_t<state_kind>;
  using dstate_indexer_t = indexing::regular_indexer_t<dstate_kind>;
  using control_indexer_t = indexing::regular_indexer_t<control_kind>;
  using eq_indexer_t = decltype(std::declval<Constraint_Target_Range const&>().eq_idx());
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

  using eq_const = typename constraint_t::out_const;
  using eq_mut = typename constraint_t::out_mut;
  using deq_mut = typename constraint_t::dout_mut;

  using eq_x_mut = typename constraint_t::out_x_mut;
  using eq_u_mut = typename constraint_t::out_u_mut;

  using eq_xx_mut = typename constraint_t::out_xx_mut;
  using eq_ux_mut = typename constraint_t::out_ux_mut;
  using eq_uu_mut = typename constraint_t::out_uu_mut;

  auto state_dim() const -> state_kind { return m_model.configuration_dim_c() + m_model.tangent_dim_c(); }
  auto dstate_dim() const noexcept -> dstate_kind { return m_model.tangent_dim_c() + m_model.tangent_dim_c(); }

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.configuration_dim_c() + m_model.tangent_dim_c());
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.tangent_dim_c() + m_model.tangent_dim_c());
  }

  void neutral_configuration(x_mut out) const {
    DDP_ASSERT_MSG("out vector does not have the correct size", out.size() == state_dim().value());
    auto nq = m_model.configuration_dim_c();
    DDP_BIND(auto, (out_q, out_v), eigen::split_at_row_mut(out, nq));
    m_model.neutral_configuration(out_q);
    out_v.setZero();
  }

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

  auto dynamics(bool second_order_finite_diff = true) const -> dynamics_t {
    return dynamics_t{
        m_model,
        dt,
        second_order_finite_diff,
    };
  }
  auto eq() const -> constraint_t {
    return constraint_t{
        dynamics(),
        constraint_target_view_t{m_eq_target},
    };
  }

  void difference(f_mut out, f_const start, f_const finish) const { dynamics().difference_out(out, start, finish); }
  void d_difference_dfinish(fx_mut out, f_const start, f_const finish) const {
    dynamics().d_difference_out_dfinish(out, start, finish);
  }

  void eval_f_to(x_mut x_out, index_t t, x_const x, u_const u) const { dynamics().eval_to(x_out, t, x, u); }
  void eval_eq_to(eq_mut eq_out, index_t t, x_const x, u_const u) const { eq().eval_to(eq_out, t, x, u); }

  using derivative_storage_t =
      ddp::derivative_storage_t<scalar_t, control_indexer_t, eq_indexer_t, state_indexer_t, dstate_indexer_t>;
  using trajectory_t = trajectory::trajectory_t<scalar_t, state_indexer_t, control_indexer_t>;

  auto compute_derivatives(
      derivative_storage_t& derivs, trajectory_t const& traj, bool second_order_finite_diff = false) const {

    derivs.lfx.setZero();
    derivs.lfxx.setZero();

    // clang-format off
    for (auto zipped : ranges::zip(
          derivs.lx, derivs.lu, derivs.lxx, derivs.lux, derivs.luu,
          derivs.f_val, derivs.fx, derivs.fu, derivs.fxx, derivs.fux, derivs.fuu,
          derivs.eq_val, derivs.eq_x, derivs.eq_u, derivs.eq_xx, derivs.eq_ux, derivs.eq_uu,
          traj)) {
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
        ddp::chronometer_t timer("computing f derivatives");
        dynamics(second_order_finite_diff)
            .second_order_deriv(fxx.get(), fux.get(), fuu.get(), fx.get(), fu.get(), f_v.get(), t, x, u);
      }
      {
        ddp::chronometer_t timer("computing eq derivatives");
        eq().second_order_deriv(eq_xx.get(), eq_ux.get(), eq_uu.get(), eq_x.get(), eq_u.get(), eq_v.get(), t, x, u);
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

        auto nv = m_model.tangent_dim();
        auto ne = eq_v.get().rows();

        auto nv_c = m_model.tangent_dim_c();

        scalar_t l;
        scalar_t l_;
        auto f = f_v.get().eval();
        auto f_ = f_v.get().eval();
        auto eq = eq_v.get().eval();
        auto eq_ = eq_v.get().eval();

        scalar_t eps_x = 1e-30;
        scalar_t eps_u = 1e-30;

        auto dx = (eps_x * decltype(eigen::make_matrix<scalar_t>(nv_c + nv_c, _1{}))::Random(nv + nv)).eval();
        auto du = (eps_u * decltype(eigen::make_matrix<scalar_t>(nv_c, _1{}))::Random(nv)).eval();

        auto x_ = x.eval();
        auto u_ = u.eval();

        dynamics().integrate_x(eigen::as_mut_view(x_), eigen::as_const_view(x), eigen::as_const_view(dx));
        dynamics().integrate_u(eigen::as_mut_view(u_), eigen::as_const_view(u), eigen::as_const_view(du));

        l = this->l(t, x, u);
        eval_f_to(eigen::as_mut_view(f), t, x, u);
        eval_eq_to(eigen::as_mut_view(eq), t, x, u);

        l_ = this->l(t, eigen::as_const_view(x_), eigen::as_const_view(u_));
        eval_f_to(eigen::as_mut_view(f_), t, eigen::as_const_view(x_), eigen::as_const_view(u_));
        eval_eq_to(eigen::as_mut_view(eq_), t, eigen::as_const_view(x_), eigen::as_const_view(u_));

        scalar_t dl;

        auto df = decltype(fu.get().col(0).eval())::Zero(nv + nv).eval();
        auto deq = eq_v.get().eval();

        scalar_t ddl;
        vec_t ddf{nv + nv};
        vec_t ddeq{ne};

        scalar_t dddl;
        vec_t dddf{nv + nv};
        vec_t dddeq{ne};

        dl = l_ - l;
        dynamics().difference_out(eigen::as_mut_view(df), eigen::as_const_view(f), eigen::as_const_view(f_));
        this->eq().difference_out(eigen::as_mut_view(deq), eigen::as_const_view(eq), eigen::as_const_view(eq_));

        ddl = dl - (lx.get() * dx + lu.get() * du).value();
        ddf = df - (fx.get() * dx + fu.get() * du);
        ddeq = deq - (eq_x.get() * dx + eq_u.get() * du);

        dddl = ddl - (0.5 * dx.transpose() * lxx.get() * dx   //
                      + 0.5 * du.transpose() * luu.get() * du //
                      + du.transpose() * lux.get() * dx)
                         .value();
        dddf = -ddf;
        dddeq = -ddeq;

        add_second_order_term(dddf, fxx.get(), fux.get(), fuu.get(), dx, du);
        add_second_order_term(dddeq, eq_xx.get(), eq_ux.get(), eq_uu.get(), dx, du);

        dddf = -dddf;
        dddeq = -dddeq;

        if (l != 0) {
          DDP_EXPECT(fabs(ddl) / fabs(dl) < sqrt(eps_x + eps_u));
          DDP_EXPECT(fabs(dddl) / fabs(ddl) < sqrt(eps_x + eps_u));
        }

        DDP_EXPECT_MSG(
            fmt::format(
                "t: {}\n"
                "df:\n"
                "{}\n"
                "ddf:\n"
                "{}",
                t,
                df.transpose(),
                ddf.transpose()),
            ddf.norm() / df.norm() < sqrt(eps_x + eps_u));

        DDP_EXPECT_MSG_ANY_OF(
            ("", fx.get().norm() != 0),
            (fmt::format(
                 "t: {}\n"
                 "ddf:\n"
                 "{}\n"
                 "dddf:\n"
                 "{}",
                 t,
                 ddf.transpose(),
                 dddf.transpose()),
             dddf.norm() / ddf.norm() < sqrt(eps_x + eps_u)));

        if (deq.size() != 0) {
          DDP_EXPECT_MSG(
              fmt::format(
                  "t: {}\n"
                  "deq:\n"
                  "{}\n"
                  "ddeq:\n"
                  "{}",
                  t,
                  deq.transpose(),
                  ddeq.transpose()),
              ddeq.norm() / deq.norm() < sqrt(eps_x + eps_u));
          DDP_EXPECT_MSG_ANY_OF(
              ("", eq_x.get().norm() != 0),
              (fmt::format(
                   "t: {}\n"
                   "ddeq:\n"
                   "{}\n"
                   "dddeq:\n"
                   "{}",
                   t,
                   ddeq.transpose(),
                   dddeq.transpose()),
               dddeq.norm() / ddeq.norm() < sqrt(eps_x + eps_u)));
        }
      }
#endif
    }
  }

  auto name() const noexcept -> fmt::string_view { return m_model.model_name(); }

  model_t const& m_model;
  index_t m_begin;
  index_t m_end;
  scalar_t dt;
  Constraint_Target_Range m_eq_target;
  scalar_t c = 1e2;
};

template <typename Dynamics, typename Eq>
struct problem_t0 {
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

  using eq_const = typename constraint_t::out_const;
  using eq_mut = typename constraint_t::out_mut;
  using deq_mut = typename constraint_t::dout_mut;

  using eq_x_mut = typename constraint_t::out_x_mut;
  using eq_u_mut = typename constraint_t::out_u_mut;

  using eq_xx_mut = typename constraint_t::out_xx_mut;
  using eq_ux_mut = typename constraint_t::out_ux_mut;
  using eq_uu_mut = typename constraint_t::out_uu_mut;

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

  void eval_f_to(x_mut x_out, index_t t, x_const x, u_const u) const { m_dynamics.eval_to(x_out, t, x, u); }
  void eval_eq_to(eq_mut eq_out, index_t t, x_const x, u_const u) const { m_constraint.eval_to(eq_out, t, x, u); }

  using derivative_storage_t =
      ddp::derivative_storage_t<scalar_t, control_indexer_t, eq_indexer_t, state_indexer_t, dstate_indexer_t>;
  using trajectory_t = trajectory::trajectory_t<scalar_t, state_indexer_t, control_indexer_t>;

  auto compute_derivatives(derivative_storage_t& derivs, trajectory_t const& traj) const {

    derivs.lfx.setZero();
    derivs.lfxx.setZero();

    // clang-format off
    for (auto zipped : ranges::zip(
          derivs.lx, derivs.lu, derivs.lxx, derivs.lux, derivs.luu,
          derivs.f_val, derivs.fx, derivs.fu, derivs.fxx, derivs.fux, derivs.fuu,
          derivs.eq_val, derivs.eq_x, derivs.eq_u, derivs.eq_xx, derivs.eq_ux, derivs.eq_uu,
          traj)) {
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
        ddp::chronometer_t timer("computing f derivatives");
        m_dynamics.second_order_deriv(fxx.get(), fux.get(), fuu.get(), fx.get(), fu.get(), f_v.get(), t, x, u);
      }
      {
        ddp::chronometer_t timer("computing eq derivatives");
        m_constraint
            .second_order_deriv(eq_xx.get(), eq_ux.get(), eq_uu.get(), eq_x.get(), eq_u.get(), eq_v.get(), t, x, u);
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

        scalar_t eps_x = 1e-30;
        scalar_t eps_u = 1e-30;

        auto dx = eigen::make_matrix<scalar_t>(m_dynamics.dstate_dim());
        dx.setRandom();
        dx *= eps_x;

        auto du = eigen::make_matrix<scalar_t>(m_dynamics.dcontrol_dim(t));
        du.setRandom();
        du *= eps_x;

        auto x_ = x.eval();
        auto u_ = u.eval();

        m_dynamics.integrate_x(eigen::as_mut_view(x_), eigen::as_const_view(x), eigen::as_const_view(dx));
        m_dynamics.integrate_u(eigen::as_mut_view(u_), eigen::as_const_view(u), eigen::as_const_view(du));

        l = this->l(t, x, u);
        eval_f_to(eigen::as_mut_view(f), t, x, u);
        eval_eq_to(eigen::as_mut_view(eq), t, x, u);

        l_ = this->l(t, eigen::as_const_view(x_), eigen::as_const_view(u_));
        eval_f_to(eigen::as_mut_view(f_), t, eigen::as_const_view(x_), eigen::as_const_view(u_));
        eval_eq_to(eigen::as_mut_view(eq_), t, eigen::as_const_view(x_), eigen::as_const_view(u_));

        scalar_t dl;

        auto df = eigen::make_matrix<scalar_t>(m_dynamics.dstate_dim());
        auto deq = eq_v.get().eval();

        scalar_t ddl;
        auto ddf = eigen::make_matrix<scalar_t>(m_dynamics.dstate_dim());
        vec_t ddeq{ne};

        scalar_t dddl;
        auto dddf = eigen::make_matrix<scalar_t>(m_dynamics.dstate_dim());
        vec_t dddeq{ne};

        dl = l_ - l;
        m_dynamics.difference_out(eigen::as_mut_view(df), eigen::as_const_view(f), eigen::as_const_view(f_));
        m_constraint.difference_out(eigen::as_mut_view(deq), eigen::as_const_view(eq), eigen::as_const_view(eq_));

        ddl = dl - (lx.get() * dx + lu.get() * du).value();
        ddf = df - (fx.get() * dx + fu.get() * du);
        ddeq = deq - (eq_x.get() * dx + eq_u.get() * du);

        dddl = ddl - (0.5 * dx.transpose() * lxx.get() * dx   //
                      + 0.5 * du.transpose() * luu.get() * du //
                      + du.transpose() * lux.get() * dx)
                         .value();
        dddf = -ddf;
        dddeq = -ddeq;

        add_second_order_term(dddf, fxx.get(), fux.get(), fuu.get(), dx, du);
        add_second_order_term(dddeq, eq_xx.get(), eq_ux.get(), eq_uu.get(), dx, du);

        dddf = -dddf;
        dddeq = -dddeq;

        if (l != 0) {
          DDP_EXPECT(fabs(ddl) / fabs(dl) < sqrt(eps_x + eps_u));
          DDP_EXPECT(fabs(dddl) / fabs(ddl) < sqrt(eps_x + eps_u));
        }

        DDP_EXPECT_MSG(
            fmt::format(
                "t: {}\n"
                "df:\n"
                "{}\n"
                "ddf:\n"
                "{}",
                t,
                df.transpose(),
                ddf.transpose()),
            ddf.norm() / df.norm() < sqrt(eps_x + eps_u));

        DDP_EXPECT_MSG_ANY_OF(
            ("", fx.get().norm() != 0),
            (fmt::format(
                 "t: {}\n"
                 "ddf:\n"
                 "{}\n"
                 "dddf:\n"
                 "{}",
                 t,
                 ddf.transpose(),
                 dddf.transpose()),
             dddf.norm() / ddf.norm() < sqrt(eps_x + eps_u)));

        if (deq.size() != 0) {
          DDP_EXPECT_MSG(
              fmt::format(
                  "t: {}\n"
                  "deq:\n"
                  "{}\n"
                  "ddeq:\n"
                  "{}",
                  t,
                  deq.transpose(),
                  ddeq.transpose()),
              ddeq.norm() / deq.norm() < sqrt(eps_x + eps_u));
          DDP_EXPECT_MSG_ANY_OF(
              ("", eq_x.get().norm() != 0),
              (fmt::format(
                   "t: {}\n"
                   "ddeq:\n"
                   "{}\n"
                   "dddeq:\n"
                   "{}",
                   t,
                   ddeq.transpose(),
                   dddeq.transpose()),
               dddeq.norm() / ddeq.norm() < sqrt(eps_x + eps_u)));
        }
      }
#endif
    }
  }

  auto name() const noexcept -> fmt::string_view { return m_dynamics.name(); }

  index_t m_begin;
  index_t m_end;
  scalar_t c = 1e2;
  dynamics_t m_dynamics;
  constraint_t m_constraint;
};

} // namespace ddp

#endif /* end of include guard PROBLEM_HPP_SQUCCNMX */
