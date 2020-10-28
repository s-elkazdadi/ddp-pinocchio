#ifndef CONSTRAINT_HPP_K6WIV1TB
#define CONSTRAINT_HPP_K6WIV1TB

#include "ddp/dynamics.hpp"

namespace ddp {

template <typename Constraint>
struct constraint_advance_time_t {
  using scalar_t = typename Constraint::scalar_t;
  using dynamics_t = typename Constraint::dynamics_t;
  using constr_indexer_t = indexing::shift_time_idx_t<typename Constraint::constr_indexer_t>;

  using dims = typename Constraint::dims;
  using key = typename Constraint::key;

  using _1 = fix_index<1>;

  auto eq_idx() const -> constr_indexer_t { return {m_constraint.eq_idx(), 1}; }
  auto eq_dim(index_t t) const -> typename constr_indexer_t::row_kind { return m_constraint.eq_dim(t + 1); }

  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const { m_constraint.integrate_x(out, x, dx); }
  void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
    m_constraint.integrate_u(out, t, u, du);
  }
  template <typename Out>
  void difference_out(Out out, out_const<dims> start, out_const<dims> finish) const {
    m_constraint.difference_out(out, start, finish);
  }

  auto eval_to(out_mut<dims> out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
    auto x_n = eigen::make_matrix<scalar_t>(dynamics().state_dim(), _1{}); // x_{t+1}
    k = dynamics().eval_to(eigen::as_mut_view(x_n), t, x, u, DDP_MOVE(k));
    k = m_constraint.eval_to(out, t + 1, eigen::as_const_view(x_n), u, DDP_MOVE(k));
    return k;
  }

  auto first_order_deriv(    //
      out_x_mut<dims> out_x, //
      out_u_mut<dims> out_u, //
      out_mut<dims> out,     //
      index_t t,             //
      x_const<dims> x,       //
      u_const<dims> u,       //
      key k) const -> key {

    if (out.rows() == 0) {
      return k;
    }

    auto nx = dynamics().state_dim();
    auto nx_ = dynamics().dstate_dim();

    auto _x_n = eigen::make_matrix<scalar_t>(nx, _1{}); // x_{t+1}
    auto x_n = eigen::as_mut_view(_x_n);

    auto _fx_n = eigen::make_matrix<scalar_t>(nx_, nx_);
    auto _fu_n = eigen::make_matrix<scalar_t>(nx_, dynamics().dcontrol_dim(t));
    auto fx_n = eigen::as_mut_view(_fx_n);
    auto fu_n = eigen::as_mut_view(_fu_n);

    k = dynamics().first_order_deriv(fx_n, fu_n, x_n, t, x, u, DDP_MOVE(k));

    auto _eq_n_x = eigen::make_matrix<scalar_t>(eigen::rows_c(out), nx_);
    auto _eq_n_u = eigen::make_matrix<scalar_t>(eigen::rows_c(out), dynamics().dcontrol_dim(t + 1));
    auto eq_n_x = eigen::as_mut_view(_eq_n_x);
    auto eq_n_u = eigen::as_mut_view(_eq_n_u);

    k = m_constraint.first_order_deriv(
        eigen::into_view(eq_n_x),
        eigen::into_view(eq_n_u),
        out,
        t + 1,
        eigen::as_const_view(x_n),
        u,
        DDP_MOVE(k));
    DDP_ASSERT_MSG("constraint depends on control", eq_n_u.isConstant(0));

    out_x.noalias() = eq_n_x * fx_n;
    out_u.noalias() = eq_n_x * fu_n;

    return k;
  }

  auto second_order_deriv(     //
      out_xx_mut<dims> out_xx, //
      out_ux_mut<dims> out_ux, //
      out_uu_mut<dims> out_uu, //
      out_x_mut<dims> out_x,   //
      out_u_mut<dims> out_u,   //
      out_mut<dims> out,       //
      index_t t,               //
      x_const<dims> x,         //
      u_const<dims> u,         //
      key k                    //
  ) const -> key {
    return finite_diff_hessian_compute<constraint_advance_time_t>{*this, dynamics().second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  Constraint m_constraint;

  auto dynamics() const DDP_DECLTYPE_AUTO(m_constraint.dynamics());
};

template <typename Constraint1, typename Constraint2>
struct concat_constraint_t {
  static_assert(std::is_same<typename Constraint1::scalar_t, typename Constraint2::scalar_t>::value, "");
  static_assert(std::is_same<typename Constraint1::dynamics_t, typename Constraint2::dynamics_t>::value, "");
  using scalar_t = typename Constraint1::scalar_t;
  using dynamics_t = typename Constraint1::dynamics_t;
  using key = typename Constraint1::key;
  using constr_indexer_t = indexing::row_concat_indexer_t< //
      typename Constraint1::constr_indexer_t,              //
      typename Constraint2::constr_indexer_t               //
      >;

  using dims = dimensions_from_idx_t<          //
      scalar_t,                                //
      typename dynamics_t::state_indexer_t,    //
      typename dynamics_t::dstate_indexer_t,   //
      typename dynamics_t::control_indexer_t,  //
      typename dynamics_t::dcontrol_indexer_t, //
      constr_indexer_t,                        //
      constr_indexer_t                         //
      >;

  auto eq_idx() const -> constr_indexer_t { return indexing::row_concat(m_constr1.eq_idx(), m_constr2.eq_idx()); }
  auto eq_dim(index_t t) const -> typename constr_indexer_t::row_kind {
    return m_constr1.eq_dim(t) + m_constr2.eq_dim(t);
  }
  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const { m_constr1.integrate_x(out, x, dx); }
  void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
    m_constr1.integrate_u(out, t, u, du);
  }
  template <typename Out>
  void difference_out(Out out, out_const<dims> start, out_const<dims> finish) const {
    out = finish - start;
  }
  auto eval_to(out_mut<dims> out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
    auto n1 = m_constr1.eq_dim(t);

    DDP_BIND(auto, (out1, out2), eigen::split_at_row_mut(out, n1));
    k = m_constr1.eval_to(eigen::into_view(out1), t, x, u, DDP_MOVE(k));
    k = m_constr2.eval_to(eigen::into_view(out2), t, x, u, DDP_MOVE(k));

    return k;
  }

  auto first_order_deriv(    //
      out_x_mut<dims> out_x, //
      out_u_mut<dims> out_u, //
      out_mut<dims> out,     //
      index_t t,             //
      x_const<dims> x,       //
      u_const<dims> u,       //
      key k) const -> key {
    auto n1 = m_constr1.eq_dim(t);

    DDP_BIND(auto, (out1, out2), eigen::split_at_row_mut(out, n1));
    DDP_BIND(auto, (out_x1, out_x2), eigen::split_at_row_mut(out_x, n1));
    DDP_BIND(auto, (out_u1, out_u2), eigen::split_at_row_mut(out_u, n1));

    k = m_constr1.first_order_deriv(
        eigen::into_view(out_x1),
        eigen::into_view(out_u1),
        eigen::into_view(out1),
        t,
        x,
        u,
        DDP_MOVE(k));
    k = m_constr2.first_order_deriv(
        eigen::into_view(out_x2),
        eigen::into_view(out_u2),
        eigen::into_view(out2),
        t,
        x,
        u,
        DDP_MOVE(k));
    return k;
  }

  auto second_order_deriv(     //
      out_xx_mut<dims> out_xx, //
      out_ux_mut<dims> out_ux, //
      out_uu_mut<dims> out_uu, //
      out_x_mut<dims> out_x,   //
      out_u_mut<dims> out_u,   //
      out_mut<dims> out,       //
      index_t t,               //
      x_const<dims> x,         //
      u_const<dims> u,         //
      key k                    //
  ) const -> key {
    return finite_diff_hessian_compute<concat_constraint_t>{*this, dynamics().second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  Constraint1 m_constr1;
  Constraint2 m_constr2;

  auto dynamics() const DDP_DECLTYPE_AUTO(m_constr1.dynamics());
};

namespace detail {

template <typename... Constraints>
struct concat_constraint_impl;

template <typename First, typename... Constraints>
struct concat_constraint_impl<First, Constraints...> {
  using type = concat_constraint_t<First, typename concat_constraint_impl<Constraints...>::type>;
  static auto run(First c, Constraints... cs) -> type {
    return {DDP_MOVE(c), DDP_MOVE(concat_constraint_impl<Constraints...>::run(cs...))};
  }
};

template <typename Constraint>
struct concat_constraint_impl<Constraint> {
  using type = Constraint;
  static auto run(Constraint c) -> type { return c; }
};

} // namespace detail

template <typename... Constraints>
auto concat_constraint(Constraints... cs) -> typename detail::concat_constraint_impl<Constraints...>::type {
  return detail::concat_constraint_impl<Constraints...>::run(DDP_MOVE(cs)...);
}

namespace detail {

template <index_t N>
struct constraint_advance_time_impl {
  template <typename Constraint>
  using type = constraint_advance_time_t<typename constraint_advance_time_impl<N - 1>::template type<Constraint>>;

  template <typename Constraint>
  static auto run(Constraint c) -> type<Constraint> {
    return {constraint_advance_time_impl<N - 1>::template run(DDP_MOVE(c))};
  }
};

template <>
struct constraint_advance_time_impl<0> {
  template <typename Constraint>
  using type = Constraint;

  template <typename Constraint>
  static auto run(Constraint c) -> type<Constraint> {
    return c;
  }
};

} // namespace detail

template <index_t N = 1, typename Constraint>
auto constraint_advance_time(Constraint c) ->
    typename detail::constraint_advance_time_impl<N>::template type<Constraint> {
  return detail::constraint_advance_time_impl<N>::template run<Constraint>(DDP_MOVE(c));
}

template <typename Model, typename Constraint_Target_View, index_t N_Frames>
struct spatial_constraint_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using constr_indexer_t = decltype(std::declval<Constraint_Target_View const&>().eq_idx());

  using dims = dimensions_from_idx_t<
      scalar_t,
      typename dynamics_t::state_indexer_t,
      typename dynamics_t::dstate_indexer_t,
      typename dynamics_t::control_indexer_t,
      typename dynamics_t::dcontrol_indexer_t,
      constr_indexer_t,
      constr_indexer_t>;

  using key = typename dynamics_t::key;

  auto eq_idx() const -> constr_indexer_t { return m_constraint_target_view.eq_idx(); }
  auto eq_dim(index_t t) const -> typename constr_indexer_t::row_kind { return eq_idx().rows(t); }

  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const { m_dynamics.integrate_x(out, x, dx); }
  void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
    m_dynamics.integrate_u(out, t, u, du);
  }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
        ("", out.rows() == start.rows()), //
        ("", out.rows() == finish.rows()));
    out = finish - start;
  }

  auto eval_to(out_mut<dims> out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
    auto nq = m_dynamics.m_model.configuration_dim_c();
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    (void)u;
    (void)v;
    bool computed = false;

    for (index_t i = 0; i < N_Frames; ++i) {

      auto const& _target = m_constraint_target_view(i, t);
      auto target = eigen::as_const_view(_target);
      if (target.rows() == 0) {
        continue;
      }

      DDP_BIND(auto, (out_head, out_tail), eigen::split_at_row_mut(out, dyn_index{3 * i}));
      DDP_BIND(auto, (out_3, out_rest), eigen::split_at_row_mut(out_tail, fix_index<3>{}));
      (void)out_head;
      (void)out_rest;

      if (not computed) {
        k = m_dynamics.m_model.frame_coordinates_precompute(q, DDP_MOVE(k));
        computed = true;
      }

      k = m_dynamics.m_model.frame_coordinates(out_3, m_frame_id[i], DDP_MOVE(k));
      out_3 -= target;
    }
    return k;
  }

  auto first_order_deriv(    //
      out_x_mut<dims> out_x, //
      out_u_mut<dims> out_u, //
      out_mut<dims> out,     //
      index_t t,             //
      x_const<dims> x,       //
      u_const<dims> u,       //
      key k                  //
  ) const -> key {
    auto nq = m_dynamics.m_model.configuration_dim_c();
    auto nv = m_dynamics.m_model.tangent_dim_c();
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    (void)u;
    (void)v;
    out_u.setZero();

    bool computed = false;
    for (index_t i = 0; i < N_Frames; ++i) {
      auto _target = m_constraint_target_view(i, t);
      auto target = eigen::as_const_view(_target);
      if (target.rows() == 0) {
        continue;
      }

      DDP_BIND(auto, (out_head, out_tail), eigen::split_at_row_mut(out, dyn_index{3 * i}));
      DDP_BIND(auto, (out_head_x, out_tail_x), eigen::split_at_row_mut(out_x, dyn_index{3 * i}));

      DDP_BIND(auto, (out_3x, out_rest_x), eigen::split_at_row_mut(out_tail_x, fix_index<3>{}));
      DDP_BIND(auto, (out_3, out_rest), eigen::split_at_row_mut(out_tail, fix_index<3>{}));
      (void)out_head;
      (void)out_rest;
      (void)out_head_x;
      (void)out_rest_x;

      DDP_BIND(auto, (out_3q, out_3v), eigen::split_at_col_mut(out_3x, nv));

      (void)v;

      if (not computed) {
        k = m_dynamics.m_model.dframe_coordinates_precompute(q, DDP_MOVE(k));
        computed = true;
      }

      k = m_dynamics.m_model.frame_coordinates(out_3, m_frame_id[i], DDP_MOVE(k));
      out_3 -= target;

      k = m_dynamics.m_model.d_frame_coordinates(out_3q, m_frame_id[i], DDP_MOVE(k));
      out_3v.setZero();
    }
    return k;
  }

  auto second_order_deriv(     //
      out_xx_mut<dims> out_xx, //
      out_ux_mut<dims> out_ux, //
      out_uu_mut<dims> out_uu, //
      out_x_mut<dims> out_x,   //
      out_u_mut<dims> out_u,   //
      out_mut<dims> out,       //
      index_t t,               //
      x_const<dims> x,         //
      u_const<dims> u,         //
      key k                    //
  ) const -> key {
    return finite_diff_hessian_compute<spatial_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
  index_t m_frame_id[N_Frames];
};

template <typename Dynamics, typename Constraint_Target_View, index_t N_Frames>
auto spatial_constraint(Dynamics d, Constraint_Target_View v, index_t const (&ids)[N_Frames])
    -> spatial_constraint_t<typename Dynamics::model_t, Constraint_Target_View, N_Frames> {
  auto out =
      spatial_constraint_t<typename Dynamics::model_t, Constraint_Target_View, N_Frames>{DDP_MOVE(d), DDP_MOVE(v), {}};
  for (index_t i = 0; i < N_Frames; ++i) {
    out.m_frame_id[i] = ids[i];
  }
  return out;
}

template <typename Model, typename Constraint_Target_View>
struct config_constraint_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using constr_indexer_t = decltype(std::declval<Constraint_Target_View const&>().eq_idx());

  using dims = dimensions_from_idx_t<
      scalar_t,
      typename dynamics_t::state_indexer_t,
      typename dynamics_t::dstate_indexer_t,
      typename dynamics_t::control_indexer_t,
      typename dynamics_t::dcontrol_indexer_t,
      constr_indexer_t,
      constr_indexer_t>;

  using key = typename dynamics_t::key;

  auto eq_idx() const -> constr_indexer_t { return m_constraint_target_view.eq_idx(); }
  auto eq_dim(index_t t) const -> typename constr_indexer_t::row_kind { return eq_idx().rows(t); }

  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const { m_dynamics.integrate_x(out, x, dx); }
  void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
    m_dynamics.integrate_u(out, t, u, du);
  }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
        ("", out.rows() == start.rows()), //
        ("", out.rows() == finish.rows()));
    out = finish - start;
  }

  auto eval_to(out_mut<dims> out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
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

  auto first_order_deriv(    //
      out_x_mut<dims> out_x, //
      out_u_mut<dims> out_u, //
      out_mut<dims> out,     //
      index_t t,             //
      x_const<dims> x,       //
      u_const<dims> u,       //
      key k                  //
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

  auto second_order_deriv(     //
      out_xx_mut<dims> out_xx, //
      out_ux_mut<dims> out_ux, //
      out_uu_mut<dims> out_uu, //
      out_x_mut<dims> out_x,   //
      out_u_mut<dims> out_u,   //
      out_mut<dims> out,       //
      index_t t,               //
      x_const<dims> x,         //
      u_const<dims> u,         //
      key k                    //
  ) const -> key {
    return finite_diff_hessian_compute<config_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
};

template <typename Dynamics, typename Constraint_Target_View>
auto config_constraint(Dynamics d, Constraint_Target_View v)
    -> config_constraint_t<typename Dynamics::model_t, Constraint_Target_View> {
  return {DDP_MOVE(d), DDP_MOVE(v)};
}

template <typename Model, typename Constraint_Target_View>
struct velocity_constraint_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using constr_indexer_t = decltype(std::declval<Constraint_Target_View const&>().eq_idx());

  using dims = dimensions_from_idx_t<
      scalar_t,
      typename dynamics_t::state_indexer_t,
      typename dynamics_t::dstate_indexer_t,
      typename dynamics_t::control_indexer_t,
      typename dynamics_t::dcontrol_indexer_t,
      constr_indexer_t,
      constr_indexer_t>;

  using key = typename dynamics_t::key;

  auto eq_idx() const -> constr_indexer_t { return m_constraint_target_view.eq_idx(); }
  auto eq_dim(index_t t) const -> typename constr_indexer_t::row_kind { return eq_idx().rows(t); }

  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const { m_dynamics.integrate_x(out, x, dx); }
  void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
    m_dynamics.integrate_u(out, t, u, du);
  }
  template <typename Out, typename In>
  void difference_out(Out out, In start, In finish) const {
    DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
        ("", out.rows() == start.rows()), //
        ("", out.rows() == finish.rows()));
    out = finish - start;
  }

  auto eval_to(out_mut<dims> out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
    auto target = eigen::as_const_view(m_constraint_target_view[t]);
    if (target.rows() == 0) {
      return k;
    }

    (void)u;

    auto nq = m_dynamics.m_model.configuration_dim_c();
    auto nv = m_dynamics.m_model.tangent_dim_c();

    DDP_BIND(auto, (xq, xv), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (target_v, target_0), eigen::split_at_row(target, nv));
    DDP_ASSERT(target_0.rows() == 0);
    (void)xq;

    out = xv - target_v;
    return k;
  }

  auto first_order_deriv(    //
      out_x_mut<dims> out_x, //
      out_u_mut<dims> out_u, //
      out_mut<dims> out,     //
      index_t t,             //
      x_const<dims> x,       //
      u_const<dims> u,       //
      key k                  //
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

    DDP_BIND(auto, (xq, xv), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (target_v, target_0), eigen::split_at_row(target, nv));
    DDP_ASSERT(target_0.rows() == 0);
    (void)xq;

    DDP_BIND(auto, (out_xq, out_xv), eigen::split_at_col_mut(out_x, nv));

    out = xv - target_v;
    out_xv.setIdentity();

    out_xq.setZero();
    out_u.setZero();
    return k;
  }

  auto second_order_deriv(     //
      out_xx_mut<dims> out_xx, //
      out_ux_mut<dims> out_ux, //
      out_uu_mut<dims> out_uu, //
      out_x_mut<dims> out_x,   //
      out_u_mut<dims> out_u,   //
      out_mut<dims> out,       //
      index_t t,               //
      x_const<dims> x,         //
      u_const<dims> u,         //
      key k                    //
  ) const -> key {
    return finite_diff_hessian_compute<velocity_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
};

template <typename Dynamics, typename Constraint_Target_View>
auto velocity_constraint(Dynamics d, Constraint_Target_View v)
    -> velocity_constraint_t<typename Dynamics::model_t, Constraint_Target_View> {
  return {DDP_MOVE(d), DDP_MOVE(v)};
}

} // namespace ddp

#endif /* end of include guard CONSTRAINT_HPP_K6WIV1TB */
