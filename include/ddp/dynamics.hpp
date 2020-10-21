#ifndef DYNAMICS_HPP_6PSM7PYM
#define DYNAMICS_HPP_6PSM7PYM

#include "ddp/detail/second_order_finite_diff.hpp"

namespace ddp {

template <typename Model>
struct dynamics_t {
  using scalar_t = typename Model::scalar_t;
  using model_t = Model;

  using x_t = decltype(
      static_cast<model_t const*>(nullptr)->configuration_dim_c() +
      static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using dx_t = decltype(
      static_cast<model_t const*>(nullptr)->tangent_dim_c() + static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using u_t = decltype(static_cast<model_t const*>(nullptr)->tangent_dim_c());
  using du_t = decltype(static_cast<model_t const*>(nullptr)->tangent_dim_c());

  using dims = dimensions_t<scalar_t, x_t, x_t, dx_t, dx_t, u_t, u_t, du_t, du_t, x_t, x_t, dx_t, dx_t>;

  using state_indexer_t = indexing::regular_indexer_t<x_t>;
  using dstate_indexer_t = indexing::regular_indexer_t<dx_t>;
  using control_indexer_t = indexing::regular_indexer_t<u_t>;
  using dcontrol_indexer_t = indexing::regular_indexer_t<du_t>;

  using key = typename model_t::key;

  auto state_dim() const -> x_t { return m_model.configuration_dim_c() + m_model.tangent_dim_c(); }
  auto dstate_dim() const noexcept -> dx_t { return m_model.tangent_dim_c() + m_model.tangent_dim_c(); }
  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.configuration_dim_c() + m_model.tangent_dim_c());
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return indexing::vec_regular_indexer(begin, end, m_model.tangent_dim_c() + m_model.tangent_dim_c());
  }
  auto control_dim(index_t /*unused*/) const noexcept -> u_t { return m_model.tangent_dim_c(); }
  auto dcontrol_dim(index_t /*unused*/) const noexcept -> du_t { return m_model.tangent_dim_c(); }

  auto acquire_workspace() const noexcept -> key { return m_model.acquire_workspace(); }

  void neutral_configuration(x_mut<dims> out) const {
    DDP_ASSERT_MSG("out vector does not have the correct size", out.size() == state_dim().value());
    auto nq = m_model.configuration_dim_c();
    DDP_BIND(auto, (out_q, out_v), eigen::split_at_row_mut(out, nq));
    m_model.neutral_configuration(out_q);
    out_v.setZero();
  }

  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (q_out, v_out), eigen::split_at_row_mut(out, nq));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (dq, dv), eigen::split_at_row(dx, nv));

    m_model.integrate(q_out, q, dq);
    v_out = v + dv;
  }

  void d_integrate_x(out_x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (out_qq, out_qv, out_vq, out_vv), eigen::split_at_mut(out, nv, nv));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (dq, dv), eigen::split_at_row(dx, nv));
    (void)v;
    (void)dv;

    m_model.d_integrate_dq(out_qq, q, dq);
    out_qv.setZero();
    out_vq.setZero();
    out_vv.setIdentity();
  }
  void d_integrate_x_dx(out_x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (out_qq, out_qv, out_vq, out_vv), eigen::split_at_mut(out, nv, nv));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    DDP_BIND(auto, (dq, dv), eigen::split_at_row(dx, nv));
    (void)v;
    (void)dv;

    m_model.d_integrate_dv(out_qq, q, dq);
    out_qv.setZero();
    out_vq.setZero();
    out_vv.setIdentity();
  }

  void integrate_u(u_mut<dims> out, index_t /*unused*/, u_const<dims> u, u_const<dims> du) const { out = u + du; }
  void difference_out(dout_mut<dims> out, out_const<dims> start, out_const<dims> finish) const {
    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();
    DDP_BIND(auto, (out_q, out_v), eigen::split_at_row_mut(out, nv));
    DDP_BIND(auto, (start_q, start_v), eigen::split_at_row(start, nq));
    DDP_BIND(auto, (finish_q, finish_v), eigen::split_at_row(finish, nq));

    m_model.difference(out_q, start_q, finish_q);
    out_v = finish_v - start_v;
  }
  void d_difference_out_dfinish(out_x_mut<dims> out, x_const<dims> start, x_const<dims> finish) const {
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

  auto eval_to(out_mut<dims> x_out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
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
      out_x_mut<dims> fx, //
      out_u_mut<dims> fu, //
      out_mut<dims> f,    //
      index_t t,          //
      x_const<dims> x,    //
      u_const<dims> u,    //
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

  auto second_order_deriv(  //
      out_xx_mut<dims> fxx, //
      out_ux_mut<dims> fux, //
      out_uu_mut<dims> fuu, //
      out_x_mut<dims> fx,   //
      out_u_mut<dims> fu,   //
      out_mut<dims> f,      //
      index_t t,            //
      x_const<dims> x,      //
      u_const<dims> u,      //
      key k                 //
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

template <typename Model>
auto pinocchio_dynamics(Model const& model, typename Model::scalar_t dt, bool use_second_order_finite_difference = true)
    -> dynamics_t<Model> {
  return {model, dt, use_second_order_finite_difference};
}

} // namespace ddp

#endif /* end of include guard DYNAMICS_HPP_6PSM7PYM */
