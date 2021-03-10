#ifndef DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS
#define DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS

#include "ddp/internal/tensor.hpp"
#include "ddp/internal/second_order_finite_diff.hpp"
#include "ddp/pinocchio_model.hpp"
#include "ddp/space.hpp"

namespace ddp {

template <typename T>
struct pinocchio_dynamics_free {
  struct layout {
    pinocchio::model<T> const& model;
    T dt;
    bool can_use_first_order_diff;
  } self;

  using scalar = T;
  using key = typename pinocchio::model<T>::key;

  auto acquire_workspace() const -> key {
    return self.model.acquire_workspace();
  }

  void neutral_configuration(view<T, colvec> q) const {
    self.model.neutral_configuration(q);
  }
  void random_configuration(view<T, colvec> q) const {
    self.model.random_configuration(q);
  }

  auto state_space() const -> pinocchio_state_space<T> {
    return {{self.model}};
  }
  auto output_space() const -> pinocchio_state_space<T> {
    return {{self.model}};
  }
  auto control_space() const -> vector_space<T> {
    return {{self.model.tangent_dim()}};
  }

  auto eval_to_req() const -> mem_req {
    (void)this;
    return {veg::tag<T>, 0};
  }
  auto eval_to(
      view<T, colvec> f_out,
      i64 t,
      view<T const, colvec> x,
      view<T const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    (void)stack;

    VEG_DEBUG_ASSERT_ALL_OF(
        (!eigen::aliases(f_out, x)), //
        (!eigen::aliases(f_out, u)),
        (k));

    VEG_DEBUG_ASSERT_ALL_OF(
        (f_out.rows() == state_space().dim(t)),
        (x.rows() == state_space().dim(t)),
        (u.rows() == control_space().dim(t)));

    (void)t;
    auto nq = self.model.config_dim();

    VEG_BIND(auto, (q_out, v_out), eigen::split_at_row(f_out, nq));
    VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));

    // v_out = dt * v_in
    eigen::mul_scalar_to(v_out, self.dt, v);

    // q_out = q_in + v_out
    //       = q_in + dt * v_in
    self.model.integrate(q_out, q, eigen::as_const(v_out));

    // v_out = acc
    k = self.model.dynamics_aba(v_out, q, v, u, veg::none, VEG_FWD(k));

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    eigen::mul_scalar_to(v_out, self.dt, v_out);
    eigen::add_to(v_out, v_out, v);
    return k;
  }

  auto d_eval_to_req() const -> mem_req {
    (void)this;
    return {veg::tag<T>, 0};
  }
  auto d_eval_to(
      view<T, colmat> fx,
      view<T, colmat> fu,
      view<T, colvec> f,
      i64 t,
      view<T const, colvec> x,
      view<T const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    (void)t, (void)stack;

    VEG_DEBUG_ASSERT_ALL_OF(
        (!eigen::aliases(fx, fu, f, x, u)), //
        (!eigen::aliases(fu, f, x, u)),
        (!eigen::aliases(f, x, u)),
        (k));

    VEG_DEBUG_ASSERT_ALL_OF(
        (fx.rows() == state_space().ddim(t)),
        (fx.cols() == state_space().ddim(t)),
        (fu.rows() == state_space().ddim(t)),
        (fu.cols() == control_space().ddim(t)),
        (f.rows() == state_space().dim(t)),
        (x.rows() == state_space().dim(t)),
        (u.rows() == control_space().dim(t)));

    auto nq = self.model.config_dim();
    auto nv = self.model.tangent_dim();

    VEG_BIND(auto, (q_out, v_out), eigen::split_at_row(f, nq));
    VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));
    VEG_BIND(auto, (fx_qq, fx_qv, fx_vq, fx_vv), eigen::split_at(fx, nv, nv));
    VEG_BIND(auto, (fu_q, fu_v), eigen::split_at_row(fu, nv));

    // v_out = dt * v_in;
    eigen::mul_scalar_to(v_out, self.dt, v);

    // q_out = q_in + dt * v_in
    self.model.integrate(q_out, q, eigen::as_const(v_out));
    self.model.d_integrate_dq(fx_qq, q, eigen::as_const(v_out));
    self.model.d_integrate_dv(fx_qv, q, eigen::as_const(v_out));
    eigen::mul_scalar_to(fx_qv, self.dt, fx_qv);

    // v_out = acc
    fu_q.setZero();
    k = self.model.dynamics_aba(v_out, q, v, u, veg::none, VEG_FWD(k));
    k = self.model.d_dynamics_aba(
        fx_vq, fx_vv, fu_v, q, v, u, veg::none, VEG_FWD(k));

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    eigen::mul_scalar_to(v_out, self.dt, v_out);
    eigen::add_to(v_out, v_out, v);

    eigen::mul_scalar_to(fx_vq, self.dt, fx_vq);
    eigen::mul_scalar_to(fx_vv, self.dt, fx_vv);
    eigen::add_identity(fx_vv);

    eigen::mul_scalar_to(fu_v, self.dt, fu_v);

    return k;
  }
};

namespace make {
namespace fn {
struct pinocchio_dynamics_free_fn {
  VEG_TEMPLATE(
      (typename T),
      requires true,
      auto
      operator(),
      (model, pinocchio::model<T> const&),
      (dt, meta::identity_t<T>),
      (can_use_first_order_diff = false, bool))
  const->pinocchio_dynamics_free<T> {
    VEG_ASSERT_ELSE("unimplemented", !can_use_first_order_diff);
    return {{model, dt, can_use_first_order_diff}};
  }
};
} // namespace fn
__VEG_ODR_VAR(pinocchio_dynamics_free, fn::pinocchio_dynamics_free_fn);
} // namespace make

} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS */
