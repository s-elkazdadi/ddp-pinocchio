#ifndef CART_MODEL_HPP_XBXDUEUC
#define CART_MODEL_HPP_XBXDUEUC

#include "ddp/indexer.hpp"
#include "ddp/dynamics.hpp"
#include <cmath>
#include <random>

namespace ddp {

template <typename T>
struct cart_pendulum_model_t {
  using scalar_t = T;
  static constexpr bool fixed_size = true;
  using state_dim_kind = fix_index<2>;
  using dstate_dim_kind = fix_index<2>;

private:
  using mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 2, 1>>;
  using mmut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 2, 2>>;
  using const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 2, 1> const>;

  scalar_t M;
  scalar_t m;
  scalar_t l;
  static constexpr long double g = 9.81L;

public:
  struct key {
    explicit operator bool() { return true; }
  };

  cart_pendulum_model_t(scalar_t mass_cart, scalar_t mass_pendulum, scalar_t length)
      : M{mass_cart}, m{mass_pendulum}, l{length} {
    DDP_ASSERT_MSG_ALL_OF( //
        ("mass should be positive", M > 0),
        ("mass should be positive", m > 0), //
        ("length should be positive", l > 0));
  }

  auto configuration_dim() const -> index_t { return 2; }
  auto tangent_dim() const -> index_t { return 2; }

  auto configuration_dim_c() const -> fix_index<2> { return {}; }
  auto tangent_dim_c() const -> fix_index<2> { return {}; }

  auto acquire_workspace() const -> key { return {}; };

  void neutral_configuration(mut_view_t out_q) const {
    out_q[0] = 0;
    out_q[1] = 0;
  }
  void random_configuration(mut_view_t out_q) const {
    constexpr long double pi = 3.141592653589793238462643383279L;
    static thread_local auto mt = std::mt19937{std::random_device{}()};

    out_q[0] = std::uniform_real_distribution<long double>{-pi, pi}(mt);
    out_q[1] = std::uniform_real_distribution<long double>{-1, 1}(mt);
  }

  void integrate(       //
      mut_view_t out_q, //
      const_view_t q,   //
      const_view_t v    //
  ) const {
    out_q = q + v;
  }

  void d_integrate_dq(      //
      mmut_view_t out_q_dq, //
      const_view_t q,       //
      const_view_t v        //
  ) const {
    (void)q;
    (void)v;
    out_q_dq.setIdentity();
  }

  void d_integrate_dv(      //
      mmut_view_t out_q_dv, //
      const_view_t q,       //
      const_view_t v        //
  ) const {
    (void)q;
    (void)v;
    out_q_dv.setIdentity();
  }

  void difference(          //
      mut_view_t out_v,     //
      const_view_t q_start, //
      const_view_t q_finish //
  ) const {
    out_v = q_finish - q_start;
  }

  void d_difference_dq_start(     //
      mmut_view_t out_v_dq_start, //
      const_view_t q_start,       //
      const_view_t q_finish       //
  ) const {
    (void)q_start, (void)q_finish;
    out_v_dq_start.setIdentity();
    out_v_dq_start *= -1;
  }

  void d_difference_dq_finish(     //
      mmut_view_t out_v_dq_finish, //
      const_view_t q_start,        //
      const_view_t q_finish        //
  ) const {
    (void)q_start, (void)q_finish;
    out_v_dq_finish.setIdentity();
  }

  auto dynamics_aba(               //
      mut_view_t out_acceleration, //
      const_view_t q,              //
      const_view_t v,              //
      const_view_t tau,            //
      key k                        //
  ) const -> key {
    (void)v;
    using std::cos;
    using std::sin;
    scalar_t c = cos(q[1]);
    scalar_t s = sin(q[1]);
    out_acceleration[0] = (m * (+g) * s * c + tau[0] + m * l * v[1] * v[1]) / (M + s * s * m);
    out_acceleration[1] = (-g) / l * s - c / l * out_acceleration[0];
    return k;
  }

  auto d_dynamics_aba(                   //
      mmut_view_t out_acceleration_dq,   //
      mmut_view_t out_acceleration_dv,   //
      mmut_view_t out_acceleration_dtau, //
      const_view_t q,                    //
      const_view_t v,                    //
      const_view_t tau,                  //
      key k                              //
  ) const -> key {
    scalar_t c = cos(q[1]);
    scalar_t s = sin(q[1]);

    scalar_t m0 = M + s * s * m;

    scalar_t out_ac = (m * (+g) * s * c + tau[0] + m * l * v[1] * v[1]) / (M + s * s * m);

    out_acceleration_dq(0, 0) = 0;
    out_acceleration_dq(1, 0) = 0;
    out_acceleration_dv(0, 0) = 0;
    out_acceleration_dv(1, 0) = 0;
    out_acceleration_dtau(0, 1) = 0;
    out_acceleration_dtau(1, 1) = 0;

    out_acceleration_dq(0, 1) =
        (m * +g * (c * c - s * s) * m0 - 2 * s * c * m * (m * +g * s * c + tau[0] + m * l * v[1] * v[1])) / (m0 * m0);
    out_acceleration_dq(1, 1) = -g / l * c - c / l * out_acceleration_dq(0, 1) + s / l * out_ac;

    out_acceleration_dv(0, 1) = 2 * m * l / m0 * v[1];
    out_acceleration_dv(1, 1) = -c / l * out_acceleration_dv(0, 1);

    out_acceleration_dtau(0, 0) = 1 / m0;
    out_acceleration_dtau(1, 0) = -c / l * out_acceleration_dtau(0, 0);

    using std::cos;
    return k;
  }

  auto model_name() const -> fmt::string_view { return "cartpole"; }
};

template <typename Model, typename Constraint_Target_View>
struct config_single_coord_constraint_t {
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

    out[0] = x[i] - target[i];
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
    if (target.rows() == 0) {
      return k;
    }

    out[0] = x[i] - target[i];
    out_x.setZero();
    out_x(0, i) = 1;
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
    return finite_diff_hessian_compute<config_single_coord_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }

  dynamics_t m_dynamics;
  Constraint_Target_View m_constraint_target_view;
  index_t i;
};

template <typename Dynamics, typename Constraint_Target_View>
auto config_single_coord_constraint(Dynamics d, Constraint_Target_View v, index_t i)
    -> config_single_coord_constraint_t<typename Dynamics::model_t, Constraint_Target_View> {
  return {DDP_MOVE(d), DDP_MOVE(v), i};
}

} // namespace ddp

#endif /* end of include guard CART_MODEL_HPP_XBXDUEUC */
