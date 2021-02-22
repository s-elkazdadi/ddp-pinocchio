#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, "  ", "\n", "", "")
#include "ddp/detail/utils.hpp"

#include "ddp/pinocchio_model.hpp"
#include "ddp/ddp.hpp"
#include "ddp/ddp_bwd.ipp"
#include "ddp/ddp_fwd.ipp"
#include "ddp/problem.hpp"
#include "ddp/pendulum_model.hpp"

#include <fmt/ostream.h>
#include <boost/multiprecision/mpfr.hpp>

#if 0
using scalar_t = boost::multiprecision::number<
    boost::multiprecision::backends::mpfr_float_backend<500, boost::multiprecision::allocate_stack>,
    boost::multiprecision::et_off>;
#else
using scalar_t = double;
#endif

namespace ddp {
struct two_arms_dynamics {
  using scalar_t = ::scalar_t;
  using model_t = pinocchio::model_t<scalar_t>;
  using dynamics_t = ddp::dynamics_t<model_t>;

  using x_t = dyn_index;
  using dx_t = dyn_index;
  using u_t = dyn_index;
  using du_t = dyn_index;

  using dims = dimensions_t<scalar_t, x_t, x_t, dx_t, dx_t, u_t, u_t, du_t, du_t, x_t, x_t, dx_t, dx_t>;

  using state_indexer_t = indexing::regular_indexer_t<x_t>;
  using dstate_indexer_t = indexing::regular_indexer_t<dx_t>;
  using control_indexer_t = indexing::regular_indexer_t<u_t>;
  using dcontrol_indexer_t = indexing::regular_indexer_t<du_t>;
  using key = typename model_t::key;

  static constexpr fix_index<2> _2{};

  auto state_dim() const -> x_t {
    return _2 * m_one_arm.state_dim(); // arm-1, arm-2
  }
  auto dstate_dim() const -> dx_t {
    return _2 * m_one_arm.dstate_dim(); // arm-1, arm-2
  }
  auto control_dim(index_t t) const -> u_t { return m_one_arm.control_dim(t) * _2; }
  auto dcontrol_dim(index_t t) const -> u_t { return m_one_arm.dcontrol_dim(t) * _2; }

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return indexing::vec_regular_indexer(begin, end, state_dim());
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return indexing::vec_regular_indexer(begin, end, dstate_dim());
  }

  auto acquire_workspace() const -> key { return m_one_arm.acquire_workspace(); }

  void neutral_configuration(x_mut<dims> out) const {
    DDP_ASSERT_MSG("out vector does not have the correct size", out.size() == state_dim().value());

    auto nx = m_one_arm.state_dim();
    DDP_BIND(auto, (out_0, out_1), eigen::split_at_row_mut(out, nx));

    m_one_arm.neutral_configuration(out_0);
    m_one_arm.neutral_configuration(out_1);
    out_1[0] += 3.14;
  }

  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    (void)this;
    out = x + dx;
  }

  void d_integrate_x(out_x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    ((void)x, (void)dx, (void)this);
    out.setIdentity();
  }
  void d_integrate_x_dx(out_x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    ((void)x, (void)dx, (void)this);
    out.setIdentity();
  }
  void integrate_u(u_mut<dims> out, index_t /*unused*/, u_const<dims> u, u_const<dims> du) const {
    (void)this;
    out = u + du;
  }
  void difference_out(dout_mut<dims> out, out_const<dims> start, out_const<dims> finish) const {
    (void)this;
    out = finish - start;
  }
  void d_difference_out_dfinish(out_x_mut<dims> out, x_const<dims> start, x_const<dims> finish) const {
    ((void)start, (void)finish, (void)this);
    out.setIdentity();
  }

  auto eval_to(out_mut<dims> x_out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
    (void)this;
    auto nx = m_one_arm.state_dim();

    DDP_BIND(auto, (out0, out1), eigen::split_at_row_mut(x_out, nx));
    DDP_BIND(auto, (x0, x1), eigen::split_at_row(x, nx));
    DDP_BIND(auto, (u0, u1), eigen::split_at_row(u, m_one_arm.control_dim(t)));

    k = m_one_arm.eval_to(out0, t, x0, u0, DDP_MOVE(k));
    k = m_one_arm.eval_to(out1, t, x1, u1, DDP_MOVE(k));

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

    auto nx = m_one_arm.state_dim();
    auto nu = m_one_arm.control_dim(t);
    DDP_BIND(auto, (f0, f1), eigen::split_at_row_mut(f, nx));
    DDP_BIND(auto, (fx00, fx01, fx10, fx11), eigen::split_at_mut(fx, nx, nx));
    DDP_BIND(auto, (fu00, fu01, fu10, fu11), eigen::split_at_mut(fu, nx, nu));
    DDP_BIND(auto, (x0, x1), eigen::split_at_row(x, nx));
    DDP_BIND(auto, (u0, u1), eigen::split_at_row(u, m_one_arm.control_dim(t)));

    k = m_one_arm.first_order_deriv(fx00, fu00, f0, t, x0, u0, DDP_MOVE(k));
    k = m_one_arm.first_order_deriv(fx11, fu11, f1, t, x1, u1, DDP_MOVE(k));

    fx01.setZero();
    fx10.setZero();
    fu01.setZero();
    fu10.setZero();

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
    return finite_diff_hessian_compute<two_arms_dynamics>{*this, m_one_arm.second_order_finite_diff()}
        .second_order_deriv(fxx, fux, fuu, fx, fu, f, t, x, u, DDP_MOVE(k));
  }

  auto second_order_finite_diff() const -> bool { return m_one_arm.second_order_finite_diff(); }
  auto name() const -> fmt::string_view {
    (void)this;
    return "two_arms";
  }

  dynamics_t m_one_arm;
};

struct two_arm_spatial_constraint {
  using scalar_t = ::scalar_t;
  using model_t = two_arms_dynamics::model_t;
  using dynamics_t = two_arms_dynamics;

  using constr_indexer_t = indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index, fix_index<1>>>;

  using dims = dimensions_from_idx_t<
      scalar_t,
      typename dynamics_t::state_indexer_t,
      typename dynamics_t::dstate_indexer_t,
      typename dynamics_t::control_indexer_t,
      typename dynamics_t::dcontrol_indexer_t,
      constr_indexer_t,
      constr_indexer_t>;

  using key = typename dynamics_t::key;

  auto eq_idx() const -> constr_indexer_t {
    (void)this;
    return {indexing::vec_regular_indexer(2, m_horizon + 2, dyn_index(3)), m_begin, m_end};
  }
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
    auto const& model = m_dynamics.m_one_arm.m_model;

    auto nx = m_dynamics.m_one_arm.state_dim();
    auto nq = model.configuration_dim_c();
    DDP_BIND(auto, (x0, x1), eigen::split_at_row(x, nx));
    DDP_BIND(auto, (q0, v0), eigen::split_at_row(x0, nq));
    DDP_BIND(auto, (q1, v1), eigen::split_at_row(x1, nq));
    (void)v0, (void)v1;

    if (eq_dim(t).value() == 0) {
      return k;
    }

    auto tmp0 = eigen::make_matrix<scalar_t>(fix_index<3>());
    auto tmp1 = eigen::make_matrix<scalar_t>(fix_index<3>());

    k = model.frame_coordinates_precompute(q0, DDP_MOVE(k));
    k = model.frame_coordinates(eigen::as_mut_view(tmp0), 18, DDP_MOVE(k));

    k = model.frame_coordinates_precompute(q1, DDP_MOVE(k));
    k = model.frame_coordinates(eigen::as_mut_view(tmp1), 18, DDP_MOVE(k));

    out = tmp1 - tmp0;
    out[0] += 0.5;
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

    auto const& model = m_dynamics.m_one_arm.m_model;

    auto nx = m_dynamics.m_one_arm.state_dim();
    auto nq = model.configuration_dim_c();
    DDP_BIND(auto, (x0, x1), eigen::split_at_row(x, nx));
    DDP_BIND(auto, (q0, v0), eigen::split_at_row(x0, nq));
    DDP_BIND(auto, (q1, v1), eigen::split_at_row(x1, nq));
    (void)v0, (void)v1;

    if (eq_dim(t).value() == 0) {
      return k;
    }
    out_u.setZero();

    DDP_BIND(auto, (out_x3, out_x_none), eigen::split_at_row_mut(out_x, fix_index<3>()));
    DDP_BIND(auto, (out_x0, out_x1), eigen::split_at_col_mut(out_x3, nx));

    DDP_BIND(auto, (out_q0, out_v0), eigen::split_at_col_mut(out_x0, nq));
    DDP_BIND(auto, (out_q1, out_v1), eigen::split_at_col_mut(out_x1, nq));

    auto tmp0 = eigen::make_matrix<scalar_t>(fix_index<3>());
    auto tmp1 = eigen::make_matrix<scalar_t>(fix_index<3>());

    k = model.dframe_coordinates_precompute(q0, DDP_MOVE(k));
    k = model.d_frame_coordinates(out_q0, 18, DDP_MOVE(k));
    k = model.frame_coordinates(eigen::as_mut_view(tmp0), 18, DDP_MOVE(k));

    k = model.dframe_coordinates_precompute(q1, DDP_MOVE(k));
    k = model.d_frame_coordinates(out_q1, 18, DDP_MOVE(k));
    k = model.frame_coordinates(eigen::as_mut_view(tmp1), 18, DDP_MOVE(k));

    out_x0 *= -1;

    out = tmp1 - tmp0;
    out[0] += 0.5;
    out_v0.setZero();
    out_v1.setZero();

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
    return finite_diff_hessian_compute<two_arm_spatial_constraint>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }

  dynamics_t m_dynamics;
  index_t m_horizon;
  index_t m_begin;
  index_t m_end;
};

struct two_arm_velocity_constraint_t {
  using scalar_t = typename ::scalar_t;
  using dynamics_t = two_arms_dynamics;

  using constr_indexer_t = indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>>;

  using dims = dimensions_from_idx_t<
      scalar_t,
      typename dynamics_t::state_indexer_t,
      typename dynamics_t::dstate_indexer_t,
      typename dynamics_t::control_indexer_t,
      typename dynamics_t::dcontrol_indexer_t,
      constr_indexer_t,
      constr_indexer_t>;

  using key = typename dynamics_t::key;

  auto eq_idx() const -> constr_indexer_t {
    return {
        indexing::vec_regular_indexer(1, m_horizon + 1, dynamics().m_one_arm.m_model.tangent_dim_c() * fix_index<2>{}),
        m_begin,
        m_end};
  }
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
    if (eq_dim(t).value() == 0) {
      return k;
    }

    (void)u;

    auto nq = m_dynamics.m_one_arm.m_model.configuration_dim_c();
    auto nv = m_dynamics.m_one_arm.m_model.tangent_dim_c();

    DDP_BIND(auto, (x0, x1), eigen::split_at_row(x, nq + nv));
    DDP_BIND(auto, (q0, v0), eigen::split_at_row(x0, nq));
    DDP_BIND(auto, (q1, v1), eigen::split_at_row(x1, nq));
    (void)q0, (void)q1;

    DDP_BIND(auto, (out0, out1), eigen::split_at_row_mut(out, nv));
    out0 = v0;
    out1 = v1;
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
    if (eq_dim(t).value() == 0) {
      return k;
    }

    (void)u;

    auto const& m_model = m_dynamics.m_one_arm.m_model;

    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();

    DDP_BIND(auto, (x0, x1), eigen::split_at_row(x, nq + nv));
    DDP_BIND(auto, (q0, v0), eigen::split_at_row(x0, nq));
    DDP_BIND(auto, (q1, v1), eigen::split_at_row(x1, nq));
    (void)q0, (void)q1;

    DDP_BIND(auto, (out0, out1), eigen::split_at_row_mut(out, nv));

    out0 = v0;
    out1 = v1;

    DDP_BIND(auto, (out_00, out_01, out_10, out_11), eigen::split_at_mut(out_x, nv, nv + nv));
    DDP_BIND(auto, (out_00q, out_00v), eigen::split_at_col_mut(out_00, nv));
    DDP_BIND(auto, (out_11q, out_11v), eigen::split_at_col_mut(out_11, nv));

    out_00v.setIdentity();
    out_11v.setIdentity();

    out_01.setZero();
    out_10.setZero();
    out_00q.setZero();
    out_11q.setZero();
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
    return finite_diff_hessian_compute<two_arm_velocity_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }

  dynamics_t m_dynamics;
  index_t m_horizon;
  index_t m_begin;
  index_t m_end;
};

} // namespace ddp

auto main() -> int {
  using ddp::index_t;
  ddp::pinocchio::model_t<scalar_t> model(
      "/opt/openrobots/share/example-robot-data/robots/ur_description/urdf/ur5_gripper.urdf",
      omp_get_num_procs(),
      false);

  for (index_t i = 0; i < model.n_frames(); ++i) {
    fmt::print("{}\n", model.frame_name(i));
  }

  index_t horizon = 200;

  auto _dy = ddp::pinocchio_dynamics(model, 0.01, false);
  auto dy = ddp::two_arms_dynamics{_dy};

  auto _constr = ddp::two_arm_spatial_constraint{dy, horizon, (horizon) / 2, horizon + 2};
  auto _constrv = ddp::two_arm_velocity_constraint_t{dy, horizon, (horizon) / 2 + 2, horizon + 1};
  auto constr =
      ddp::concat_constraint(ddp::constraint_advance_time(_constrv), ddp::constraint_advance_time<2>(_constr));

  auto prob = ddp::problem(0, horizon, scalar_t(1e2), dy, constr);
  using problem_t = decltype(prob);

  auto u_idx = ddp::indexing::vec_regular_indexer(0, horizon, dy.control_dim(0));
  auto eq_idx = constr.eq_idx();

  auto x_init = ddp::eigen::make_matrix<scalar_t>(dy.state_dim());
  dy.neutral_configuration(ddp::eigen::as_mut_view(x_init));

  struct control_generator_t {
    using u_mat_t = ddp::eigen::matrix_from_idx_t<scalar_t, problem_t::control_indexer_t>;
    using x_mat_t = ddp::eigen::matrix_from_idx_t<scalar_t, problem_t::state_indexer_t>;
    problem_t::control_indexer_t const& m_u_idx;
    index_t m_current_index = 0;
    u_mat_t m_value = u_mat_t::Zero(m_u_idx.rows(m_current_index).value()).eval();

    auto operator()() const -> ddp::eigen::view_t<u_mat_t const> { return ddp::eigen::as_const_view(m_value); }
    void next(ddp::eigen::view_t<x_mat_t const>) {
      ++m_current_index;
      m_value.resize(m_u_idx.rows(m_current_index).value());
      m_value.setZero();
    }
  };

  ddp::ddp_solver_t<problem_t> solver{prob, u_idx, eq_idx, x_init};

  {
    using std::pow;

    constexpr auto M = ddp::method::primal_dual_constant_multipliers;

    auto derivs = solver.uninit_derivative_storage();

    scalar_t const mu_init = 1e8;
    auto res = solver.solve<M>({10000, 1e-80, mu_init}, solver.make_trajectory(control_generator_t{u_idx}));
    DDP_BIND(auto&&, (traj, fb), res);
    (void)fb;

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }
    fmt::print("x: {}\n", traj.x_f().transpose());
  }
}
