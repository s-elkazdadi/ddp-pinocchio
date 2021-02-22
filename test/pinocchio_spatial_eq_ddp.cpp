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

auto actual_neutral() -> Eigen::Matrix<scalar_t, -1, 1> {
  Eigen::Matrix<scalar_t, -1, 1> q(19);

  q[0] = 0;
  q[1] = 0;
  q[2] = 0.4792;
  q[3] = 0;
  q[4] = 0;
  q[5] = 0;
  q[6] = 1;
  q[7] = -0.1;
  q[8] = 0.7;
  q[9] = -1;
  q[10] = -0.1;
  q[11] = -0.7;
  q[12] = 1;
  q[13] = 0.1;
  q[14] = 0.7;
  q[15] = -1;
  q[16] = 0.1;
  q[17] = -0.7;
  q[18] = 1.;

  return q;
}

using namespace ddp;

using model_t = pinocchio::model_t<scalar_t>;
struct anymal_contact_dynamics {
  using scalar_t = ::scalar_t;
  using model_t = ::model_t;

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

  auto model() const -> model_t const& { return m_dynamics.m_model; }

  auto state_dim() const -> x_t { return m_dynamics.state_dim(); }
  auto dstate_dim() const -> dx_t { return m_dynamics.dstate_dim(); }
  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return m_dynamics.state_indexer(begin, end);
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return m_dynamics.dstate_indexer(begin, end);
  }
  auto control_dim(index_t t) const -> u_t { return m_dynamics.control_dim(t) - fix_index<6>(); }
  auto dcontrol_dim(index_t t) const -> du_t { return m_dynamics.dcontrol_dim(t) - fix_index<6>(); }

  auto acquire_workspace() const -> key { return m_dynamics.acquire_workspace(); }

  void neutral_configuration(x_mut<dims> out) const { m_dynamics.neutral_configuration(out); }
  void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const { m_dynamics.integrate_x(out, x, dx); }
  void d_integrate_x(out_x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    m_dynamics.d_integrate_x(out, x, dx);
  }
  void d_integrate_x_dx(out_x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
    m_dynamics.d_integrate_x_dx(out, x, dx);
  }
  void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
    m_dynamics.integrate_u(out, t, u, du);
  }
  void difference_out(dout_mut<dims> out, out_const<dims> start, out_const<dims> finish) const {
    m_dynamics.difference_out(out, start, finish);
  }
  void d_difference_out_dfinish(out_x_mut<dims> out, x_const<dims> start, x_const<dims> finish) const {
    m_dynamics.d_difference_out_dfinish(out, start, finish);
  }

  auto eval_to(out_mut<dims> x_out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {

    auto nq = model().configuration_dim_c();
    auto nv = model().tangent_dim_c();

    DDP_BIND(auto, (q_out, v_out), eigen::split_at_row_mut(x_out, nq));
    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));

    v_out = m_dynamics.dt * v;
    model().integrate(eigen::as_mut_view(q_out), q, eigen::as_const_view(v_out));

    thread_local auto u_full = eigen::make_matrix<scalar_t>(nv);

    if (u_full.size() != nv.value()) {
      u_full.resize(nv.value());
      u_full.setZero();
    }

    DDP_BIND(auto, (u_full_0, u_full_u), eigen::split_at_row_mut(u_full, fix_index<6>()));
    (void)u_full_0;
    u_full_u = u;

    k = model().contact_dynamics(
        eigen::as_mut_view(v_out),
        eigen::as_const_view(q),
        eigen::as_const_view(v),
        eigen::as_const_view(u_full),
        frame_ids,
        4,
        DDP_MOVE(k));

    v_out = v + v_out * m_dynamics.dt;
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
    auto nq = model().configuration_dim_c();
    auto nv = model().tangent_dim_c();

    thread_local auto u_full = eigen::make_matrix<scalar_t>(nv);
    thread_local auto fu_full = eigen::make_matrix<scalar_t>(nv, nv);

    if (u_full.size() != nv.value()) {
      u_full.resize(nv.value());
      fu_full.resize(nv.value(), nv.value());
      u_full.setZero();
      fu_full.setZero();
    }
    DDP_BIND(auto, (u_full_0, u_full_u), eigen::split_at_row_mut(u_full, fix_index<6>()));
    DDP_BIND(auto, (fu_full_0, fu_full_u), eigen::split_at_col_mut(fu_full, fix_index<6>()));
    (void)u_full_0;
    (void)fu_full_0;
    u_full_u = u;

    DDP_BIND(auto, (q, v), eigen::split_at_row(x, nq));

    auto tmp = detail::get<1>(eigen::split_at_row_mut(fx.col(0), nv));
    // v_out = dt * v_in;
    tmp = m_dynamics.dt * v;
    auto dt_v = eigen::as_const_view(tmp);

    DDP_BIND(auto, (fx_top_left, fx_top_right, fx_bot_left, fx_bot_right), eigen::split_at_mut(fx, nv, nv));

    // q_out = q_in + dt * v_in
    model().d_integrate_dq(fx_top_left, q, dt_v);
    model().d_integrate_dv(fx_top_right, q, dt_v);
    fx_top_right *= m_dynamics.dt;

    // v_out = acc
    DDP_BIND(auto, (fu_top, fu_bot), eigen::split_at_row_mut(fu, nv));

    fu_top.setZero();
    k = model().d_contact_dynamics(
        fx_bot_left,
        fx_bot_right,
        eigen::as_mut_view(fu_full),
        q,
        v,
        eigen::as_const_view(u_full),
        frame_ids,
        4,
        DDP_MOVE(k));
    fu_bot = fu_full_u;

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    fx_bot_left *= m_dynamics.dt;
    fx_bot_right *= m_dynamics.dt;
    fx_bot_right += decltype(fx_bot_right.eval())::Identity(nv.value(), nv.value());
    fu_bot *= m_dynamics.dt;
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
    return finite_diff_hessian_compute<anymal_contact_dynamics>{*this, second_order_finite_diff()}
        .second_order_deriv(fxx, fux, fuu, fx, fu, f, t, x, u, DDP_MOVE(k));
  }

  auto second_order_finite_diff() const -> bool { return m_dynamics.second_order_finite_diff(); }

  auto name() const -> fmt::string_view { return "anymal_contact"; }

  ddp::dynamics_t<model_t> m_dynamics;
  index_t const (&frame_ids)[4];
};

struct anymal_config_constraint_t {
  using scalar_t = ::scalar_t;
  using model_t = ::model_t;
  using dynamics_t = anymal_contact_dynamics;

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
    return {indexing::vec_regular_indexer(2, m_horizon, dyn_index(6)), m_begin, m_end};
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
    auto nq = m_dynamics.model().configuration_dim_c();
    auto nv = m_dynamics.model().tangent_dim_c();

    static const auto target = actual_neutral();
    auto tmp = eigen::make_matrix<scalar_t>(nv);

    m_dynamics.model().difference(
        eigen::as_mut_view(tmp),
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x, nq)) // configuration part of x_nn
    );

    out = tmp.topRows(6);
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

    if (eq_dim(t).value() == 0) {
      return k;
    }

    auto const& m_model = m_dynamics.model();

    auto nq = m_model.configuration_dim_c();
    auto nv = m_model.tangent_dim_c();

    static const auto target = actual_neutral();

    auto tmp = eigen::make_matrix<scalar_t>(nv);

    m_model.difference(
        eigen::as_mut_view(tmp),
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x, nq)) // configuration part of x
    );

    auto d_diff = eigen::make_matrix<scalar_t>(nv, nv);
    m_dynamics.model().d_difference_dq_finish(
        eigen::as_mut_view(d_diff),
        detail::get<0>(eigen::split_at_row(target, nq)),
        detail::get<0>(eigen::split_at_row(x, nq)) // configuration part of x
    );

    DDP_BIND(auto, (out_xq, out_xv), eigen::split_at_col_mut(out_x, nv));
    out_xq.noalias() = d_diff.topRows(6);
    out = tmp.topRows(6);
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
    return finite_diff_hessian_compute<anymal_config_constraint_t>{*this, m_dynamics.second_order_finite_diff()}
        .second_order_deriv(out_xx, out_ux, out_uu, out_x, out_u, out, t, x, u, DDP_MOVE(k));
  }

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }

  dynamics_t m_dynamics;
  index_t m_horizon;
  index_t m_begin;
  index_t m_end;
};

auto main() -> int {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  using model_t = pinocchio::model_t<scalar_t>;
  auto model = model_t{
      fmt::string_view{
          "/opt/openrobots/share/example-robot-data/robots/anymal_b_simple_description/robots/anymal.urdf"},
      omp_get_num_procs(),
      true};

  char const* feet_names[] = {
      "LF_FOOT",
      "LH_FOOT",
      "RF_FOOT",
      "RH_FOOT",
  };

  index_t frame_ids[4] = {};
  index_t j = 0;
  for (index_t i = 0; i < model.n_frames(); ++i) {
    for (auto name : feet_names) {
      if (model.frame_name(i) == name) {
        frame_ids[j] = i;
        ++j;
      }
    }
  }
  index_t horizon = 20;

  auto _dy = pinocchio_dynamics(model, 0.05, true);
  auto dy = anymal_contact_dynamics{_dy, frame_ids};

  auto _constr = anymal_config_constraint_t{dy, horizon + 2, 2, horizon};

  auto constr = constraint_advance_time<2>(_constr);

  auto prob = problem(0, horizon, scalar_t(1.0), dy, constr);

  auto u_idx = indexing::vec_regular_indexer(0, horizon, dyn_index{12});
  auto eq_idx = constr.eq_idx();
  using problem_t = decltype(prob);

  struct control_generator_t {
    using u_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::control_indexer_t>;
    using x_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::state_indexer_t>;
    problem_t::control_indexer_t const& m_u_idx;
    model_t const& model;
    index_t const (&frame_ids)[4];
    index_t m_current_index = 0;
    u_mat_t m_value = u_mat_t::Zero(m_u_idx.rows(m_current_index).value()).eval();

    auto operator()() const -> eigen::view_t<u_mat_t const> { return eigen::as_const_view(m_value); }
    void next(eigen::view_t<x_mat_t const> x) {
      ++m_current_index;
      DDP_BIND(auto, (q, v), eigen::split_at_row(x, model.configuration_dim_c()));
      m_value.resize(m_u_idx.rows(m_current_index).value());
      m_value = model.tau_sol(q, v, frame_ids, 4);
    }
  };

  auto x_init = eigen::make_matrix<scalar_t>(dy.state_dim());
  x_init.topRows(19) = actual_neutral();

  {
    DDP_BIND(auto, (q, v), eigen::split_at_row(x_init, model.configuration_dim_c()));
    fmt::print("{}\n", model.tau_sol(q, v, frame_ids, 4).transpose());
  }

  ddp_solver_t<problem_t> solver{prob, u_idx, eq_idx, x_init};

  {
    using std::pow;

    constexpr auto M = method::primal_dual_constant_multipliers;

    auto derivs = solver.uninit_derivative_storage();

    scalar_t const mu_init = 1e8;
    auto init_traj = solver.make_trajectory(control_generator_t{u_idx, model, frame_ids});
    auto res = solver.solve<M>({1000, 1e-80, mu_init}, DDP_MOVE(init_traj));
    DDP_BIND(auto&&, (traj, fb), res);
    (void)fb;

    {
      log_file_t f{"anymal_traj"};
      traj.println_to_file(f.ptr);
    }

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }
    fmt::print("x: {}\n", traj.x_f().transpose());
  }
}
