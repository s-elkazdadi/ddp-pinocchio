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

#if 1
using scalar_t = boost::multiprecision::number<
    boost::multiprecision::backends::mpfr_float_backend<1000, boost::multiprecision::allocate_stack>,
    boost::multiprecision::et_off>;
#else
using scalar_t = double;
#endif

using namespace ddp;

auto main() -> int {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  using vec_t = Eigen::Matrix<scalar_t, -1, 1>;

  using model_t = pinocchio::model_t<scalar_t>;
  auto model = model_t{
      fmt::string_view{"~/pinocchio/models/others/robots/ur_description/urdf/ur5_gripper.urdf"},
      omp_get_num_procs()};
  auto nq = model.configuration_dim_c();
  auto nv = model.tangent_dim_c();
  constexpr static index_t horizon = 10;

  struct constraint_t {
    vec_t m_target;
    auto eq_idx() const DDP_DECLTYPE_AUTO(indexing::vec_regular_indexer(2, horizon + 2, dyn_index{m_target.size()}));
    auto operator[](index_t) const -> vec_t const& { return m_target; }
  };
  using dynamics_t = ddp::dynamics_t<model_t>;
  using problem_t = ddp::problem_t<
      dynamics_t,
      constraint_advance_time_t<constraint_advance_time_t<config_constraint_t<model_t, constraint_t>>>>;

  auto eq_gen = constraint_t{[&] {
    auto q = eigen::make_matrix<scalar_t>(nq, fix_index<1>{});
    model.neutral_configuration(eigen::as_mut_view(q));
    return q;
  }()};

  auto x_init = [&] {
    auto x = eigen::make_matrix<scalar_t>(nq + nv, fix_index<1>{});

    DDP_BIND(auto, (q, v), eigen::split_at_row_mut(x, nq));
    model.neutral_configuration(q);
    v.setZero();
    return x;
  }();

  dynamics_t dy{model, 0.01, false};
  problem_t prob{
      0,
      horizon,
      1.0,
      dy,
      constraint_advance_time(
          constraint_advance_time(config_constraint_t<model_t, constraint_t>{dy, DDP_MOVE(eq_gen)})),
  };
  auto u_idx = indexing::vec_regular_indexer(0, horizon, nv);
  auto eq_idx = prob.m_constraint.eq_idx();

  struct control_generator_t {
    using u_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::control_indexer_t>;
    using x_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::state_indexer_t>;
    problem_t::control_indexer_t const& m_u_idx;
    index_t m_current_index = 0;
    u_mat_t m_value = u_mat_t::Zero(m_u_idx.rows(m_current_index).value()).eval();

    auto operator()() const -> eigen::view_t<u_mat_t const> { return eigen::as_const_view(m_value); }
    void next(eigen::view_t<x_mat_t const>) {
      ++m_current_index;
      m_value.resize(m_u_idx.rows(m_current_index).value());
    }
  };

  ddp_solver_t<problem_t> solver{prob, u_idx, eq_idx, x_init};

  {
    using std::pow;

    constexpr auto M = method::primal_dual_affine_multipliers;

    auto derivs = solver.uninit_derivative_storage();

    scalar_t const mu_init = 1e20;
    scalar_t w = 1 / mu_init;
    scalar_t n = 1 / pow(mu_init, static_cast<scalar_t>(0.1L));
    scalar_t reg = 0;
    auto res = solver.solve<M>({200, 1e-80, mu_init, 0.0, w, n}, solver.make_trajectory(control_generator_t{u_idx}));
    DDP_BIND(auto&&, (traj, fb), res);
    (void)fb;

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }
    fmt::print("x: {}\n", traj.x_f().transpose());
  }
}
