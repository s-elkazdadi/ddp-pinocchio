#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, "  ", "\n", "", "")
#include "ddp/detail/utils.hpp"

#include "ddp/pinocchio_model.hpp"
#include "ddp/ddp.hpp"
#include "ddp/ddp_bwd.ipp"
#include "ddp/ddp_fwd.ipp"
#include "ddp/problem.hpp"
#include "ddp/cart_model.hpp"

#include <fmt/ostream.h>
#include <boost/multiprecision/mpfr.hpp>

#if 1
using scalar_t = boost::multiprecision::number<
    boost::multiprecision::backends::mpfr_float_backend<500, boost::multiprecision::allocate_stack>,
    boost::multiprecision::et_off>;
#else
using scalar_t = double;
#endif

using namespace ddp;

auto main() -> int {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  using vec_t = Eigen::Matrix<scalar_t, -1, 1>;

  using model_t = cart_pendulum_model_t<scalar_t>;
  auto model = model_t{1.0, 1.0, 1.0};
  auto nq = model.configuration_dim_c();
  auto nv = model.tangent_dim_c();
  constexpr static index_t horizon = 200;

  struct constraint_t {
    vec_t m_target;
    auto eq_idx() const -> indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>> {
      auto unfiltered = indexing::vec_regular_indexer(2, horizon + 2, dyn_index{m_target.size()});
      return {unfiltered, horizon, horizon + 1};
    }
    auto operator[](index_t t) const -> eigen::view_t<vec_t const> {
      static const vec_t empty{};
      if (t != horizon) {
        return {nullptr, 0, 1, 0};
      }
      return eigen::as_const_view(m_target);
    }
  };
  using dynamics_t = ddp::dynamics_t<model_t>;
  using problem_t = ddp::problem_t<
      dynamics_t,
      constraint_advance_time_t<constraint_advance_time_t<config_constraint_t<model_t, constraint_t>>>>;

  auto eq_gen = constraint_t{vec_t{2}};
  eq_gen.m_target[0] = 0;
  eq_gen.m_target[1] = 3.14;

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
      constraint_advance_time<2>(config_constraint_t<model_t, constraint_t>{dy, DDP_MOVE(eq_gen)}),
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

    constexpr auto M = method::primal_dual_constant_multipliers;

    auto derivs = solver.uninit_derivative_storage();

    scalar_t const mu_init = 1e2;
    auto res = solver.solve<M>({200, 1e-200, mu_init}, solver.make_trajectory(control_generator_t{u_idx}));
    DDP_BIND(auto&&, (traj, fb), res);
    (void)fb;

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }
    fmt::print("x: {}\n", traj.x_f().transpose());
  }
}

