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
    boost::multiprecision::backends::mpfr_float_backend<500, boost::multiprecision::allocate_stack>,
    boost::multiprecision::et_off>;
#else
using scalar_t = double;
#endif

using namespace ddp;

auto main() -> int {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  using vec_t = Eigen::Matrix<scalar_t, -1, 1>;

  using model_t = pendulum_model_t<scalar_t>;
  auto model = model_t{1.0, 1.0};
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

  auto eq_gen = constraint_t{vec_t{1}};
  eq_gen.m_target[0] = 3.14;

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

    auto new_traj = traj.clone();

    auto rnd = eigen::make_matrix<scalar_t>(eigen::rows_c(x_init));

    while (true) {
      long double eps{};
      std::scanf("%Lf", &eps);
      fmt::print("eps: {}\n", eps);
      auto const& traj_c = traj;

      for (auto zipped : ranges::zip(traj_c, new_traj, fb)) {
        rnd.setRandom();
        DDP_BIND(auto, (xu, new_xu, K), zipped);
        new_xu.u() = xu.u() + K.jac() * (new_xu.x() - xu.x());
        prob.eval_f_to(new_xu.x_next(), new_xu.current_index(), new_xu.as_const().x(), new_xu.as_const().u());
        new_xu.x_next() += eps * rnd;
      }

      for (auto xu : new_traj) {
        index_t t = xu.current_index();
        auto eq = eigen::make_matrix<scalar_t>(prob.constraint().eq_dim(t));
        prob.eval_eq_to(eigen::as_mut_view(eq), t, xu.as_const().x(), xu.as_const().u());
        if (eq.size() > 0) {
          fmt::print("{}\n", eq.norm());
        }
      }
    }
  }
}
