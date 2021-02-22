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

#if 0
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
      (void)this;
      auto unfiltered = indexing::vec_regular_indexer(2, horizon + 2, dyn_index{1});
      return {unfiltered, horizon, horizon + 1};
    }
    auto operator[](index_t t) const -> eigen::view_t<vec_t const> {
      if (t != horizon) {
        return {nullptr, 0, 1, 0};
      }
      return eigen::as_const_view(m_target);
    }
  };
  using dynamics_t = ddp::dynamics_t<model_t>;

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

  struct vel_constr_t {
    auto eq_idx() const {
      return indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>>{
          indexing::vec_regular_indexer(1, horizon + 1, dyn_index(2)),
          horizon,
          horizon + 1};
    }
    auto operator[](index_t t) const -> vec_t {
      if (t != horizon) {
        return eigen::make_matrix<scalar_t>(dyn_index(0));
      }
      return eigen::make_matrix<scalar_t>(dyn_index(2));
    }
  };
  auto eq_gen_v = vel_constr_t{};

  auto constr = constraint_advance_time<2>(config_single_coord_constraint(dy, DDP_MOVE(eq_gen), 1));
  auto _constrv = constraint_advance_time(velocity_constraint(dy, eq_gen_v));
  auto prob = problem(0, horizon, 1.0, dy, concat_constraint(constr, _constrv));

  using problem_t = decltype(prob);

  auto u_idx = indexing::vec_regular_indexer(0, horizon, nv);
  auto eq_idx = prob.m_constraint.eq_idx();

  struct control_generator_t {
    using u_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::control_indexer_t>;
    using x_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::state_indexer_t>;
    problem_t::control_indexer_t const& m_u_idx;
    index_t m_current_index = 0;
    u_mat_t m_value = u_mat_t::Zero(m_u_idx.rows(m_current_index).value()).eval();

    auto operator()() const -> eigen::view_t<u_mat_t const> { return eigen::as_const_view(m_value); }
    void next(eigen::view_t<x_mat_t const> /*unused*/) {
      ++m_current_index;
      m_value.resize(m_u_idx.rows(m_current_index).value());
    }
  };

  ddp_solver_t<problem_t> solver{prob, u_idx, eq_idx, x_init};

  auto test_noise = [&](log_file_t out, auto const& traj, auto const& fb) {
    auto new_traj = traj.clone();
    auto rnd = eigen::make_matrix<scalar_t>(eigen::rows_c(x_init));

    fmt::print(out.ptr, "{}", "{");

    for (index_t i = 3; i < 30; ++i) {
      fmt::print("{}\n", i);
      double eps = std::pow(10, -i);
      fmt::print(out.ptr, "{}: [", eps);

      for (index_t k = 0; k < 10; ++k) {
        fmt::print("{}\n", k);
        for (auto zipped : ranges::zip(traj, new_traj, fb)) {
          rnd.setRandom();
          rnd.normalize();

          DDP_BIND(auto, (xu, new_xu, K), zipped);
          chronometer_t c{"u update"};
          new_xu.u() = xu.u() + K.jac() * (new_xu.x() - xu.x());
          prob.eval_f_to(new_xu.x_next(), new_xu.current_index(), new_xu.as_const().x(), new_xu.as_const().u());
          new_xu.x_next() += eps * rnd;
        }
        for (auto xu : new_traj) {
          index_t t = xu.current_index();
          auto eq = eigen::make_matrix<scalar_t>(prob.constraint().eq_dim(t));
          prob.eval_eq_to(eigen::as_mut_view(eq), t, xu.as_const().x(), xu.as_const().u());
          if (eq.size() > 0) {
            fmt::print(out.ptr, "{},", eq.norm());
          }
        }
      }

      fmt::print(out.ptr, "],");
    }

    fmt::print(out.ptr, "{}\n", "}");
  };

  auto derivs = solver.uninit_derivative_storage();
  {
    constexpr auto M = method::primal_dual_constant_multipliers;

    scalar_t const mu_init = 1e4;
    auto res = solver.solve<M>({200, 1e-200, mu_init}, solver.make_trajectory(control_generator_t{u_idx}));
    DDP_BIND(auto&&, (traj, fb), res);
    (void)fb;
  }
}
