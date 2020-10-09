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

#if 0
  using model_t = pinocchio::model_t<scalar_t>;
  auto model = model_t{"~/pinocchio/models/others/robots/ur_description/urdf/ur5_gripper.urdf"};
  auto nq = model.configuration_dim_c();
  auto nv = model.tangent_dim_c();
  constexpr static index_t horizon = 10;

  struct constraint_t {
    vec_t m_target;
    auto eq_idx() const DDP_DECLTYPE_AUTO(indexing::vec_regular_indexer(0, horizon, dyn_index{m_target.size()}));
    auto operator[](index_t) const -> vec_t const& { return m_target; }
  };
  auto eq_gen = constraint_t{[&] {
    auto q = eigen::make_matrix<scalar_t>(nq, fix_index<1>{});
    model.neutral_configuration(eigen::as_mut_view(q));
    return q;
  }()};
#else
  using model_t = pendulum_model_t<scalar_t>;
  auto model = model_t{1.0, 1.0};
  auto nq = model.configuration_dim_c();
  auto nv = model.tangent_dim_c();
  constexpr static index_t horizon = 200;

  struct constraint_t {
    vec_t m_target;
    auto eq_idx() const -> indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>> {
      auto unfiltered = indexing::vec_regular_indexer(0, horizon, dyn_index{m_target.size()});
      return {unfiltered, horizon - 2, horizon - 1};
    }
    auto operator[](index_t t) const -> vec_t const& {
      static const vec_t empty{};
      if (t != horizon) {
        return empty;
      }
      return m_target;
    }
  };
  auto eq_gen = constraint_t{vec_t{1}};
  eq_gen.m_target[0] = 3.14;

#endif
  using problem_t = ddp::problem_t<model_t, constraint_t>;

  auto x_init = [&] {
    auto x = eigen::make_matrix<scalar_t>(nq + nv, fix_index<1>{});

    DDP_BIND(auto, (q, v), eigen::split_at_row_mut(x, nq));
    model.neutral_configuration(q);
    v.setZero();
    return x;
  }();

  problem_t prob{
      model,
      0,
      horizon,
      0.01,
      {static_cast<decltype(eq_gen)&&>(eq_gen)},
      1.0,
  };
  auto u_idx = indexing::vec_regular_indexer(0, horizon, nv);
  auto eq_idx = prob.m_eq_target.eq_idx();

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
    scalar_t mu = mu_init;
    scalar_t w = 1 / mu_init;
    scalar_t n = 1 / pow(mu_init, static_cast<scalar_t>(0.1L));
    scalar_t reg = 0;
    auto traj = solver.make_trajectory(control_generator_t{u_idx});
    auto new_traj = traj.clone();
    auto mults = solver.zero_multipliers<M>();

    for (auto zipped : ranges::zip(mults.eq, traj)) {
      DDP_BIND(auto&&, (eq, xu), zipped);
      eq.jac().setRandom();
      eq.origin() = xu.x();
    }

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }

    prob.compute_derivatives(derivs, traj);
    auto bres = solver.backward_pass<M>(traj, mults, reg, mu, derivs);

    mu = bres.mu;
    // reg = bres.reg;
    for (auto fb : bres.feedback) {
      fmt::print("val: {}\n", fb.val().transpose());
      fmt::print("jac:\n{}\n\n", fb.val().transpose());
    }

    auto step = solver.forward_pass<M>(new_traj, traj, mults, bres, true);

    traj = new_traj.clone();

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }

    for (index_t t = 0; t < 200; ++t) {
      auto mult_update_rv = solver.update_derivatives<M>(derivs, bres.feedback, mults, traj, mu, w, n);
      switch (mult_update_rv) {
      case mult_update_attempt_result_e::no_update:
        break;
      case mult_update_attempt_result_e::update_failure:
        mu *= 10;
        break;
      case mult_update_attempt_result_e::update_success:
        auto opt_obj = solver.optimality_obj(traj, mults, mu, derivs);
        n = opt_obj / pow(mu, static_cast<scalar_t>(0.1L));
        w /= pow(mu, static_cast<scalar_t>(1));
      }

      bres = solver.backward_pass<M>(traj, mults, reg, mu, derivs);
      mu = bres.mu;
      reg = bres.reg;
      fmt::print("mu: {:20}   reg: {:20}   w: {:20}   n: {:20}\n", mu, reg, w, n);

      step = solver.forward_pass<M>(new_traj, traj, mults, bres, true);
      if (step >= 0.5) {
        reg /= 2;
        if (reg < 1e-5) {
          reg = 0;
        }
      }

      traj = new_traj.clone();
      fmt::print("step: {}\n", step);
      fmt::print("eq: ");
      for (auto eq : derivs.eq()) {
        if (eq.val.size() > 0) {
          fmt::print("{} ", eq.val.norm());
        }
      }
      fmt::print("\n");
    }

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }
    fmt::print("x: {}\n", traj.x_f().transpose());
  }
}
