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

  using model_t = pinocchio::model_t<scalar_t>;
  auto model = model_t{fmt::string_view{"./test/anymal.urdf"}, omp_get_num_procs()};

  index_t foot_frames[4] = {};
  index_t k = 0;
  for (index_t i = 0; i < model.n_frames(); ++i) {
    if ((model.frame_name(i) == "LF_FOOT" or //
         model.frame_name(i) == "LH_FOOT" or //
         model.frame_name(i) == "RF_FOOT" or //
         model.frame_name(i) == "RH_FOOT")) {
      DDP_ASSERT(k < 4);
      foot_frames[k] = i;
      ++k;
    }
  }
  DDP_ASSERT(k == 4);

  auto nq = model.configuration_dim_c();
  auto nv = model.tangent_dim_c();
  constexpr static index_t horizon = 10;

  for (index_t i = 0; i < model.n_frames(); ++i) {
    fmt::print(
        "{} {}\n",
        model.frame_name(i),
        (std::find(foot_frames, foot_frames + 4, i) != foot_frames + 4) ? "<-" : "");
  }

  struct position_target_range_t {
    using vec3 = decltype(eigen::make_matrix<scalar_t>(fix_index<3>{}));
    vec3 m_target;

    auto eq_idx() const -> indexing::range_row_filter_t<indexing::regular_indexer_t<fix_index<3>>> {
      (void)this;
      auto unfiltered = indexing::vec_regular_indexer(2, horizon + 2, fix_index<3>{});
      return {unfiltered, 2, horizon};
    }
    auto operator[](index_t t) const
        -> eigen::matrix_view_t<scalar_t const, dyn_index, fix_index<1>, fix_index<3>, fix_index<1>> {
      if (eq_idx().rows(t).value() == 0) {
        return {nullptr, 0, 1, 0};
      }
      return eigen::into_view(eigen::as_const_view(m_target));
    }
  };

  struct config_target_range_t {
    using vec_t = Eigen::Matrix<scalar_t, -1, 1>;
    vec_t m_target;
    auto eq_idx() const -> indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>> {
      (void)this;
      auto unfiltered = indexing::vec_regular_indexer(2, horizon + 2, dyn_index{m_target.rows()});
      return {unfiltered, horizon, horizon + 1};
    }
    auto operator[](index_t t) const
        -> eigen::matrix_view_t<scalar_t const, dyn_index, fix_index<1>, dyn_index, fix_index<1>> {
      if (t != horizon) {
        DDP_ASSERT(eq_idx().rows(t).value() == 0);
        return {nullptr, 0, 1, 0};
      }
      DDP_ASSERT(eq_idx().rows(t).value() == m_target.rows());
      return eigen::as_const_view(m_target);
    }
  };
  struct velocity_target_range_t {
    Eigen::Matrix<scalar_t, -1, 1> m_target;

    auto eq_idx() const -> indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>> {
      (void)this;
      auto unfiltered = indexing::vec_regular_indexer(1, horizon + 1, dyn_index{m_target.rows()});
      return {unfiltered, horizon, horizon + 1};
    }
    auto operator[](index_t t) const
        -> eigen::matrix_view_t<scalar_t const, dyn_index, fix_index<1>, dyn_index, fix_index<1>> {
      if (t != horizon) {
        DDP_ASSERT(eq_idx().rows(t).value() == 0);
        return {nullptr, 0, 1, 0};
      }
      DDP_ASSERT(eq_idx().rows(t).value() == m_target.rows());
      return eigen::as_const_view(m_target);
    }
  };

  auto dy = pinocchio_dynamics(model, 0.01, false);

  auto q0 = eigen::make_matrix<scalar_t>(nq);
  model.neutral_configuration(eigen::as_mut_view(q0));

  auto constr_feet = [&](index_t idx) {
    Eigen::Matrix<scalar_t, 3, 1> q;

    model.frame_coordinates(
        eigen::as_mut_view(q),
        foot_frames[idx],
        eigen::as_const_view(q0),
        model.acquire_workspace());
    return constraint_advance_time<2>(spatial_constraint(dy, DDP_MOVE(position_target_range_t{q}), foot_frames[idx]));
  };

  auto eq_gen = config_target_range_t{q0};
  auto eq_gen_vel = velocity_target_range_t{eigen::make_matrix<scalar_t>(nv)};
  auto constr0 = constr_feet(0);
  auto constr1 = constr_feet(1);
  auto constr2 = constr_feet(2);
  auto constr3 = constr_feet(3);
  auto constr4 = constraint_advance_time<1>(velocity_constraint(dy, DDP_MOVE(eq_gen_vel)));
  auto constr5 = constraint_advance_time<2>(config_constraint(dy, DDP_MOVE(eq_gen)));

  auto constr = concat_constraint(constr0, constr1, constr2, constr3, constr4, constr5);

  auto prob_ = problem(0, horizon, 1.0, dy, constr);

  auto prob = multi_shooting(prob_, indexing::periodic_row_filter(prob_.state_indexer(0, horizon), 3, 1));
  auto u_idx = indexing::row_concat(indexing::vec_regular_indexer(0, horizon, nv), prob.m_slack_idx);

  using problem_t = decltype(prob);

  auto eq_idx = prob.constraint().eq_idx();

  auto x_init = [&] {
    auto x = eigen::make_matrix<scalar_t>(nq + nv, fix_index<1>{});

    DDP_BIND(auto, (q, v), eigen::split_at_row_mut(x, nq));
    model.neutral_configuration(q);
    v.setZero();
    return x;
  }();

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
      m_value.setZero();
    }
  };

  ddp_solver_t<problem_t> solver{prob, u_idx, eq_idx, x_init};

  {
    using std::pow;

    constexpr auto M = method::primal_dual_affine_multipliers;

    auto derivs = solver.uninit_derivative_storage();

    scalar_t const mu_init = 1e2;
    auto res = solver.solve<M>({200, 1e-80, mu_init}, solver.make_trajectory(control_generator_t{u_idx}));
    DDP_BIND(auto&&, (traj, fb), res);
    (void)fb;

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }
    fmt::print("x: {}\n", traj.x_f().transpose());
  }
}
