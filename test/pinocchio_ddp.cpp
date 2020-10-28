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
  constexpr static index_t horizon = 200;

  for (index_t i = 0; i < model.n_frames(); ++i) {
    fmt::print("{}\n", model.frame_name(i));
  }

  auto make_config = [&](std::initializer_list<scalar_t> values) {
    auto q = eigen::make_matrix<scalar_t>(nq, fix_index<1>{});
    model.neutral_configuration(eigen::as_mut_view(q));
    for (index_t i = 0; i < model.configuration_dim(); ++i) {
      q[i] = *(values.begin() + i);
    }
    return q;
  };

  using filter_t = indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>>;

  struct {
    model_t const& model;
    index_t m_ts[4];
    vec_t m_targets[3];
    auto eq_idx() const {
      auto range_idx = [&](index_t s, index_t f) {
        return filter_t{indexing::vec_regular_indexer(2, horizon + 2, model.configuration_dim_c()), s, f};
      };
      return row_concat(
          row_concat(range_idx(m_ts[0], m_ts[0] + 1), range_idx(m_ts[1], m_ts[1] + 1)),
          range_idx(m_ts[2], m_ts[3]));
    };
    auto operator[](index_t t) const -> eigen::view_t<vec_t const> {
      if (t == m_ts[0]) {
        return eigen::as_const_view(m_targets[0]);
      }
      if (t == m_ts[1]) {
        return eigen::as_const_view(m_targets[1]);
      }
      if (t >= m_ts[2] and t < m_ts[3]) {
        return eigen::as_const_view(m_targets[2]);
      }
      return {nullptr, 0, 1};
    }
  } eq_gen = {
      model,
      {horizon / 4, horizon / 4 * 2, horizon / 4 * 3, horizon + 2},
      {
          make_config({1.8, -0.78, 0, 0, 0, 1}),
          make_config({0, -0.78, 0, 0, 0, 1}),
          make_config({0.0, -1.57, 0.0, 0.0, 0.0, 0.0}),
      }};

  for (auto& target : eq_gen.m_targets) {
    vec_t out{3};
    model.frame_coordinates_precompute(eigen::as_const_view(target), model.acquire_workspace());
    model.frame_coordinates(eigen::into_view(eigen::as_mut_view(out)), 18, model.acquire_workspace());
    fmt::print("{}\n", out.transpose());
  }
  std::terminate();

  auto x_init = [&] {
    auto x = eigen::make_matrix<scalar_t>(nq + nv, fix_index<1>{});

    DDP_BIND(auto, (q, v), eigen::split_at_row_mut(x, nq));
    model.neutral_configuration(q);
    v.setZero();
    return x;
  }();

  auto dy = pinocchio_dynamics(model, 0.01, false);
  auto prob = problem(0, horizon, 1.0, dy, constraint_advance_time<2>(config_constraint(dy, DDP_MOVE(eq_gen))));
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
