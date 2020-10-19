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

  for (index_t i = 0; i < model.n_frames(); ++i) {
    fmt::print("{}\n", model.frame_name(i));
  }

  struct constraint_t {
    using vec3 = decltype(eigen::make_matrix<scalar_t>(fix_index<3>{}));
    vec3 m_target;
    auto eq_idx() const -> indexing::range_row_filter_t<indexing::regular_indexer_t<dyn_index>> {
      (void)this;
      auto unfiltered = indexing::vec_regular_indexer(2, horizon + 2, dyn_index{3});
      return {unfiltered, horizon, horizon + 1};
    }
    auto operator[](index_t t) const -> Eigen::Map<vec_t const> {
      static const vec_t empty{};
      if (t != horizon) {
        return {nullptr, 0, 1};
      }
      return {m_target.data(), m_target.rows(), m_target.cols()};
    }
  };

  auto eq_gen = constraint_t{[&] {
    auto q0 = eigen::make_matrix<scalar_t>(nq);
    model.neutral_configuration(eigen::as_mut_view(q0));

    vec_t q{3};
    DDP_BIND(auto, (out_3, out_0), eigen::split_at_row_mut(q, fix_index<3>{}));
    (void)out_0;
    model.frame_coordinates(out_3, model.n_frames() - 1, eigen::as_const_view(q0), model.acquire_workspace());
    return q;
  }()};

  auto x_init = [&] {
    auto x = eigen::make_matrix<scalar_t>(nq + nv, fix_index<1>{});

    DDP_BIND(auto, (q, v), eigen::split_at_row_mut(x, nq));
    model.neutral_configuration(q);
    v.setZero();
    return x;
  }();

  auto dy = pinocchio_dynamics(model, 0.01, false);
  auto prob_ = problem(
      0,
      horizon,
      1.0,
      dy,
      constraint_advance_time<2>(spatial_constraint(dy, DDP_MOVE(eq_gen), model.n_frames() - 1)));

#if 1
  auto prob = multi_shooting(prob_, indexing::periodic_row_filter(prob_.state_indexer(0, horizon), 3, 1));
  auto u_idx = indexing::row_concat(indexing::vec_regular_indexer(0, horizon, nv), prob.m_slack_idx);
#else
  auto prob = prob_;
  auto u_idx = indexing::vec_regular_indexer(0, horizon, nv);
#endif
  using problem_t = decltype(prob);


  auto eq_idx = prob.constraint().eq_idx();

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
