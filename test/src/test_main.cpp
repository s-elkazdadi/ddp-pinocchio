#include "ddp/dynamics.hpp"
#include "ddp/constraint.hpp"
#include "ddp/cost.hpp"
#include "ddp/ddp.hpp"
#include <fmt/core.h>

using scalar = double;

namespace eigen = ddp::eigen;
using namespace veg::literals;

auto main() -> int {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  ddp::pinocchio::model<scalar> m{EXAMPLE_ROBOT_DATA_MODEL_DIR
                                  "/ur_description/urdf/ur5_gripper.urdf"};

  auto nq = m.config_dim();
  auto nv = m.tangent_dim();

  auto dynamics = ddp::make::pinocchio_dynamics_free(m, 0.01);
  eigen::matrix<scalar, ddp::colvec> q0(m.config_dim());
  dynamics.neutral_configuration(eigen::as_mut(q0));

  auto constraint = ddp::make::config_constraint(
      dynamics,
      [&](veg::i64 /*t*/, veg::dynamic_stack_view /*stack*/) {
        return eigen::as_const(q0);
      },
      ddp::mem_req{veg::tag<scalar>, 0});

  auto cost = [&] {
    eigen::matrix<scalar, ddp::colvec> u0(nv);
    eigen::matrix<scalar, ddp::colmat> U0(nv, nv);
    eigen::matrix<scalar, ddp::colvec> x0(nq + nv);
    eigen::matrix<scalar, ddp::colmat> X0(nq + nv, nq + nv);
    u0.setZero();
    x0.setZero();

    U0.setIdentity();
    X0.setZero();
    fmt::print("{:>8.3e}\n", U0);

    return ddp::make::quadratic_cost_fixed_size(
        eigen::as_const(u0),
        eigen::as_const(U0),
        eigen::as_const(x0),
        eigen::as_const(X0),
        eigen::as_const(x0),
        eigen::as_const(X0));
  }();

  auto solver =
      ddp::make::ddp(VEG_FWD(dynamics), VEG_FWD(cost), VEG_FWD(constraint));

  auto stack_storage = std::vector<unsigned char>(veg::narrow<veg::usize>(
      dynamics.eval_to_req().size + dynamics.eval_to_req().align));

  auto traj = [&] {
    eigen::matrix<scalar, ddp::colvec> x0(nq + nv);
    dynamics.neutral_configuration(
        eigen::split_at_row(x0, m.config_dim())[0_c]);
    auto zero = decltype(x0)::Zero(nv).eval();

    return solver.make_trajectory( //
        0,
        20,
        eigen::as_const(x0),
        [&](auto /*x_init*/) { return eigen::as_const(zero); });
  }();

  fmt::print("{}\n", m.model_name());
  fmt::print("{}\n", q0.transpose());

  solver.solve<ddp::method::affine_multipliers>(traj);
}
