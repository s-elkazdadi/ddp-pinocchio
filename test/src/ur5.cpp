#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(5, 0, "  ", "\n", "", "")

#include "ddp/dynamics.hpp"
#include "ddp/constraint.hpp"
#include "ddp/cost.hpp"
#include "ddp/ddp.hpp"
#include <omp.h>
#include <fmt/core.h>
#include <example-robot-data/path.hpp>

#include <boost/multiprecision/mpfr.hpp>

namespace mp = boost::multiprecision;
using bignum = mp::number<
		mp::backends::mpfr_float_backend<500, mp::allocate_stack>,
		mp::et_off>;

using scalar = bignum;

namespace eigen = ddp::eigen;
using namespace veg::literals;
using veg::i64;
using veg::usize;

auto main() -> int {
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	i64 horizon = 10;
	eigen::matrix<scalar, ddp::colvec> empty(0);

	ddp::pinocchio::model<scalar> m{
			EXAMPLE_ROBOT_DATA_MODEL_DIR "/ur_description/urdf/ur5_gripper.urdf",
			::omp_get_num_procs()};

	auto nq = m.config_dim();
	auto nv = m.tangent_dim();

	i64 nx = nq + nv;
	i64 ndx = nv + nv;
	i64 nu = nv;
	i64 ndu = nv;

	auto dynamics = ddp::make::pinocchio_dynamics_free(m, 0.01);

	auto constraint = ddp::no_constraint<decltype(dynamics)>{};

	auto cost = [&] {
		eigen::heap_matrix<scalar, ddp::colvec> u0(ndu);
		eigen::heap_matrix<scalar, ddp::colmat> U0(ndu, ndu);
		eigen::heap_matrix<scalar, ddp::colvec> x0(ndx);
		eigen::heap_matrix<scalar, ddp::colmat> X0(ndx, ndx);
		eigen::heap_matrix<scalar, ddp::colvec> xf(ndx);
		eigen::heap_matrix<scalar, ddp::colmat> Xf(ndx, ndx);

		u0.mut().setZero();
		x0.mut().setZero();
		xf.mut().setZero();

		U0.mut().setIdentity();
		X0.mut().setZero();
		Xf.mut().setIdentity();
		Xf.mut() *= 1e2;

		return ddp::make::quadratic_cost_fixed_size(
				VEG_FWD(u0),
				VEG_FWD(U0),
				VEG_FWD(x0),
				VEG_FWD(X0),
				VEG_FWD(xf),
				VEG_FWD(Xf));
	}();

	auto solver = ddp::make::ddp(dynamics, cost, constraint);

	auto traj = [&] {
		eigen::matrix<scalar, ddp::colvec> x0(nx);
		dynamics.neutral_configuration(eigen::as_mut(x0));
		auto zero = eigen::matrix<scalar, ddp::colvec>::Zero(nu).eval();

		return solver.make_trajectory( //
				0,
				horizon,
				eigen::as_const(x0),
				[&](auto /*x_init*/) { return eigen::as_const(zero); });
	}();

	auto d = solver.make_derivative_storage(traj);

	auto res = solver.solve<ddp::method::affine_multipliers>(200, 0, 1e2, traj);
	fmt::print("{}\n", res[0_c].x_f().transpose());
}
