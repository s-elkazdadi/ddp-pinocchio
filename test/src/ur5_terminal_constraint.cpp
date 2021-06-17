#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(5, 0, "  ", "\n", "", "")

#include "ddp/dynamics.hpp"
#include "ddp/constraint.hpp"
#include "ddp/cost.hpp"
#include "ddp/ddp.hpp"
#include <omp.h>
#include <fmt/core.h>
#include <iostream>
#include <example-robot-data/path.hpp>

#include <boost/multiprecision/mpfr.hpp>

namespace boostmp = boost::multiprecision;
using bignum_boost = boostmp::number<
		boostmp::backends::mpfr_float_backend<500, boostmp::allocate_stack>,
		boostmp::et_off>;

using scalar = double;

namespace eigen = ddp::eigen;
using namespace veg::literals;
using veg::i64;
using veg::usize;

auto main() -> int {
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	std::cout << std::scientific;
	std::cout.precision(20);
	i64 horizon = 10;
	eigen::Matrix<scalar, ddp::colvec> empty(0);

	ddp::pinocchio::Model<scalar> m{
			EXAMPLE_ROBOT_DATA_MODEL_DIR "/ur_description/urdf/ur5_gripper.urdf",
			::omp_get_num_procs()};

	auto nq = m.config_dim();
	auto nv = m.tangent_dim();
	eigen::Matrix<scalar, ddp::colvec> target(nq);
	target.setZero();

	i64 nx = nq + nv;
	i64 ndx = nv + nv;
	i64 nu = nv;
	i64 ndu = nv;

	auto dynamics = ddp::dynamics::pinocchio_free(m, 0.01);

	auto constraint =                     //
			ddp::constraint::advance_time<2>( //
					ddp::constraint::config(      //
							dynamics,
							[&](i64 t, ddp::DynStackView stack) {
								(void)stack;
								if (t == horizon) {
									return eigen::as_const(target);
								}
								return eigen::as_const(empty);
							},
							target.rows(),
							ddp::MemReq{ddp::tag<scalar>, 0}));

	auto cost = [&] {
		eigen::HeapMatrix<scalar, ddp::colvec> u0{ddp::eigen::with_dims, ndu};
		eigen::HeapMatrix<scalar, ddp::colmat> U0{ddp::eigen::with_dims, ndu, ndu};
		eigen::HeapMatrix<scalar, ddp::colvec> x0{ddp::eigen::with_dims, ndx};
		eigen::HeapMatrix<scalar, ddp::colmat> X0{ddp::eigen::with_dims, ndx, ndx};
		eigen::HeapMatrix<scalar, ddp::colvec> xf{ddp::eigen::with_dims, ndx};
		eigen::HeapMatrix<scalar, ddp::colmat> Xf{ddp::eigen::with_dims, ndx, ndx};

		u0.mut().setZero();
		x0.mut().setZero();
		xf.mut().setZero();

		U0.mut().setIdentity();
		X0.mut().setZero();
		Xf.mut().setZero();

		return ddp::cost::homogeneous_quadratic(
				VEG_FWD(u0),
				VEG_FWD(U0),
				VEG_FWD(x0),
				VEG_FWD(X0),
				VEG_FWD(xf),
				VEG_FWD(Xf));
	}();

	auto solver = ddp::ddp(dynamics, cost, constraint);

	auto traj = [&] {
		eigen::Matrix<scalar, ddp::colvec> x0(nx);
		dynamics.neutral_configuration(eigen::as_mut(x0));
		auto zero = eigen::Matrix<scalar, ddp::colvec>::Zero(nu).eval();

		return solver.make_trajectory( //
				0,
				horizon,
				eigen::as_const(x0),
				[&](auto /*x_init*/) { return eigen::as_const(zero); });
	}();

	auto d = solver.make_derivative_storage(traj);
	auto res = solver.solve<ddp::Method_e::affine_multipliers>(
			200, 0, 1e2, VEG_FWD(traj));
	fmt::print("{}\n", res[0_c].x_f());
}
