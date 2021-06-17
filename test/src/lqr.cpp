#include "ddp/dynamics.hpp"
#include "ddp/constraint.hpp"
#include "ddp/cost.hpp"
#include "ddp/ddp.hpp"
#include <fmt/core.h>

#include <boost/multiprecision/mpfr.hpp>

namespace boostmp = boost::multiprecision;
using bignum_boost = boostmp::number<
		boostmp::backends::mpfr_float_backend<500, boostmp::allocate_stack>,
		boostmp::et_off>;
using scalar = double;

DDP_CHECK_CONCEPT(scalar<scalar>);

namespace eigen = ddp::eigen;
using namespace veg::literals;
using veg::i64;

auto main() -> int {
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	i64 horizon = 10;
	eigen::matrix<scalar, ddp::colvec> empty(0);

	eigen::matrix<scalar, ddp::colmat> a(3, 3);
	eigen::matrix<scalar, ddp::colmat> b(3, 3);
	eigen::matrix<scalar, ddp::colvec> c(3);
	a.setRandom();
	b.setRandom();
	c.setZero();
	auto dynamics = ddp::make::lqr_dynamics(a, b, c);
	auto constraint = ddp::no_constraint<decltype(dynamics)>{};
	i64 nx = 3;
	i64 ndx = 3;
	i64 nu = 3;
	i64 ndu = 3;

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
		Xf.mut() *= 1e3;

		xf.mut()[0] = 0.1;
		xf.mut()[1] = 0.2;
		xf.mut()[2] = 0.3;
		xf.mut() *= 1e3;

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
		eigen::heap_matrix<scalar, ddp::colvec> x0(nx);
		eigen::heap_matrix<scalar, ddp::colvec> zero(nu);
		dynamics.neutral_configuration(x0.mut());

		return solver.make_trajectory( //
				0,
				horizon,
				x0.get(),
				[&](auto /*x_init*/) { return zero.get(); });
	}();

	// fmt::print("{}\n", m.model_name());
	// fmt::print("{}\n", q0.transpose());

	solver.solve<ddp::method::affine_multipliers>(5, 0, 1e2, traj);
}
