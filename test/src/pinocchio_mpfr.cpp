#include <pinocchio/math/multiprecision-mpfr.hpp>
#include <ddp/pinocchio_model.ipp>
#include <boost/multiprecision/mpfr.hpp>

namespace boostmp = boost::multiprecision;
using scalar = boostmp::number<
		boostmp::backends::mpfr_float_backend<500, boostmp::allocate_stack>,
		boostmp::et_off>;

template struct ddp::pinocchio::model<scalar>;
