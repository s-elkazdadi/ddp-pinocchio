#include <pinocchio/math/multiprecision-mpfr.hpp>
#include <ddp/pinocchio_model.ipp>
#include <boost/multiprecision/mpfr.hpp>

namespace mp = boost::multiprecision;
using scalar = mp::number<
    mp::backends::mpfr_float_backend<500, mp::allocate_stack>,
    mp::et_off>;

template struct ddp::pinocchio::model<scalar>;
