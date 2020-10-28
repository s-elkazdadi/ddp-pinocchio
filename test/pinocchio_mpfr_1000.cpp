#include "ddp/pinocchio_model.ipp"

#include <pinocchio/math/multiprecision.hpp>
#include <pinocchio/math/multiprecision-mpfr.hpp>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/mpfr.hpp>

template struct ddp::pinocchio::model_t<                                //
    boost::multiprecision::number<                                      //
        boost::multiprecision::mpfr_float_backend<                      //
            500,                                                        //
            boost::multiprecision::mpfr_allocation_type::allocate_stack //
            >,                                                          //
        boost::multiprecision::et_off>                                  //
    >;
