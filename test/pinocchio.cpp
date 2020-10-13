#include "ddp/pinocchio_model.hpp"

#include <boost/multiprecision/mpfr.hpp>
#include <doctest/doctest.h>
#include <fmt/ostream.h>

using scalar_t =                                                        //
    boost::multiprecision::number<                                      //
        boost::multiprecision::mpfr_float_backend<                      //
            1000,                                                       //
            boost::multiprecision::mpfr_allocation_type::allocate_stack //
            >,                                                          //
        boost::multiprecision::et_off>;

auto const& eps = std::numeric_limits<scalar_t>::epsilon();

DOCTEST_TEST_CASE("integrate-difference") {
  using namespace ddp;
  using eigen::as_const_view;
  using eigen::as_mut_view;
  using vec_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;

  auto model = ddp::pinocchio::model_t<scalar_t>::all_joints_test_model();

  auto nq = model.configuration_dim();
  auto nv = model.tangent_dim();

  vec_t q0{nq};
  vec_t q1{nq};
  vec_t q2{nq};

  vec_t v0{nv};
  vec_t v1{nv};

  model.random_configuration(as_mut_view(q0));
  q2 = q0;
  v0.setRandom();

  model.integrate( //
      as_mut_view(q1),
      as_const_view(q0),
      as_const_view(v0));
  model.difference( //
      as_mut_view(v1),
      as_const_view(q0),
      as_const_view(q1));

  DOCTEST_CHECK(v0.isApprox(v1, eps * 1e3));

  v0 = -v0;
  model.integrate( //
      as_mut_view(q0),
      as_const_view(q1),
      as_const_view(v0));

  DOCTEST_CHECK(q0.isApprox(q2, eps * 1e3));
}

DOCTEST_TEST_CASE("affine-function") {
  using namespace ddp;
  using eigen::as_const_view;
  using eigen::as_mut_view;

  using vec_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
  using mat_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
  using rvec_t = Eigen::Matrix<scalar_t, 1, Eigen::Dynamic>;

  auto model = ddp::pinocchio::model_t<scalar_t>::all_joints_test_model();

  index_t nq = model.configuration_dim();
  index_t nv = model.tangent_dim();

  scalar_t f0{Eigen::Matrix<scalar_t, 1, 1>::Random().value()};
  auto j0 = (rvec_t::Random(1, nv)).eval();

  vec_t q0{nq};
  model.random_configuration(as_mut_view(q0));
  vec_t q1{nq};
  model.random_configuration(as_mut_view(q1));
  vec_t v1{nv};
  model.difference(as_mut_view(v1), as_const_view(q0), as_const_view(q1));

  scalar_t dh = 1e-50;

  mat_t J{nv, nv};
  model.d_difference_dq_finish(as_mut_view(J), as_const_view(q0), as_const_view(q1));
  vec_t dv = (vec_t::Random(nv) * dh).eval();

  vec_t q2{nq};
  model.integrate(as_mut_view(q2), as_const_view(q1), as_const_view(dv));
  vec_t v2{nv};
  model.difference(as_mut_view(v2), as_const_view(q0), as_const_view(q2));

  scalar_t f1 = f0 + (j0 * v1).value();
  auto j1 = (j0 * J).eval();

  scalar_t eps2 = dh * dh * 1e8;
  DOCTEST_CHECK(fabs((f0 + (j0 * v2).value()) - (f1 + (j1 * dv).value())) < eps2);
  DOCTEST_CHECK((v2 - v1 - J * dv).norm() < eps2);
}
