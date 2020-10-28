#ifndef CART_MODEL_HPP_XBXDUEUC
#define CART_MODEL_HPP_XBXDUEUC

#include "ddp/indexer.hpp"
#include <cmath>
#include <random>

namespace ddp {

template <typename T>
struct cart_pendulum_model_t {
  using scalar_t = T;
  static constexpr bool fixed_size = true;
  using state_dim_kind = fix_index<2>;
  using dstate_dim_kind = fix_index<2>;

private:
  using mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 2, 1>>;
  using mmut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 2, 2>>;
  using const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 2, 1> const>;

  scalar_t M;
  scalar_t m;
  scalar_t l;
  static constexpr long double g = 9.81L;

public:
  struct key {
    explicit operator bool() { return true; }
  };

  cart_pendulum_model_t(scalar_t mass_cart, scalar_t mass_pendulum, scalar_t length)
      : M{mass_cart}, m{mass_pendulum}, l{length} {
    DDP_ASSERT_MSG_ALL_OF( //
        ("mass should be positive", M > 0),
        ("mass should be positive", m > 0), //
        ("length should be positive", l > 0));
  }

  auto configuration_dim() const -> index_t { return 2; }
  auto tangent_dim() const -> index_t { return 2; }

  auto configuration_dim_c() const -> fix_index<2> { return {}; }
  auto tangent_dim_c() const -> fix_index<2> { return {}; }

  auto acquire_workspace() const -> key { return {}; };

  void neutral_configuration(mut_view_t out_q) const {
    out_q[0] = 0;
    out_q[1] = 0;
  }
  void random_configuration(mut_view_t out_q) const {
    constexpr long double pi = 3.141592653589793238462643383279L;
    static thread_local auto mt = std::mt19937{std::random_device{}()};

    out_q[0] = std::uniform_real_distribution<long double>{-pi, pi}(mt);
    out_q[1] = std::uniform_real_distribution<long double>{-1, 1}(mt);
  }

  void integrate(       //
      mut_view_t out_q, //
      const_view_t q,   //
      const_view_t v    //
  ) const {
    out_q = q + v;
  }

  void d_integrate_dq(      //
      mmut_view_t out_q_dq, //
      const_view_t q,       //
      const_view_t v        //
  ) const {
    (void)q;
    (void)v;
    out_q_dq.setIdentity();
  }

  void d_integrate_dv(      //
      mmut_view_t out_q_dv, //
      const_view_t q,       //
      const_view_t v        //
  ) const {
    (void)q;
    (void)v;
    out_q_dv.setIdentity();
  }

  void difference(          //
      mut_view_t out_v,     //
      const_view_t q_start, //
      const_view_t q_finish //
  ) const {
    out_v = q_finish - q_start;
  }

  void d_difference_dq_start(     //
      mmut_view_t out_v_dq_start, //
      const_view_t q_start,       //
      const_view_t q_finish       //
  ) const {
    (void)q_start, (void)q_finish;
    out_v_dq_start.setIdentity();
    out_v_dq_start *= -1;
  }

  void d_difference_dq_finish(     //
      mmut_view_t out_v_dq_finish, //
      const_view_t q_start,        //
      const_view_t q_finish        //
  ) const {
    (void)q_start, (void)q_finish;
    out_v_dq_finish.setIdentity();
  }

  auto dynamics_aba(               //
      mut_view_t out_acceleration, //
      const_view_t q,              //
      const_view_t v,              //
      const_view_t tau,            //
      key k                        //
  ) const -> key {
    (void)v;
    using std::cos;
    using std::sin;
    scalar_t c = cos(q[1]);
    scalar_t s = sin(q[1]);
    out_acceleration[0] = (m * (+g) * s * c + tau[0] + m * l * v[1] * v[1]) / (M + s * s * m);
    out_acceleration[1] = (-g) / l * s - c / l * out_acceleration[0];
    return k;
  }

  auto d_dynamics_aba(                   //
      mmut_view_t out_acceleration_dq,   //
      mmut_view_t out_acceleration_dv,   //
      mmut_view_t out_acceleration_dtau, //
      const_view_t q,                    //
      const_view_t v,                    //
      const_view_t tau,                  //
      key k                              //
  ) const -> key {
    scalar_t c = cos(q[1]);
    scalar_t s = sin(q[1]);

    scalar_t m0 = M + s * s * m;

    scalar_t out_ac = (m * (+g) * s * c + tau[0] + m * l * v[1] * v[1]) / (M + s * s * m);

    out_acceleration_dq(0, 0) = 0;
    out_acceleration_dq(1, 0) = 0;
    out_acceleration_dv(0, 0) = 0;
    out_acceleration_dv(1, 0) = 0;
    out_acceleration_dtau(0, 1) = 0;
    out_acceleration_dtau(1, 1) = 0;

    out_acceleration_dq(0, 1) =
        (m * +g * (c * c - s * s) * m0 - 2 * s * c * m * (m * +g * s * c + tau[0] + m * l * v[1] * v[1])) / (m0 * m0);
    out_acceleration_dq(1, 1) = -g / l * c - c / l * out_acceleration_dq(0, 1) + s / l * out_ac;

    out_acceleration_dv(0, 1) = 2 * m * l / m0 * v[1];
    out_acceleration_dv(1, 1) = -c / l * out_acceleration_dv(0, 1);

    out_acceleration_dtau(0, 0) = 1 / m0;
    out_acceleration_dtau(1, 0) = -c / l * out_acceleration_dtau(0, 0);

    using std::cos;
    return k;
  }

  auto model_name() const -> fmt::string_view { return "pendulum"; }
};

} // namespace ddp

#endif /* end of include guard CART_MODEL_HPP_XBXDUEUC */
