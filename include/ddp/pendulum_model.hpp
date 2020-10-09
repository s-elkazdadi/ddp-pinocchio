#ifndef PENDULUM_MODEL_HPP_J49J0ZTO
#define PENDULUM_MODEL_HPP_J49J0ZTO

#include "ddp/indexer.hpp"
#include <memory>
#include <random>

namespace ddp {

template <typename T>
struct pendulum_model_t {
  using scalar_t = T;
  static constexpr bool fixed_size = true;
  using state_dim_kind = fix_index<1>;
  using dstate_dim_kind = fix_index<1>;

private:
  using mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 1, 1>>;
  using const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 1, 1> const>;

  using row_major_mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 1, 1, Eigen::RowMajor>>;
  using row_major_const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, 1, 1, Eigen::RowMajor> const>;

  scalar_t m_mass;
  scalar_t m_length;
  static constexpr long double g = 9.81L;

public:
  pendulum_model_t(scalar_t mass, scalar_t length) : m_mass{mass}, m_length{length} {
    DDP_ASSERT_MSG_ALL_OF(                       //
        ("mass should be positive", m_mass > 0), //
        ("length should be positive", m_length > 0));
  }

  auto configuration_dim() const noexcept -> index_t { return 1; }
  auto tangent_dim() const noexcept -> index_t { return 1; }

  auto configuration_dim_c() const noexcept -> fix_index<1> { return {}; }
  auto tangent_dim_c() const noexcept -> fix_index<1> { return {}; }

  void neutral_configuration(mut_view_t out_q) const noexcept { out_q[0] = 0; }
  void random_configuration(mut_view_t out_q) const noexcept {
    constexpr long double pi = 3.141592653589793238462643383279L;
    static thread_local auto mt = std::mt19937{std::random_device{}()};

    out_q[0] = std::uniform_real_distribution<long double>{-pi, pi}(mt);
  }

  void integrate(       //
      mut_view_t out_q, //
      const_view_t q,   //
      const_view_t v    //
  ) const noexcept {
    out_q = q + v;
  }

  void d_integrate_dq(     //
      mut_view_t out_q_dq, //
      const_view_t q,      //
      const_view_t v       //
  ) const noexcept {
    (void)q;
    (void)v;
    out_q_dq[0] = 1;
  }

  void d_integrate_dv(     //
      mut_view_t out_q_dv, //
      const_view_t q,      //
      const_view_t v       //
  ) const noexcept {
    (void)q;
    (void)v;
    out_q_dv[0] = 1;
  }

  void difference(          //
      mut_view_t out_v,     //
      const_view_t q_start, //
      const_view_t q_finish //
  ) const noexcept {
    out_v = q_finish - q_start;
  }

  void d_difference_dq_start(    //
      mut_view_t out_v_dq_start, //
      const_view_t q_start,      //
      const_view_t q_finish      //
  ) const noexcept {
    (void)q_start;
    (void)q_finish;
    out_v_dq_start[0] = -1;
  }

  void d_difference_dq_finish(    //
      mut_view_t out_v_dq_finish, //
      const_view_t q_start,       //
      const_view_t q_finish       //
  ) const noexcept {
    (void)q_start;
    (void)q_finish;
    out_v_dq_finish[0] = 1;
  }

  void dynamics_aba(               //
      mut_view_t out_acceleration, //
      const_view_t q,              //
      const_view_t v,              //
      const_view_t tau             //
  ) const noexcept {
    (void)v;
    using std::sin;
    out_acceleration[0] = -g / m_length * sin(q[0]) + tau[0] / m_mass;
  }

  void d_dynamics_aba(                  //
      mut_view_t out_acceleration_dq,   //
      mut_view_t out_acceleration_dv,   //
      mut_view_t out_acceleration_dtau, //
      const_view_t q,                   //
      const_view_t v,                   //
      const_view_t tau                  //
  ) const noexcept {
    (void)v;
    (void)tau;
    using std::cos;
    out_acceleration_dq[0] = -g / m_length * cos(q[0]);
    out_acceleration_dv[0] = 0;
    out_acceleration_dtau[0] = 1 / m_mass;
  }

  auto model_name() const noexcept -> fmt::string_view { return "pendulum"; }
};

} // namespace ddp

#endif /* end of include guard PENDULUM_MODEL_HPP_J49J0ZTO */
