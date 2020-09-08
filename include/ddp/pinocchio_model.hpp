#ifndef PINOCCHIO_INTERFACE_HPP_EVURJTS1
#define PINOCCHIO_INTERFACE_HPP_EVURJTS1

#include "ddp/indexer.hpp"
#include <memory>

namespace ddp {
namespace pinocchio {

namespace gsl {
template <typename T>
using owner = T;
} // namespace gsl

template <typename T, index_t Nq = Eigen::Dynamic, index_t Nv = Eigen::Dynamic>
struct model_t {
private:
  struct impl_model_t;
  struct impl_data_t;

  gsl::owner<impl_model_t*> m_model = nullptr;
  gsl::owner<impl_data_t*> m_data = nullptr;
  index_t m_num_data = 0;
  index_t m_config_dim = 0;
  index_t m_tangent_dim = 0;

public:
  using scalar_t = T;
  template <index_t Rows, index_t Cols = 1>
  using mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols>>;

  template <index_t Rows, index_t Cols = 1>
  using const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols> const>;

  template <index_t Rows, index_t Cols = 1>
  using row_major_mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols, Eigen::RowMajor>>;
  template <index_t Rows, index_t Cols = 1>
  using row_major_const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols, Eigen::RowMajor> const>;

  static constexpr bool fixed_size = Nq != Eigen::Dynamic and Nv != Eigen::Dynamic;

  using state_size_kind = DDP_CONDITIONAL(fixed_size, fix_index<Nq + Nv>, dyn_index);
  using dstate_size_kind = DDP_CONDITIONAL(fixed_size, fix_index<Nv + Nv>, dyn_index);

  static constexpr index_t eigen_config_size_c = fixed_size ? Nq : Eigen::Dynamic;
  static constexpr index_t eigen_vel_size_c = fixed_size ? Nv : Eigen::Dynamic;

  static constexpr index_t eigen_state_size_c = fixed_size ? (Nq + Nv) : Eigen::Dynamic;
  static constexpr index_t eigen_dstate_size_c = fixed_size ? (Nv + Nv) : Eigen::Dynamic;

  using state_indexer_t = indexing::regular_indexer_t<state_size_kind>;
  using dstate_indexer_t = indexing::regular_indexer_t<dstate_size_kind>;

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return indexing::vec_regular_indexer(begin, end, state_size_kind{configuration_dim() + tangent_dim()});
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return indexing::vec_regular_indexer(begin, end, dstate_size_kind{tangent_dim() + tangent_dim()});
  }

private:
  auto get_data() const noexcept -> impl_data_t*;

  template <typename Model_Builder>
  model_t(Model_Builder&& model_builder, index_t n_parallel) noexcept(false);

  void validate_config_vector(
      const_view_t<Nq> q,   //
      fmt::string_view name //
  ) const noexcept;
  void validate_tangent_vector(
      const_view_t<Nv> v,   //
      fmt::string_view name //
  ) const noexcept;
  void validate_tangent_matrix(
      const_view_t<Nv, Nv> m, //
      fmt::string_view name   //
  ) const noexcept;
  template <typename Out, typename In>
  void validate_similar_jac_matrices(
      Out m1,                 //
      In m2,                  //
      fmt::string_view name1, //
      fmt::string_view name2  //
  ) const noexcept;
  template <typename Out, typename In>
  void
  d_integrate_transport_dq_or_dv(Out out_J, In in_J, const_view_t<Nq> q, const_view_t<Nv> v, bool dq) const noexcept;

public:
  ~model_t() noexcept;
  model_t(model_t const&) = delete;
  model_t(model_t&&) noexcept;

  auto operator=(model_t const&) -> model_t& = delete;
  auto operator=(model_t &&) -> model_t& = delete;

  explicit model_t(fmt::string_view urdf_path, index_t n_parallel = 1) noexcept(false);
  static auto all_joints_test_model(index_t n_parallel = 1) noexcept(false) -> model_t;

  auto configuration_dim() const noexcept -> index_t { return m_config_dim; }
  auto tangent_dim() const noexcept -> index_t { return m_tangent_dim; }

  void neutral_configuration(mut_view_t<Nq> out_q) const noexcept;
  void random_configuration(mut_view_t<Nq> out_q) const noexcept;

  void integrate(           //
      mut_view_t<Nq> out_q, //
      const_view_t<Nq> q,   //
      const_view_t<Nv> v    //
  ) const noexcept;

  void d_integrate_dq(             //
      mut_view_t<Nv, Nv> out_q_dq, //
      const_view_t<Nq> q,          //
      const_view_t<Nv> v           //
  ) const noexcept;

  void d_integrate_dv(             //
      mut_view_t<Nv, Nv> out_q_dv, //
      const_view_t<Nq> q,          //
      const_view_t<Nv> v           //
  ) const noexcept;

  void d_integrate_transport_dq(             //
      mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                    //
      const_view_t<Nv> v                     //
  ) const noexcept;

  void d_integrate_transport_dv(             //
      mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                    //
      const_view_t<Nv> v                     //
  ) const noexcept;

  void d_integrate_transport_dq(                       //
      row_major_mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      row_major_const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                              //
      const_view_t<Nv> v                               //
  ) const noexcept;

  void d_integrate_transport_dv(                       //
      row_major_mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      row_major_const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                              //
      const_view_t<Nv> v                               //
  ) const noexcept;

  void difference(              //
      mut_view_t<Nv> out_v,     //
      const_view_t<Nq> q_start, //
      const_view_t<Nq> q_finish //
  ) const noexcept;

  void d_difference_dq_start(            //
      mut_view_t<Nv, Nv> out_v_dq_start, //
      const_view_t<Nq> q_start,          //
      const_view_t<Nq> q_finish          //
  ) const noexcept;

  void d_difference_dq_finish(            //
      mut_view_t<Nv, Nv> out_v_dq_finish, //
      const_view_t<Nq> q_start,           //
      const_view_t<Nq> q_finish           //
  ) const noexcept;

  void dynamics_aba(                   //
      mut_view_t<Nv> out_acceleration, //
      const_view_t<Nq> q,              //
      const_view_t<Nv> v,              //
      const_view_t<Nv> tau             //
  ) const noexcept;

  void d_dynamics_aba(                          //
      mut_view_t<Nv, Nv> out_acceleration_dq,   //
      mut_view_t<Nv, Nv> out_acceleration_dv,   //
      mut_view_t<Nv, Nv> out_acceleration_dtau, //
      const_view_t<Nq> q,                       //
      const_view_t<Nv> v,                       //
      const_view_t<Nv> tau                      //
  ) const noexcept;

  void inverse_dynamics_rnea( //
      mut_view_t<Nv> out_tau, //
      const_view_t<Nq> q,     //
      const_view_t<Nv> v,     //
      const_view_t<Nv> a      //
  ) const noexcept;

  auto _neutral_configuration() const -> Eigen::Matrix<scalar_t, Nq, 1> {
    Eigen::Matrix<scalar_t, Nq, 1> out{m_config_dim};
    neutral_configuration(eigen::as_mut_view(out));
    return out;
  }
  auto _random_configuration() const -> Eigen::Matrix<scalar_t, Nq, 1> {
    Eigen::Matrix<scalar_t, Nq, 1> out{m_config_dim};
    random_configuration(eigen::as_mut_view(out));
    return out;
  }

  template <typename Q, typename V>
  auto _integrate(Q const& q, V const& v) const -> Eigen::Matrix<scalar_t, Nq, 1> {
    Eigen::Matrix<scalar_t, Nq, 1> out{m_config_dim};
    integrate(eigen::as_mut_view(out), eigen::as_const_view(q.eval()), eigen::as_const_view(v.eval()));
    return out;
  }

  template <typename Q, typename V>
  auto _d_integrate_dq(Q const& q, V const& v) const -> Eigen::Matrix<scalar_t, Nv, Nv> {
    Eigen::Matrix<scalar_t, Nv, Nv> out{m_tangent_dim, m_tangent_dim};
    integrate_dq(eigen::as_mut_view(out), eigen::as_const_view(q.eval()), eigen::as_const_view(v.eval()));
    return out;
  }

  template <typename Q, typename V>
  auto _d_integrate_dv(Q const& q, V const& v) const -> Eigen::Matrix<scalar_t, Nv, Nv> {
    Eigen::Matrix<scalar_t, Nv, Nv> out{m_tangent_dim, m_tangent_dim};
    integrate_dv(eigen::as_mut_view(out), eigen::as_const_view(q.eval()), eigen::as_const_view(v.eval()));
    return out;
  }

  template <typename J_In, typename Q, typename V>
  auto _d_integrate_transport_dq(J_In const& in_J, Q const& q, V const& v) const -> Eigen::Matrix<
      scalar_t,
      J_In::RowsAtCompileTime,
      Nv,
      J_In::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,
      J_In::MaxRowsAtCompileTime> {
    Eigen::Matrix<
        scalar_t,
        J_In::RowsAtCompileTime,
        Nv,
        J_In::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,
        J_In::MaxRowsAtCompileTime>
        out{in_J.rows(), in_J.cols()};
    d_integrate_transport_dq(
        eigen::dyn_rows(eigen::as_mut_view(out)),
        eigen::dyn_rows(eigen::as_const_view(in_J)),
        eigen::as_const_view(q.eval()),
        eigen::as_const_view(v.eval()));
    return out;
  }

  template <typename J_In, typename Q, typename V>
  auto _d_integrate_transport_dv(J_In const& in_J, Q const& q, V const& v) const -> Eigen::Matrix<
      scalar_t,
      J_In::RowsAtCompileTime,
      Nv,
      J_In::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,
      J_In::MaxRowsAtCompileTime> {
    Eigen::Matrix<
        scalar_t,
        J_In::RowsAtCompileTime,
        Nv,
        J_In::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor,
        J_In::MaxRowsAtCompileTime>
        out{in_J.rows(), in_J.cols()};
    d_integrate_transport_dv(
        eigen::dyn_rows(eigen::as_mut_view(out)),
        eigen::dyn_rows(eigen::as_const_view(in_J)),
        eigen::as_const_view(q.eval()),
        eigen::as_const_view(v.eval()));
    return out;
  }

  template <typename Q_start, typename Q_finish>
  auto _difference(Q_start const& q_start, Q_finish const& q_finish) const -> Eigen::Matrix<scalar_t, Nv, 1> {

    Eigen::Matrix<scalar_t, Nv, 1> out{m_tangent_dim};
    difference(eigen::as_mut_view(out), eigen::as_const_view(q_start.eval()), eigen::as_const_view(q_finish.eval()));
    return out;
  }

  template <typename Q_start, typename Q_finish>
  auto _d_difference_dq_start(Q_start const& q_start, Q_finish const& q_finish) const
      -> Eigen::Matrix<scalar_t, Nv, Nv> {
    Eigen::Matrix<scalar_t, Nv, Nv> out{m_tangent_dim, m_tangent_dim};
    d_difference_dq_start(
        eigen::as_mut_view(out),
        eigen::as_const_view(q_start.eval()),
        eigen::as_const_view(q_finish.eval()));
    return out;
  }

  template <typename Q_start, typename Q_finish>
  auto _d_difference_dq_finish(Q_start const& q_start, Q_finish const& q_finish) const
      -> Eigen::Matrix<scalar_t, Nv, Nv> {
    Eigen::Matrix<scalar_t, Nv, Nv> out{m_tangent_dim, m_tangent_dim};
    d_difference_dq_finish(
        eigen::as_mut_view(out),
        eigen::as_const_view(q_start.eval()),
        eigen::as_const_view(q_finish.eval()));
    return out;
  }
};

} // namespace pinocchio
} // namespace ddp

#endif /* end of include guard PINOCCHIO_INTERFACE_HPP_EVURJTS1 */
