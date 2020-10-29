#ifndef PINOCCHIO_INTERFACE_HPP_EVURJTS1
#define PINOCCHIO_INTERFACE_HPP_EVURJTS1

#include "ddp/indexer.hpp"
#include "ddp/detail/eigen.hpp"
#include <memory>

namespace ddp {
namespace pinocchio {

namespace gsl {
template <typename T>
using owner = T;
} // namespace gsl

template <typename T, index_t Nq = Eigen::Dynamic, index_t Nv = Eigen::Dynamic>
struct model_t {
  using scalar_t = T;
  static constexpr bool fixed_size = Nq != Eigen::Dynamic and Nv != Eigen::Dynamic;

private:
  struct impl_model_t;
  struct impl_data_t;

  gsl::owner<impl_model_t*> m_model = nullptr;
  gsl::owner<impl_data_t*> m_data = nullptr;
  index_t m_num_data = 0;
  index_t m_config_dim = 0;
  index_t m_tangent_dim = 0;

  template <index_t Rows, index_t Cols = 1>
  using mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols>>;

  template <index_t Rows, index_t Cols = 1>
  using const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols> const>;

  template <index_t Rows, index_t Cols = 1>
  using row_major_mut_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols, Eigen::RowMajor>>;
  template <index_t Rows, index_t Cols = 1>
  using row_major_const_view_t = eigen::view_t<Eigen::Matrix<scalar_t, Rows, Cols, Eigen::RowMajor> const>;

  template <typename Model_Builder>
  model_t(Model_Builder&& model_builder, index_t n_parallel);

  void validate_config_vector(
      const_view_t<Nq> q,   //
      fmt::string_view name //
  ) const;
  void validate_tangent_vector(
      const_view_t<Nv> v,   //
      fmt::string_view name //
  ) const;
  void validate_tangent_matrix(
      const_view_t<Nv, Nv> m, //
      fmt::string_view name   //
  ) const;
  template <typename Out, typename In>
  void validate_similar_jac_matrices(
      Out m1,                 //
      In m2,                  //
      fmt::string_view name1, //
      fmt::string_view name2  //
  ) const;
  template <typename Out, typename In>
  void d_integrate_transport_dq_or_dv(Out out_J, In in_J, const_view_t<Nq> q, const_view_t<Nv> v, bool dq) const;

public:
  struct key {
    key() = default;

    key(key const&) = delete;
    key(key&& other) : m_parent{other.m_parent} { other.m_parent = nullptr; }
    auto operator=(key const&) -> key& = delete;
    auto operator=(key&& other) -> key& {
      if (this != &other) {
        destroy();
        m_parent = other.m_parent;
        other.m_parent = nullptr;
      }
      return *this;
    }
    ~key() { destroy(); }

    explicit operator bool() { return m_parent != nullptr; }

  private:
    void destroy();
    friend struct model_t;
    explicit key(impl_data_t& ref) : m_parent{&ref} {};
    impl_data_t* m_parent = nullptr;
  };

  ~model_t();
  model_t(model_t const&) = delete;
  model_t(model_t&&);

  auto operator=(model_t const&) -> model_t& = delete;
  auto operator=(model_t&&) -> model_t& = delete;

  auto model_name() const -> fmt::string_view;

  explicit model_t(fmt::string_view urdf_path, index_t n_parallel = 1);
  static auto all_joints_test_model(index_t n_parallel = 1) -> model_t;

  auto configuration_dim() const -> index_t { return m_config_dim; }
  auto tangent_dim() const -> index_t { return m_tangent_dim; }

  auto configuration_dim_c() const -> DDP_CONDITIONAL(fixed_size, fix_index<Nq>, dyn_index) {
    return DDP_CONDITIONAL(fixed_size, fix_index<Nq>, dyn_index){m_config_dim};
  }
  auto tangent_dim_c() const -> DDP_CONDITIONAL(fixed_size, fix_index<Nv>, dyn_index) {
    return DDP_CONDITIONAL(fixed_size, fix_index<Nv>, dyn_index){m_tangent_dim};
  }

  auto acquire_workspace() const -> key;

  void neutral_configuration(mut_view_t<Nq> out_q) const;
  void random_configuration(mut_view_t<Nq> out_q) const;

  void integrate(           //
      mut_view_t<Nq> out_q, //
      const_view_t<Nq> q,   //
      const_view_t<Nv> v    //
  ) const;

  void d_integrate_dq(             //
      mut_view_t<Nv, Nv> out_q_dq, //
      const_view_t<Nq> q,          //
      const_view_t<Nv> v           //
  ) const;

  void d_integrate_dv(             //
      mut_view_t<Nv, Nv> out_q_dv, //
      const_view_t<Nq> q,          //
      const_view_t<Nv> v           //
  ) const;

  void d_integrate_transport_dq(             //
      mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                    //
      const_view_t<Nv> v                     //
  ) const;

  void d_integrate_transport_dv(             //
      mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                    //
      const_view_t<Nv> v                     //
  ) const;

  void d_integrate_transport_dq(                       //
      row_major_mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      row_major_const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                              //
      const_view_t<Nv> v                               //
  ) const;

  void d_integrate_transport_dv(                       //
      row_major_mut_view_t<Eigen::Dynamic, Nv> out_J,  //
      row_major_const_view_t<Eigen::Dynamic, Nv> in_J, //
      const_view_t<Nq> q,                              //
      const_view_t<Nv> v                               //
  ) const;

  void difference(              //
      mut_view_t<Nv> out_v,     //
      const_view_t<Nq> q_start, //
      const_view_t<Nq> q_finish //
  ) const;

  void d_difference_dq_start(            //
      mut_view_t<Nv, Nv> out_v_dq_start, //
      const_view_t<Nq> q_start,          //
      const_view_t<Nq> q_finish          //
  ) const;

  void d_difference_dq_finish(            //
      mut_view_t<Nv, Nv> out_v_dq_finish, //
      const_view_t<Nq> q_start,           //
      const_view_t<Nq> q_finish           //
  ) const;

  auto dynamics_aba(                   //
      mut_view_t<Nv> out_acceleration, //
      const_view_t<Nq> q,              //
      const_view_t<Nv> v,              //
      const_view_t<Nv> tau,            //
      key k                            //
  ) const -> key;

  auto d_dynamics_aba(                          //
      mut_view_t<Nv, Nv> out_acceleration_dq,   //
      mut_view_t<Nv, Nv> out_acceleration_dv,   //
      mut_view_t<Nv, Nv> out_acceleration_dtau, //
      const_view_t<Nq> q,                       //
      const_view_t<Nv> v,                       //
      const_view_t<Nv> tau,                     //
      key k                                     //
  ) const -> key;

  auto frame_coordinates_precompute(const_view_t<Nq> q, key k) const -> key;
  auto frame_coordinates(mut_view_t<3> out, index_t i, key k) const -> key;
  auto dframe_coordinates_precompute(const_view_t<Nq> q, key k) const -> key;
  auto d_frame_coordinates(mut_view_t<3, Nv> out, index_t i, key k) const -> key;
  auto n_frames() const -> index_t;
  auto frame_name(index_t i) const -> fmt::string_view;

  void inverse_dynamics_rnea( //
      mut_view_t<Nv> out_tau, //
      const_view_t<Nq> q,     //
      const_view_t<Nv> v,     //
      const_view_t<Nv> a      //
  ) const;
};

} // namespace pinocchio
} // namespace ddp

#endif /* end of include guard PINOCCHIO_INTERFACE_HPP_EVURJTS1 */
