#ifndef PINOCCHIO_MODEL_IPP_2I6Y8FFV
#define PINOCCHIO_MODEL_IPP_2I6Y8FFV

#include "ddp/pinocchio_model.hpp"

#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <boost/align/aligned_alloc.hpp>
#include <boost/filesystem.hpp>

#include <new>
#include <stdexcept>

#if __cplusplus >= 201703L
#define DDP_LAUNDER(...) ::std::launder(__VA_ARGS__)
#else
#define DDP_LAUNDER(...) (__VA_ARGS__)
#endif

namespace pinocchio {
template <typename D>
void addJointAndBody(
    Model& model,
    const JointModelBase<D>& jmodel,
    const Model::JointIndex parent_id,
    const SE3& joint_placement,
    const std::string& name,
    const Inertia& Y) {
  using TV = typename D::TangentVector_t;
  using CV = typename D::ConfigVector_t;
  Model::JointIndex idx = model.addJoint(
      parent_id,
      jmodel,
      joint_placement,
      name + "_joint",
      TV::Zero(),
      1e3 * (TV::Random() + TV::Constant(1)),
      1e3 * (CV::Random() - CV::Constant(1)),
      1e3 * (CV::Random() + CV::Constant(1)));

  model.appendBodyToJoint(idx, Y, SE3::Identity());
}

void buildAllJointsModel(Model& model) {
  addJointAndBody(
      model,
      JointModelFreeFlyer(),
      model.getJointId("universe"),
      SE3::Identity(),
      "freeflyer",
      Inertia::Random());
  addJointAndBody(
      model,
      JointModelSpherical(),
      model.getJointId("freeflyer_joint"),
      SE3::Identity(),
      "spherical",
      Inertia::Random());
  addJointAndBody(
      model,
      JointModelPlanar(),
      model.getJointId("spherical_joint"),
      SE3::Identity(),
      "planar",
      Inertia::Random());
  addJointAndBody(model, JointModelRX(), model.getJointId("planar_joint"), SE3::Identity(), "rx", Inertia::Random());
  addJointAndBody(model, JointModelPX(), model.getJointId("rx_joint"), SE3::Identity(), "px", Inertia::Random());
  addJointAndBody(
      model,
      JointModelPrismaticUnaligned(SE3::Vector3(1, 0, 0)),
      model.getJointId("px_joint"),
      SE3::Identity(),
      "pu",
      Inertia::Random());
  addJointAndBody(
      model,
      JointModelRevoluteUnaligned(SE3::Vector3(0, 0, 1)),
      model.getJointId("pu_joint"),
      SE3::Identity(),
      "ru",
      Inertia::Random());
  addJointAndBody(
      model,
      JointModelSphericalZYX(),
      model.getJointId("ru_joint"),
      SE3::Identity(),
      "sphericalZYX",
      Inertia::Random());
  addJointAndBody(
      model,
      JointModelTranslation(),
      model.getJointId("sphericalZYX_joint"),
      SE3::Identity(),
      "translation",
      Inertia::Random());
}
} // namespace pinocchio

namespace ddp {
namespace pinocchio {

namespace detail {

struct builder_from_urdf_t {
  fmt::string_view urdf_path;
  void operator()(::pinocchio::Model& model) const {
    std::string path;
    if (urdf_path[0] == '~') {
      fmt::string_view home_dir = std::getenv("HOME"); // TODO: windows?
      fmt::string_view tail{urdf_path.begin() + 1, urdf_path.size() - 1};
      path.reserve(home_dir.size() + tail.size());
      path.append(home_dir.data(), home_dir.size());
      path.append(tail.data(), tail.size());
    } else {
      path = std::string{urdf_path.begin(), urdf_path.end()};
    }
    ::pinocchio::urdf::buildModel(path, model);
  }
};

} // namespace detail

template <typename T, index_t Nq, index_t Nv>
struct model_t<T, Nq, Nv>::impl_model_t {
  ::pinocchio::ModelTpl<scalar_t> m_impl;
};
template <typename T, index_t Nq, index_t Nv>
struct model_t<T, Nq, Nv>::impl_data_t {
  ::pinocchio::DataTpl<scalar_t> m_impl;
};

template <typename T, index_t Nq, index_t Nv>
template <typename Model_Builder>
model_t<T, Nq, Nv>::model_t(Model_Builder&& model_builder, index_t n_parallel) noexcept(false) {

  assert(n_parallel == 1); // TODO: multi-threading
  assert(n_parallel > 0);
  assert(n_parallel < 256);

  m_num_data = n_parallel;
  m_model = static_cast<gsl::owner<impl_model_t*>>(boost::alignment::aligned_alloc( //
      alignof(impl_model_t),                                                        //
      sizeof(impl_model_t))                                                         //
  );
  if (m_model == nullptr) {
    throw std::bad_alloc();
  }
  m_data = DDP_LAUNDER(static_cast<gsl::owner<impl_data_t*>>(boost::alignment::aligned_alloc( //
      alignof(impl_data_t),                                                                   //
      static_cast<std::size_t>(m_num_data) * sizeof(impl_data_t))                             //
                                                             ));
  if (m_data == nullptr) {
    boost::alignment::aligned_free(m_model);
    throw std::bad_alloc();
  }

  ::pinocchio::Model model_double{};
  static_cast<Model_Builder&&>(model_builder)(model_double);
  m_model = new (m_model) impl_model_t{model_double.cast<scalar_t>()};

  for (index_t i = 0; i < m_num_data; ++i) {
    new (m_data + i) impl_data_t{::pinocchio::DataTpl<scalar_t>{m_model->m_impl}};
  }

  m_config_dim = m_model->m_impl.nq;
  m_tangent_dim = m_model->m_impl.nv;
}

template <typename T, index_t Nq, index_t Nv>
model_t<T, Nq, Nv>::model_t(fmt::string_view urdf_path, index_t n_parallel) noexcept(false)
    : model_t{detail::builder_from_urdf_t{urdf_path}, n_parallel} {}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::all_joints_test_model(index_t n_parallel) noexcept(false) -> model_t {
  return model_t{::pinocchio::buildAllJointsModel, n_parallel};
}

template <typename T, index_t Nq, index_t Nv>
model_t<T, Nq, Nv>::~model_t() noexcept {
  assert((m_model == nullptr) == (m_data == nullptr));

  if (m_data != nullptr) {
    for (index_t i = 0; i < m_num_data; ++i) {
      (m_data + i)->~impl_data_t();
    }
    boost::alignment::aligned_free(m_data);
  }

  if (m_model != nullptr) {
    m_model->~impl_model_t();
    boost::alignment::aligned_free(m_model);
  }
}

template <typename T, index_t Nq, index_t Nv>
model_t<T, Nq, Nv>::model_t(model_t&& other) noexcept {
  m_model = static_cast<gsl::owner<impl_model_t*>>(other.m_model);
  m_data = static_cast<gsl::owner<impl_data_t*>>(other.m_data);
  m_num_data = other.m_num_data;
  m_config_dim = other.m_config_dim;
  m_tangent_dim = other.m_tangent_dim;
  other.m_model = nullptr;
  other.m_data = nullptr;
  other.m_num_data = 0;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::get_data() const noexcept -> impl_data_t* {
  return static_cast<impl_data_t*>(m_data);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::validate_config_vector(
    const_view_t<Nq> q,   //
    fmt::string_view name //
) const noexcept {
  if (not fixed_size) {
    DDP_ASSERT(
        q.rows() == m_config_dim,
        fmt::format( //
            "{} has {} rows. (expected: {})\n",
            name,
            q.rows(),
            m_config_dim));
  }
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::validate_tangent_vector(
    const_view_t<Nv> v,   //
    fmt::string_view name //
) const noexcept {
  if (not fixed_size) {
    DDP_ASSERT(
        v.rows() == m_tangent_dim,
        fmt::format( //
            "{} has {} rows. (expected: {})\n",
            name,
            v.rows(),
            m_tangent_dim));
  }
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::validate_tangent_matrix(
    const_view_t<Nv, Nv> m, //
    fmt::string_view name   //
) const noexcept {
  if (not fixed_size) {
    DDP_ASSERT(
        m.rows() == m_tangent_dim and m.cols() == m_tangent_dim,
        (fmt::format( //
            "{} has shape [{}, {}]. (expected: [{}, {}])\n",
            name,
            m.rows(),
            m.cols(),
            m_tangent_dim,
            m_tangent_dim)));
  }
}

template <typename T, index_t Nq, index_t Nv>
template <typename Out, typename In>
void model_t<T, Nq, Nv>::validate_similar_jac_matrices(
    Out m1,                 //
    In m2,                  //
    fmt::string_view name1, //
    fmt::string_view name2  //
) const noexcept {
  if (not fixed_size) {
    DDP_ASSERT(                                    //
        m1.cols() == m_tangent_dim,                //
        fmt::format(                               //
            "{} has {} columns. (expected: {})\n", //
            name1,                                 //
            m1.cols(),                             //
            m_tangent_dim)                         //
    );

    DDP_ASSERT(                                    //
        m2.cols() == m_tangent_dim,                //
        fmt::format(                               //
            "{} has {} columns. (expected: {})\n", //
            name2,                                 //
            m2.cols(),                             //
            m_tangent_dim)                         //
    );

    DDP_ASSERT(                                                           //
        m1.rows() == m2.rows(),                                           //
        fmt::format(                                                      //
            "{} and {} have a mismatching number of rows. ({} and {})\n", //
            name1,                                                        //
            name2,                                                        //
            m1.rows(),                                                    //
            m2.rows())                                                    //
    );
  }
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::neutral_configuration(mut_view_t<Nq> out_q) const noexcept {
  validate_config_vector(eigen::as_const_view(out_q), "output configuration vector");
  ::pinocchio::neutral(m_model->m_impl, out_q);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::random_configuration(mut_view_t<Nq> out_q) const noexcept {
  validate_config_vector(eigen::as_const_view(out_q), "output configuration vector");
  ::pinocchio::randomConfiguration(       //
      m_model->m_impl,                    //
      m_model->m_impl.lowerPositionLimit, //
      m_model->m_impl.upperPositionLimit, //
      out_q                               //
  );
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::integrate(
    mut_view_t<Nq> out_q, //
    const_view_t<Nq> q,   //
    const_view_t<Nv> v    //
) const noexcept {
  validate_config_vector(eigen::as_const_view(out_q), "output configuration vector");
  validate_config_vector(q, "configuration vector");
  validate_tangent_vector(v, "tangent vector");
  ::pinocchio::integrate(m_model->m_impl, q, v, out_q);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_dq(
    mut_view_t<Nv, Nv> out_q_dq, //
    const_view_t<Nq> q,          //
    const_view_t<Nv> v           //
) const noexcept {
  validate_tangent_matrix(eigen::as_const_view(out_q_dq), "output jacobian matrix");
  validate_config_vector(q, "configuration vector");
  validate_tangent_vector(v, "tangent vector");
  ::pinocchio::dIntegrate(m_model->m_impl, q, v, out_q_dq, ::pinocchio::ArgumentPosition::ARG0);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_dv(
    mut_view_t<Nv, Nv> out_q_dv, //
    const_view_t<Nq> q,          //
    const_view_t<Nv> v           //
) const noexcept {
  validate_tangent_matrix(eigen::as_const_view(out_q_dv), "output jacobian");
  validate_config_vector(q, "configuration vector");
  validate_tangent_vector(v, "tangent vector");
  ::pinocchio::dIntegrate(m_model->m_impl, q, v, out_q_dv, ::pinocchio::ArgumentPosition::ARG1);
}

template <typename T, index_t Nq, index_t Nv>
template <typename Out, typename In>
void model_t<T, Nq, Nv>::d_integrate_transport_dq_or_dv(
    Out out_J,          //
    In in_J,            //
    const_view_t<Nq> q, //
    const_view_t<Nv> v, //
    bool dq             //
) const noexcept {
  validate_similar_jac_matrices(eigen::as_const_view(out_J), in_J, "output jacobian", "input jacobian");
  validate_config_vector(q, "configuration vector");
  validate_tangent_vector(v, "tangent vector");

  auto arg = dq ? ::pinocchio::ArgumentPosition::ARG0 : ::pinocchio::ArgumentPosition::ARG1;
  if (in_J.data() == out_J.data()) {
    assert(in_J.rows() == out_J.rows());
    assert(in_J.outerStride() == out_J.outerStride());
    ::pinocchio::dIntegrateTransport(m_model->m_impl, q, v, out_J.transpose(), arg);
  } else {
    ::pinocchio::dIntegrateTransport(m_model->m_impl, q, v, in_J.transpose(), out_J.transpose(), arg);
  }
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_transport_dq(
    mut_view_t<Eigen::Dynamic, Nv> out_J,  //
    const_view_t<Eigen::Dynamic, Nv> in_J, //
    const_view_t<Nq> q,                    //
    const_view_t<Nv> v                     //
) const noexcept {
  d_integrate_transport_dq_or_dv(out_J, in_J, q, v, true);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_transport_dv(
    mut_view_t<Eigen::Dynamic, Nv> out_J,  //
    const_view_t<Eigen::Dynamic, Nv> in_J, //
    const_view_t<Nq> q,                    //
    const_view_t<Nv> v                     //
) const noexcept {
  d_integrate_transport_dq_or_dv(out_J, in_J, q, v, false);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_transport_dq(
    row_major_mut_view_t<Eigen::Dynamic, Nv> out_J,  //
    row_major_const_view_t<Eigen::Dynamic, Nv> in_J, //
    const_view_t<Nq> q,                              //
    const_view_t<Nv> v                               //
) const noexcept {
  d_integrate_transport_dq_or_dv(out_J, in_J, q, v, true);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_transport_dv(
    row_major_mut_view_t<Eigen::Dynamic, Nv> out_J,  //
    row_major_const_view_t<Eigen::Dynamic, Nv> in_J, //
    const_view_t<Nq> q,                              //
    const_view_t<Nv> v                               //
) const noexcept {
  d_integrate_transport_dq_or_dv(out_J, in_J, q, v, false);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::difference(
    mut_view_t<Nv> out_v,     //
    const_view_t<Nq> q_start, //
    const_view_t<Nq> q_finish //
) const noexcept {
  validate_tangent_vector(eigen::as_const_view(out_v), "output tangent vector");
  validate_config_vector(q_start, "starting configuration vector");
  validate_config_vector(q_finish, "finish configuration vector");

  ::pinocchio::difference(m_model->m_impl, q_start, q_finish, out_v);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_difference_dq_start(
    mut_view_t<Nv, Nv> out_v_dq_start, //
    const_view_t<Nq> q_start,          //
    const_view_t<Nq> q_finish          //
) const noexcept {
  validate_tangent_matrix(eigen::as_const_view(out_v_dq_start), "output jacobian");
  validate_config_vector(q_start, "starting configuration vector");
  validate_config_vector(q_finish, "finish configuration vector");

  ::pinocchio::dDifference(m_model->m_impl, q_start, q_finish, out_v_dq_start, ::pinocchio::ArgumentPosition::ARG0);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_difference_dq_finish(
    mut_view_t<Nv, Nv> out_v_dq_finish, //
    const_view_t<Nq> q_start,           //
    const_view_t<Nq> q_finish           //
) const noexcept {
  validate_tangent_matrix(eigen::as_const_view(out_v_dq_finish), "output jacobian");
  validate_config_vector(q_start, "starting configuration vector");
  validate_config_vector(q_finish, "finish configuration vector");

  ::pinocchio::dDifference(m_model->m_impl, q_start, q_finish, out_v_dq_finish, ::pinocchio::ArgumentPosition::ARG1);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::dynamics_aba( //
    mut_view_t<Nv> out_acceleration,   //
    const_view_t<Nq> q,                //
    const_view_t<Nv> v,                //
    const_view_t<Nv> tau               //
) const noexcept {
  validate_tangent_vector(eigen::as_const_view(out_acceleration), "output acceleration vector");
  validate_config_vector(q, "configuration vector");
  validate_tangent_vector(v, "velocity vector");
  validate_tangent_vector(tau, "control vector");

  impl_data_t* data = this->get_data();
  ::pinocchio::aba(m_model->m_impl, data->m_impl, q, v, tau);
  out_acceleration = data->m_impl.ddq;
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_dynamics_aba(      //
    mut_view_t<Nv, Nv> out_acceleration_dq,   //
    mut_view_t<Nv, Nv> out_acceleration_dv,   //
    mut_view_t<Nv, Nv> out_acceleration_dtau, //
    const_view_t<Nq> q,                       //
    const_view_t<Nv> v,                       //
    const_view_t<Nv> tau                      //
) const noexcept {
  validate_tangent_matrix(
      eigen::as_const_view(out_acceleration_dq),
      "output acceleration jacobian with respect to the configuration");
  validate_tangent_matrix(
      eigen::as_const_view(out_acceleration_dv),
      "output acceleration jacobian with respect to the velocity");
  validate_tangent_matrix(
      eigen::as_const_view(out_acceleration_dtau),
      "output acceleration jacobian with respect to the control");
  validate_config_vector(q, "configuration vector");
  validate_tangent_vector(v, "velocity vector");
  validate_tangent_vector(tau, "control vector");

  impl_data_t* data = this->get_data();
  ::pinocchio::computeABADerivatives(
      m_model->m_impl,
      data->m_impl,
      q,
      v,
      tau,
      out_acceleration_dq,
      out_acceleration_dv,
      out_acceleration_dtau);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::inverse_dynamics_rnea( //
    mut_view_t<Nv> out_tau,                     //
    const_view_t<Nq> q,                         //
    const_view_t<Nv> v,                         //
    const_view_t<Nv> a                          //
) const noexcept {
  validate_tangent_vector(eigen::as_const_view(out_tau), "output control vector");
  validate_config_vector(q, "configuration vector");
  validate_tangent_vector(v, "velocity vector");
  validate_tangent_vector(a, "acceleration vector");

  impl_data_t* data = this->get_data();
  ::pinocchio::rnea(m_model->m_impl, data->m_impl, q, v, a);
  out_tau = data->m_impl.tau;
}

} // namespace pinocchio
} // namespace ddp
#endif /* end of include guard PINOCCHIO_MODEL_IPP_2I6Y8FFV */
