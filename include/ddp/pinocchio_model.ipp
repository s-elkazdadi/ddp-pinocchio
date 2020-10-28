#ifndef PINOCCHIO_MODEL_IPP_2I6Y8FFV
#define PINOCCHIO_MODEL_IPP_2I6Y8FFV

#include "ddp/pinocchio_model.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <boost/align/aligned_alloc.hpp>

#include <new>
#include <stdexcept>
#include <atomic>
#include <fmt/ostream.h>

#if __cplusplus >= 201703L
#define DDP_LAUNDER(...) ::std::launder(__VA_ARGS__)
#else
#define DDP_LAUNDER(...) (__VA_ARGS__)
#endif

#if !defined(DDP_PINOCCHIO_GENERAL) && !defined(DDP_PINOCCHIO_ABA) && !defined(DDP_PINOCCHIO_FRAMES)
#define DDP_PINOCCHIO_GENERAL
#define DDP_PINOCCHIO_ABA
#define DDP_PINOCCHIO_FRAMES
#endif

namespace ddp {
namespace pinocchio {

template <typename T, index_t Nq, index_t Nv>
struct model_t<T, Nq, Nv>::impl_model_t {
  ::pinocchio::ModelTpl<scalar_t> m_impl;
};
template <typename T, index_t Nq, index_t Nv>
struct model_t<T, Nq, Nv>::impl_data_t {
  ::pinocchio::DataTpl<scalar_t> m_impl;
  std::atomic<bool> m_available;
};

} // namespace pinocchio
} // namespace ddp

#ifdef DDP_PINOCCHIO_GENERAL

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
}
} // namespace pinocchio

namespace ddp {
namespace pinocchio {

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::key::destroy() {
  if (m_parent != nullptr) {
    m_parent->m_available = true;
    m_parent = nullptr;
  }
}

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
template <typename Model_Builder>
model_t<T, Nq, Nv>::model_t(Model_Builder&& model_builder, index_t n_parallel) {

  DDP_ASSERT_MSG_ALL_OF( //
      ("invalid parallelization parameter", n_parallel > 0),
      ("invalid parallelization parameter", n_parallel < 256));

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
    new (m_data + i) impl_data_t{::pinocchio::DataTpl<scalar_t>{m_model->m_impl}, {true}};
  }

  m_config_dim = m_model->m_impl.nq;
  m_tangent_dim = m_model->m_impl.nv;
}

template <typename T, index_t Nq, index_t Nv>
model_t<T, Nq, Nv>::model_t(fmt::string_view urdf_path, index_t n_parallel)
    : model_t{detail::builder_from_urdf_t{urdf_path}, n_parallel} {}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::all_joints_test_model(index_t n_parallel) -> model_t {
  return model_t{::pinocchio::buildAllJointsModel, n_parallel};
}

template <typename T, index_t Nq, index_t Nv>
model_t<T, Nq, Nv>::~model_t() {
  DDP_ASSERT(
      "model and data should either be both valid or both invalid (moved-from state)",
      (m_model == nullptr) == (m_data == nullptr));

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
model_t<T, Nq, Nv>::model_t(model_t&& other) {
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
auto model_t<T, Nq, Nv>::acquire_workspace() const -> key {
  for (index_t i = 0; i < m_num_data; ++i) {
    std::atomic<bool>& available = m_data[i].m_available;

    if (available) {
      bool expected = true;
      bool const changed = available.compare_exchange_strong(expected, false);
      if (changed) {
        return key{m_data[i]};
      }
    }
  }
  DDP_ASSERT_MSG("no workspace available", false);
  std::terminate();
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::model_name() const -> fmt::string_view {
  return m_model->m_impl.name;
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::neutral_configuration(mut_view_t<Nq> out_q) const {

  DDP_ASSERT_MSG("output configuration vector is not correctly sized", out_q.rows() == m_config_dim);
  ::pinocchio::neutral(m_model->m_impl, out_q);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::random_configuration(mut_view_t<Nq> out_q) const {

  DDP_ASSERT_MSG("output configuration vector is not correctly sized", out_q.rows() == m_config_dim);
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
) const {
  DDP_ASSERT_MSG_ALL_OF(
      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("tangent vector is not correctly sized", v.rows() == m_tangent_dim),
      ("output configuration vector is not correctly sized", out_q.rows() == m_config_dim),
      ("invalid data", not q.hasNaN()),
      ("invalid data", not v.hasNaN()));
  ;
  ::pinocchio::integrate(m_model->m_impl, q, v, out_q);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_dq(
    mut_view_t<Nv, Nv> out_q_dq, //
    const_view_t<Nq> q,          //
    const_view_t<Nv> v           //
) const {
  DDP_ASSERT_MSG_ALL_OF(
      ("output jacobian matrix has the wrong number of rows", out_q_dq.rows() == m_tangent_dim),
      ("output jacobian matrix has the wrong number of columns", out_q_dq.cols() == m_tangent_dim),
      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("tangent vector is not correctly sized", v.rows() == m_tangent_dim),
      ("invalid data", not q.hasNaN()),
      ("invalid data", not v.hasNaN()));
  ::pinocchio::dIntegrate(m_model->m_impl, q, v, out_q_dq, ::pinocchio::ArgumentPosition::ARG0);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_integrate_dv(
    mut_view_t<Nv, Nv> out_q_dv, //
    const_view_t<Nq> q,          //
    const_view_t<Nv> v           //
) const {

  DDP_ASSERT_MSG_ALL_OF(
      ("output jacobian has the wrong number of rows", out_q_dv.rows() == m_tangent_dim),
      ("output jacobian has the wrong number of columns", out_q_dv.cols() == m_tangent_dim),
      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("tangent vector is not correctly sized", v.rows() == m_tangent_dim),
      ("invalid data", not q.hasNaN()),
      ("invalid data", not v.hasNaN()));
  ::pinocchio::dIntegrate(m_model->m_impl, q, v, out_q_dv, ::pinocchio::ArgumentPosition::ARG1);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::difference(
    mut_view_t<Nv> out_v,     //
    const_view_t<Nq> q_start, //
    const_view_t<Nq> q_finish //
) const {

  DDP_ASSERT_MSG_ALL_OF(
      ("output tangent vector is not correctly sized", out_v.rows() == m_tangent_dim),
      ("starting configuration vector is not correctly sized", q_start.rows() == m_config_dim),
      ("finish configuration vector is not correctly sized", q_finish.rows() == m_config_dim),
      ("invalid data", not q_start.hasNaN()),
      ("invalid data", not q_finish.hasNaN()));

  ::pinocchio::difference(m_model->m_impl, q_start, q_finish, out_v);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_difference_dq_start(
    mut_view_t<Nv, Nv> out_v_dq_start, //
    const_view_t<Nq> q_start,          //
    const_view_t<Nq> q_finish          //
) const {

  DDP_ASSERT_MSG_ALL_OF(
      ("output jacobian has the wrong number of rows", out_v_dq_start.rows() == m_tangent_dim),
      ("output jacobian has the wrong number of columns", out_v_dq_start.cols() == m_tangent_dim),
      ("starting configuration vector is not correctly sized", q_start.rows() == m_config_dim),
      ("finish configuration vector is not correctly sized", q_finish.rows() == m_config_dim),
      ("invalid data", not q_start.hasNaN()),
      ("invalid data", not q_finish.hasNaN()));

  ::pinocchio::dDifference(m_model->m_impl, q_start, q_finish, out_v_dq_start, ::pinocchio::ArgumentPosition::ARG0);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::d_difference_dq_finish(
    mut_view_t<Nv, Nv> out_v_dq_finish, //
    const_view_t<Nq> q_start,           //
    const_view_t<Nq> q_finish           //
) const {

  DDP_ASSERT_MSG_ALL_OF(
      ("output jacobian has the wrong number of rows", out_v_dq_finish.rows() == m_tangent_dim),
      ("output jacobian has the wrong number of columns", out_v_dq_finish.cols() == m_tangent_dim),
      ("starting configuration vector is not correctly sized", q_start.rows() == m_config_dim),
      ("finish configuration vector is not correctly sized", q_finish.rows() == m_config_dim),
      ("invalid data", not q_start.hasNaN()),
      ("invalid data", not q_finish.hasNaN()));

  ::pinocchio::dDifference(m_model->m_impl, q_start, q_finish, out_v_dq_finish, ::pinocchio::ArgumentPosition::ARG1);
}

} // namespace pinocchio
} // namespace ddp

#endif

#ifdef DDP_PINOCCHIO_ABA

#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>

namespace ddp {
namespace pinocchio {

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::dynamics_aba( //
    mut_view_t<Nv> out_acceleration,   //
    const_view_t<Nq> q,                //
    const_view_t<Nv> v,                //
    const_view_t<Nv> tau,              //
    key k                              //
) const -> key {

  DDP_ASSERT_MSG_ALL_OF(
      ("output acceleration vector is not correctly sized", out_acceleration.rows() == m_tangent_dim),
      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("velocity vector is not correctly sized", v.rows() == m_tangent_dim),
      ("control vector is not correctly sized", tau.rows() == m_tangent_dim),
      ("invalid data", not q.hasNaN()),
      ("invalid data", not v.hasNaN()),
      ("invalid data", not tau.hasNaN()),
      ("invalid workspace key", static_cast<void*>(k.m_parent) != nullptr));

  impl_data_t* data = k.m_parent;
  ::pinocchio::aba(m_model->m_impl, data->m_impl, q, v, tau);
  out_acceleration = data->m_impl.ddq;

  DDP_ASSERT_MSG(
      fmt::format(
          "invalid output acceleration:\n"
          "{}\n"
          "inputs are\n"
          "q  : {}\n"
          "v  : {}\n",
          "tau: {}\n",
          out_acceleration.transpose(),
          q.transpose(),
          v.transpose(),
          tau.transpose()),
      not out_acceleration.hasNaN());

  return k;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::d_dynamics_aba(      //
    mut_view_t<Nv, Nv> out_acceleration_dq,   //
    mut_view_t<Nv, Nv> out_acceleration_dv,   //
    mut_view_t<Nv, Nv> out_acceleration_dtau, //
    const_view_t<Nq> q,                       //
    const_view_t<Nv> v,                       //
    const_view_t<Nv> tau,                     //
    key k                                     //
) const -> key {

  DDP_ASSERT_MSG_ALL_OF(
      ("output acceleration jacobian with respect to the velocity has the wrong number of rows",
       out_acceleration_dv.rows() == m_tangent_dim),
      ("output acceleration jacobian with respect to the configuration has the wrong number of rows",
       out_acceleration_dq.rows() == m_tangent_dim),
      ("output acceleration jacobian with respect to the configuration has the wrong number of columns",
       out_acceleration_dq.cols() == m_tangent_dim),
      ("output acceleration jacobian with respect to the velocity has the wrong number of columns",
       out_acceleration_dv.cols() == m_tangent_dim),
      ("output acceleration jacobian with respect to the control has the wrong number of rows",
       out_acceleration_dtau.rows() == m_tangent_dim),
      ("output acceleration jacobian with respect to the control has the wrong number of columns",
       eigen::as_const_view(out_acceleration_dtau).cols() == m_tangent_dim),

      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("velocity vector is not correctly sized", v.rows() == m_tangent_dim),
      ("control vector is not correctly sized", tau.rows() == m_tangent_dim),

      ("invalid data", not q.hasNaN()),
      ("invalid data", not v.hasNaN()),
      ("invalid data", not tau.hasNaN()),

      ("invalid workspace key", static_cast<void*>(k.m_parent) != nullptr));

  impl_data_t* data = k.m_parent;
  ::pinocchio::computeABADerivatives(
      m_model->m_impl,
      data->m_impl,
      q,
      v,
      tau,
      out_acceleration_dq,
      out_acceleration_dv,
      out_acceleration_dtau);

  DDP_ASSERT_MSG_ALL_OF(
      (fmt::format(
           "invalid output derivative:\n"
           "{}\n"
           "inputs are\n"
           "q  : {}\n"
           "v  : {}\n",
           "tau: {}\n",
           out_acceleration_dq.transpose(),
           q.transpose(),
           v.transpose(),
           tau.transpose()),
       not out_acceleration_dq.hasNaN()),
      (fmt::format(
           "invalid output derivative:\n"
           "{}\n"
           "inputs are\n"
           "q  : {}\n"
           "v  : {}\n",
           "tau: {}\n",
           out_acceleration_dv.transpose(),
           q.transpose(),
           v.transpose(),
           tau.transpose()),
       not out_acceleration_dv.hasNaN()),
      (fmt::format(
           "invalid output derivative:\n"
           "{}\n"
           "inputs are\n"
           "q  : {}\n"
           "v  : {}\n",
           "tau: {}\n",
           out_acceleration_dtau.transpose(),
           q.transpose(),
           v.transpose(),
           tau.transpose()),
       not out_acceleration_dtau.hasNaN()));

  return k;
}

} // namespace pinocchio
} // namespace ddp

#endif

#ifdef DDP_PINOCCHIO_FRAMES

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace ddp {
namespace pinocchio {

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::frame_coordinates_precompute(const_view_t<Nq> q, key k) const -> key {
  DDP_ASSERT_MSG_ALL_OF( //
      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("invalid data", not q.hasNaN()),
      ("invalid workspace key", static_cast<void*>(k.m_parent) != nullptr));

  impl_data_t* data = k.m_parent;
  ::pinocchio::framesForwardKinematics(m_model->m_impl, data->m_impl, q);
  return k;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::frame_coordinates(mut_view_t<3> out, index_t i, key k) const -> key {
  DDP_ASSERT_MSG_ALL_OF( //
      ("frame index must be in bounds", i >= 0),
      ("frame index must be in bounds", i < m_model->m_impl.nframes),
      ("invalid workspace key", static_cast<void*>(k.m_parent) != nullptr));

  impl_data_t* data = k.m_parent;
  out = data->m_impl.oMf[static_cast<std::size_t>(i)].translation();
  return k;
};

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::dframe_coordinates_precompute(const_view_t<Nq> q, key k) const -> key {
  DDP_ASSERT_MSG_ALL_OF( //
      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("invalid data", not q.hasNaN()),
      ("invalid workspace key", static_cast<void*>(k.m_parent) != nullptr));

  impl_data_t* data = k.m_parent;
  ::pinocchio::computeJointJacobians(m_model->m_impl, data->m_impl, q);
  ::pinocchio::framesForwardKinematics(m_model->m_impl, data->m_impl, q);
  return k;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::d_frame_coordinates(mut_view_t<3, Nv> out, index_t i, key k) const -> key {
  DDP_ASSERT_MSG_ALL_OF( //
      ("frame index must be in bounds", i >= 0),
      ("frame index must be in bounds", i < m_model->m_impl.nframes),

      ("output constraint jacobian matrix is not correctly sized", out.cols() == m_tangent_dim),
      ("invalid workspace key", static_cast<void*>(k.m_parent) != nullptr));

  impl_data_t* data = k.m_parent;
  out.setZero();

  auto nv = tangent_dim_c();
  thread_local auto workspace = eigen::make_matrix<scalar_t>(fix_index<6>{}, nv);

  if (nv.value() > workspace.cols()) {
    workspace.resize(6, nv.value());
  } else if (nv.value() != workspace.cols()) {
  }
  workspace.setZero();

  Eigen::Map<decltype(workspace)> w{workspace.data(), 6, nv.value()};

  ::pinocchio::getFrameJacobian(
      m_model->m_impl,
      data->m_impl,
      static_cast<std::size_t>(i),
      ::pinocchio::LOCAL_WORLD_ALIGNED,
      w);
  out = w.template topRows<3>();
  return k;
};

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::n_frames() const -> index_t {
  return m_model->m_impl.nframes;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::frame_name(index_t i) const -> fmt::string_view {
  DDP_ASSERT_MSG_ALL_OF( //
      ("frame index must be in bounds", i >= 0),
      ("frame index must be in bounds", i < m_model->m_impl.nframes));

  return m_model->m_impl.frames[static_cast<std::size_t>(i)].name;
}

} // namespace pinocchio
} // namespace ddp

#endif
#endif /* end of include guard PINOCCHIO_MODEL_IPP_2I6Y8FFV */
