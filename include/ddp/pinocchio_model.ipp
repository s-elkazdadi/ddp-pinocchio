#ifndef PINOCCHIO_MODEL_IPP_2I6Y8FFV
#define PINOCCHIO_MODEL_IPP_2I6Y8FFV

#include "ddp/pinocchio_model.hpp"

#include <Eigen/src/Core/util/Constants.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <boost/align/aligned_alloc.hpp>

#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>

#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>

#include <new>
#include <stdexcept>
#include <atomic>
#include <fmt/ostream.h>

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

void buildAllJointsModel(Model& model, bool /*unused*/) {
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

namespace pin = ::pinocchio;

template <typename T, index_t Nq, index_t Nv>
struct model_t<T, Nq, Nv>::impl_model_t {
  pin::ModelTpl<scalar_t> m_impl;
};
template <typename T, index_t Nq, index_t Nv>
struct model_t<T, Nq, Nv>::impl_data_t {
  pin::DataTpl<scalar_t> m_impl;
  std::atomic<bool> m_available;
};

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::key::destroy() {
  if (m_parent != nullptr) {
    m_parent->m_available = true;
    m_parent = nullptr;
  }
}

namespace detail {

template <typename R, typename... Args>
struct function_ref_t {

  template <typename Fn>
  static auto from_fn(Fn&& fn) -> function_ref_t {

    using Fn_obj = typename std::remove_reference<Fn>::type;

    return function_ref_t{layout{
        any_ptr(&fn),

        +[](any_ptr ptr, Args... args) {
          auto* p = static_cast<Fn_obj*>(ptr);
          return static_cast<Fn&&>(*p)(static_cast<Args&&>(args)...);
        },
    }

    };
  }

  auto operator()(Args... args) -> R { return m_impl.m_call(m_impl.m_ptr, static_cast<Args&&>(args)...); }

private:
  using any_ptr = void*;

  struct layout {
    any_ptr m_ptr;
    R (*m_call)(any_ptr ptr, Args... args);
  } m_impl;

  explicit function_ref_t(layout l) : m_impl{l} {}
};

struct builder_from_urdf_t {
  fmt::string_view urdf_path;
  void operator()(pin::Model& model, bool add_freeflyer_base) const {
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

    if (add_freeflyer_base) {
      pin::urdf::buildModel(path, pin::JointModelFreeFlyer(), model);
    } else {
      pin::urdf::buildModel(path, model);
    }
  }
};

struct pinocchio_impl {
  template <typename T, index_t Nq, index_t Nv>
  static void from_model_geom_builder(
      model_t<T, Nq, Nv>& model,
      function_ref_t<void, pin::Model&, bool> builder,
      index_t n_parallel,
      bool add_freeflyer_base) {
    DDP_ASSERT_MSG_ALL_OF( //
        ("invalid parallelization parameter", n_parallel > 0),
        ("invalid parallelization parameter", n_parallel < 256));

    using impl_model_t = typename model_t<T, Nq, Nv>::impl_model_t;
    using impl_data_t = typename model_t<T, Nq, Nv>::impl_data_t;
    using scalar_t = typename model_t<T, Nq, Nv>::scalar_t;

    model.m_num_data = n_parallel;
    model.m_model = static_cast<gsl::owner<impl_model_t*>>(
        boost::alignment::aligned_alloc(alignof(impl_model_t), sizeof(impl_model_t)));

    if (model.m_model == nullptr) {
      throw std::bad_alloc();
    }
    model.m_data = DDP_LAUNDER(static_cast<gsl::owner<impl_data_t*>>(boost::alignment::aligned_alloc(
        alignof(impl_data_t),
        static_cast<std::size_t>(model.m_num_data) * sizeof(impl_data_t))));

    if (model.m_data == nullptr) {
      boost::alignment::aligned_free(model.m_model);
      throw std::bad_alloc();
    }

    pin::Model model_double{};
    builder(model_double, add_freeflyer_base);
    model.m_model = new (model.m_model) impl_model_t{model_double.cast<scalar_t>()};

    for (index_t i = 0; i < model.m_num_data; ++i) {
      new (model.m_data + i) impl_data_t{pin::DataTpl<scalar_t>{model.m_model->m_impl}, {true}};
    }

    model.m_config_dim = model.m_model->m_impl.nq;
    model.m_tangent_dim = model.m_model->m_impl.nv;
  }
};

} // namespace detail

template <typename T, index_t Nq, index_t Nv>
model_t<T, Nq, Nv>::model_t(fmt::string_view urdf_path, index_t n_parallel, bool add_freeflyer_base) {
  using fn_ref = detail::function_ref_t<void, pin::Model&, bool>;
  detail::pinocchio_impl::from_model_geom_builder(
      *this,
      fn_ref::from_fn(detail::builder_from_urdf_t{urdf_path}),
      n_parallel,
      add_freeflyer_base);
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::all_joints_test_model(index_t n_parallel) -> model_t {
  using fn_ref = detail::function_ref_t<void, pin::Model&, bool>;
  model_t m;
  detail::pinocchio_impl::from_model_geom_builder(m, fn_ref::from_fn(&pin::buildAllJointsModel), n_parallel, false);
  return m;
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
  pin::neutral(m_model->m_impl, out_q);
}

template <typename T, index_t Nq, index_t Nv>
void model_t<T, Nq, Nv>::random_configuration(mut_view_t<Nq> out_q) const {

  DDP_ASSERT_MSG("output configuration vector is not correctly sized", out_q.rows() == m_config_dim);
  pin::randomConfiguration(               //
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
  pin::integrate(m_model->m_impl, q, v, out_q);
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
  pin::dIntegrate(m_model->m_impl, q, v, out_q_dq, pin::ArgumentPosition::ARG0);
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
  pin::dIntegrate(m_model->m_impl, q, v, out_q_dv, pin::ArgumentPosition::ARG1);
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

  pin::difference(m_model->m_impl, q_start, q_finish, out_v);
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

  pin::dDifference(m_model->m_impl, q_start, q_finish, out_v_dq_start, pin::ArgumentPosition::ARG0);
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

  pin::dDifference(m_model->m_impl, q_start, q_finish, out_v_dq_finish, pin::ArgumentPosition::ARG1);
}

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
  pin::aba(m_model->m_impl, data->m_impl, q, v, tau);
  out_acceleration = data->m_impl.ddq;

  DDP_ASSERT_MSG(
      fmt::format(
          "invalid output acceleration:\n"
          "{}\n"
          "inputs are\n"
          "q  : {}\n"
          "v  : {}\n"
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
      ("output acceleration jacobian with respect to the configuration has the wrong number of rows",
       out_acceleration_dq.rows() == m_tangent_dim),
      ("output acceleration jacobian with respect to the velocity has the wrong number of rows",
       out_acceleration_dv.rows() == m_tangent_dim),
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
  pin::computeABADerivatives(
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
           "v  : {}\n"
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
           "v  : {}\n"
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
           "v  : {}\n"
           "tau: {}\n",
           out_acceleration_dtau.transpose(),
           q.transpose(),
           v.transpose(),
           tau.transpose()),
       not out_acceleration_dtau.hasNaN()));

  return k;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::tau_sol(  //
    const_view_t<Nq> q,            //
    const_view_t<Nv> v,            //
    index_t const frame_indices[], //
    index_t n_frames               //
) const -> Eigen::Matrix<scalar_t, -1, 1> {

  auto k = acquire_workspace();
  auto& model = m_model->m_impl;
  auto& data = k.m_parent->m_impl;

  auto nv = tangent_dim_c();

  pin::forwardKinematics(model, data, q, v, decltype(v)::Zero(nv.value()));
  pin::computeJointJacobians(model, data, q);

  auto J_constraint = eigen::make_matrix<scalar_t>(dyn_index{n_frames * 3}, nv);
  auto gamma_constraint = eigen::make_matrix<scalar_t>(dyn_index{n_frames * 3});

  for (index_t idx = 0; idx < n_frames; ++idx) {
    auto foot_id = size_t(frame_indices[idx]);
    auto J_foot = eigen::make_matrix<scalar_t>(fix_index<6>{}, nv);
    pin::getFrameJacobian(model, data, foot_id, pin::ReferenceFrame::LOCAL, J_foot);

    auto gamma_foot =
        pin::getFrameClassicalAcceleration(model, data, foot_id, pin::ReferenceFrame::LOCAL).linear().eval();

    J_constraint.template middleRows<3>(3 * idx) = J_foot.template topRows<3>();
    gamma_constraint.template middleRows<3>(3 * idx) = gamma_foot;
  }

  auto C = eigen::make_matrix<scalar_t>(nv, nv - fix_index<6>() + dyn_index(n_frames * 3));
  C.bottomLeftCorner(nv.value() - 6, nv.value() - 6).setIdentity();
  C.rightCols(n_frames * 3) = J_constraint.transpose();
  auto const& d = pin::rnea(model, data, q, v, Eigen::Matrix<scalar_t, -1, 1>::Zero(nv.value()));

  auto pinv = (C.transpose() * (C * C.transpose()).inverse()).eval();
  auto ls_sol = pinv * d;
  return ls_sol.topRows(nv.value() - 6);
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::contact_dynamics( //
    mut_view_t<Nv> out_acceleration,       //
    const_view_t<Nq> q,                    //
    const_view_t<Nv> v,                    //
    const_view_t<Nv> tau,                  //
    index_t const frame_indices[],         //
    index_t n_frames,                      //
    key k                                  //
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

  for (index_t i = 0; i < n_frames; ++i) {
    DDP_ASSERT_MSG_ALL_OF(
        ("frame index is out of bounds", frame_indices[i] < m_model->m_impl.nframes),
        ("frame index is out of bounds", frame_indices[i] >= 0));
  }

  auto& model = m_model->m_impl;
  auto& data = k.m_parent->m_impl;

  auto nv = tangent_dim_c();

  pin::forwardKinematics(model, data, q, v, decltype(tau)::Zero(nv.value()));
  pin::computeJointJacobians(model, data, q);

  auto J_constraint = eigen::make_matrix<scalar_t>(dyn_index{n_frames * 3}, nv);

  auto gamma_constraint = eigen::make_matrix<scalar_t>(dyn_index{n_frames * 3});

  for (index_t idx = 0; idx < n_frames; ++idx) {
    auto foot_id = size_t(frame_indices[idx]);
    auto J_foot = eigen::make_matrix<scalar_t>(fix_index<6>{}, nv);
    pin::getFrameJacobian(model, data, foot_id, pin::ReferenceFrame::LOCAL, J_foot);

    auto gamma_foot =
        pin::getFrameClassicalAcceleration(model, data, foot_id, pin::ReferenceFrame::LOCAL).linear().eval();

    J_constraint.template middleRows<3>(3 * idx) = J_foot.template topRows<3>();
    gamma_constraint.template middleRows<3>(3 * idx) = gamma_foot;
  }

  pin::forwardDynamics(model, data, q, v, tau, J_constraint, gamma_constraint, scalar_t(0));
  out_acceleration = data.ddq;

  return k;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::d_contact_dynamics(  //
    mut_view_t<Nv, Nv> out_acceleration_dq,   //
    mut_view_t<Nv, Nv> out_acceleration_dv,   //
    mut_view_t<Nv, Nv> out_acceleration_dtau, //
    const_view_t<Nq> q,                       //
    const_view_t<Nv> v,                       //
    const_view_t<Nv> tau,                     //
    index_t const frame_indices[],            //
    index_t n_frames,                         //
    key k                                     //
) const -> key {

  DDP_ASSERT_MSG_ALL_OF(
      ("output acceleration jacobian with respect to the configuration has the wrong number of rows",
       out_acceleration_dq.rows() == m_tangent_dim),
      ("output acceleration jacobian with respect to the velocity has the wrong number of rows",
       out_acceleration_dv.rows() == m_tangent_dim),
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

  auto& model = m_model->m_impl;
  auto& data = k.m_parent->m_impl;

  for (index_t i = 0; i < n_frames; ++i) {
    DDP_ASSERT_MSG_ALL_OF(
        ("frame index is out of bounds", frame_indices[i] < m_model->m_impl.nframes),
        ("frame index is out of bounds", frame_indices[i] >= 0));
  }

  auto nv = tangent_dim_c();

  pin::forwardKinematics(model, data, q, v, decltype(tau)::Zero(nv.value()));
  pin::computeJointJacobians(model, data, q);

  auto J_constraint = eigen::make_matrix<scalar_t>(dyn_index{n_frames * 3}, nv);

  auto gamma_constraint = eigen::make_matrix<scalar_t>(dyn_index{n_frames * 3});

  for (index_t idx = 0; idx < n_frames; ++idx) {
    auto foot_id = size_t(frame_indices[idx]);
    auto J_foot = eigen::make_matrix<scalar_t>(fix_index<6>{}, nv);
    pin::getFrameJacobian(model, data, foot_id, pin::ReferenceFrame::LOCAL, J_foot);

    auto gamma_foot =
        pin::getFrameClassicalAcceleration(model, data, foot_id, pin::ReferenceFrame::LOCAL).linear().eval();

    J_constraint.template middleRows<3>(long(3 * idx)) = J_foot.template topRows<3>();
    gamma_constraint.template middleRows<3>(long(3 * idx)) = gamma_foot;
  }

  pin::forwardDynamics(model, data, q, v, tau, J_constraint, gamma_constraint, scalar_t(0));
  auto const& a_sol = data.ddq;
  auto const& contact_forces_sol = data.lambda_c;

  dyn_index constr_dim(3 * n_frames);

  auto x = eigen::make_matrix<scalar_t>(nv + constr_dim);
  DDP_BIND(auto, (x_a, x_c), eigen::split_at_row_mut(x, nv));
  x_a = a_sol;
  x_c = contact_forces_sol;

  pin::container::aligned_vector<pin::ForceTpl<scalar_t>> external_forces;
  for (index_t i = 0; i < model.njoints; ++i) {
    external_forces.push_back(pin::ForceTpl<scalar_t>::Zero());
  }

  for (index_t i = 0; i < n_frames; ++i) {
    auto const& frame = model.frames[size_t(frame_indices[i])];
    auto const& joint_id = frame.parent;
    auto const& frame_placement = frame.placement;

    auto fext = pin::ForceTpl<scalar_t>(
        contact_forces_sol.template middleRows<3>(3 * i),
        eigen::make_matrix<scalar_t>(fix_index<3>{}));
    external_forces[joint_id] += frame_placement.act(fext);
  }

  auto Ainv = eigen::make_matrix<scalar_t>(nv + constr_dim, nv + constr_dim);
  pin::getKKTContactDynamicMatrixInverse(model, data, J_constraint, Ainv);
  pin::computeRNEADerivatives(model, data, q, v, a_sol, external_forces);

  auto const& dtau_dq = data.dtau_dq;
  auto const& dtau_dv = data.dtau_dv;
  auto const& dtau_da = data.M;
  (void)dtau_da;

  auto rhs_dq = eigen::make_matrix<scalar_t>(nv + constr_dim, nv);
  auto rhs_dv = eigen::make_matrix<scalar_t>(nv + constr_dim, nv);

  DDP_BIND(auto, (rhs_dq_0, rhs_dq_1), eigen::split_at_row_mut(rhs_dq, nv));
  DDP_BIND(auto, (rhs_dv_0, rhs_dv_1), eigen::split_at_row_mut(rhs_dv, nv));

  rhs_dq_0 = -dtau_dq;
  rhs_dv_0 = -dtau_dv;

  for (index_t i = 0; i < n_frames; ++i) {
    auto dv_dq = eigen::make_matrix<scalar_t>(fix_index<6>(), nv).eval();
    auto da_dq = eigen::make_matrix<scalar_t>(fix_index<6>(), nv).eval();
    auto da_dv = eigen::make_matrix<scalar_t>(fix_index<6>(), nv).eval();
    auto da_da = eigen::make_matrix<scalar_t>(fix_index<6>(), nv).eval();

    auto const& dv_dv = da_da;
    pin::getFrameAccelerationDerivatives(model, data, size_t(frame_indices[i]), pin::LOCAL, dv_dq, da_dq, da_dv, da_da);
    auto v_foot = pin::getFrameVelocity(model, data, size_t(frame_indices[i]), pin::LOCAL);

    rhs_dq_1.template middleRows<3>(3 * i) =
        -(da_dq.template topRows<3>()                                     //
          - pin::skew(v_foot.linear()) * (dv_dq.template bottomRows<3>()) //
          + pin::skew(v_foot.angular()) * (dv_dq.template bottomRows<3>()));

    rhs_dv_1.template middleRows<3>(3 * i) =
        -(da_dv.template topRows<3>()                                     //
          - pin::skew(v_foot.linear()) * (dv_dv.template bottomRows<3>()) //
          + pin::skew(v_foot.angular()) * (dv_dv.template bottomRows<3>()));
  }

  out_acceleration_dq = (Ainv * rhs_dq).topRows(nv.value());
  out_acceleration_dv = (Ainv * rhs_dv).topRows(nv.value());
  out_acceleration_dtau = Ainv.topLeftCorner(nv.value(), nv.value());

  return k;
}

template <typename T, index_t Nq, index_t Nv>
auto model_t<T, Nq, Nv>::frame_coordinates_precompute(const_view_t<Nq> q, key k) const -> key {
  DDP_ASSERT_MSG_ALL_OF( //
      ("configuration vector is not correctly sized", q.rows() == m_config_dim),
      ("invalid data", not q.hasNaN()),
      ("invalid workspace key", static_cast<void*>(k.m_parent) != nullptr));

  impl_data_t* data = k.m_parent;
  pin::framesForwardKinematics(m_model->m_impl, data->m_impl, q);
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
  pin::computeJointJacobians(m_model->m_impl, data->m_impl, q);
  pin::framesForwardKinematics(m_model->m_impl, data->m_impl, q);
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

  pin::getFrameJacobian(m_model->m_impl, data->m_impl, static_cast<std::size_t>(i), pin::LOCAL_WORLD_ALIGNED, w);
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

#endif /* end of include guard PINOCCHIO_MODEL_IPP_2I6Y8FFV */
