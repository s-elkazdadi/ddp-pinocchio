#ifndef DDP_PINOCCHIO_PINOCCHIO_MODEL_IPP_2DC3RHY7S
#define DDP_PINOCCHIO_PINOCCHIO_MODEL_IPP_2DC3RHY7S

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

#include <stdexcept>
#include <atomic>
#include <fmt/ostream.h>

#include <veg/internal/memory.hpp>

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

template <typename T>
struct model<T>::model_impl {
	pin::ModelTpl<T> pin;
};
template <typename T>
struct model<T>::data_impl {
	pin::DataTpl<T> pin;
	std::atomic_bool available;

	data_impl(pin::DataTpl<T> pin, bool b) : pin{VEG_FWD(pin)}, available{b} {}
};

template <typename T>
void model<T>::key::destroy() {
	if (parent != nullptr) {
		parent->available = true;
		parent = nullptr;
	}
}

namespace internal {

struct builder_from_urdf_t {
	::fmt::string_view urdf_path;
	void operator()(pin::Model& model, bool add_freeflyer_base) const {
		std::string path;
		if (urdf_path[0] == '~') {
#ifdef _WIN32
			::fmt::string_view home_drive = std::getenv("HOMEDRIVE");
			::fmt::string_view home_path = std::getenv("HOMEPATH");
			path.append(home_drive.begin(), home_drive.end());
			path.append(home_path.begin(), home_path.end());
#else
			::fmt::string_view home_dir = std::getenv("HOME");
			path.append(home_dir.begin(), home_dir.end());
#endif
			::fmt::string_view tail{urdf_path.begin() + 1, urdf_path.size() - 1};
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
	template <typename T>
	static void from_model_geom_builder(
			model<T>& m,
			fn::fn_view<void(pin::Model&, bool)> builder,
			i64 n_parallel,
			bool add_freeflyer_base) {
		VEG_ASSERT_ALL_OF( //
				(n_parallel > 0),
				(n_parallel < 256));

		using model_impl = typename model<T>::model_impl;
		using data_impl = typename model<T>::data_impl;

		m.self.num_data = n_parallel;
		m.self.model = static_cast<model_impl*>(boost::alignment::aligned_alloc(
				alignof(model_impl), sizeof(model_impl)));

		if (m.self.model == nullptr) {
			throw std::bad_alloc();
		}
		m.self.data =
				VEG_LAUNDER(static_cast<data_impl*>(boost::alignment::aligned_alloc(
						alignof(data_impl),
						narrow<usize>(m.self.num_data) * sizeof(data_impl))));

		if (m.self.data == nullptr) {
			boost::alignment::aligned_free(m.self.model);
			throw std::bad_alloc();
		}

		pin::Model model_double{};
		builder(model_double, add_freeflyer_base);
		m.self.model =
				mem::aggregate_construct_at(m.self.model, model_double.cast<T>());

		for (i64 i = 0; i < m.self.num_data; ++i) {
			mem::aggregate_construct_at(
					m.self.data + i, pin::DataTpl<T>{m.self.model->pin}, true);
		}

		m.self.config_dim = m.self.model->pin.nq;
		m.self.tangent_dim = m.self.model->pin.nv;
	}
};

} // namespace internal

template <typename T>
model<T>::model(
		::fmt::string_view urdf_path, i64 n_parallel, bool add_freeflyer_base) {
	using fn_view = fn::fn_view<void(pin::Model&, bool)>;
	internal::pinocchio_impl::from_model_geom_builder(
			*this,
			fn_view{internal::builder_from_urdf_t{urdf_path}},
			n_parallel,
			add_freeflyer_base);
}

template <typename T>
auto model<T>::all_joints_test_model(i64 n_parallel) -> model {
	using fn_view = fn::fn_view<void(pin::Model&, bool)>;
	model m;
	internal::pinocchio_impl::from_model_geom_builder(
			m, fn_view{&pin::buildAllJointsModel}, n_parallel, false);
	return m;
}

template <typename T>
model<T>::~model() {
	VEG_ASSERT(
			"model and data should either be both valid or both invalid (moved-from "
			"state)",
			(self.model == nullptr) == (self.data == nullptr));

	if (self.model != nullptr) {
		for (i64 i = 0; i < self.num_data; ++i) {
			mem::destroy_at(self.data + i);
		}
		boost::alignment::aligned_free(self.data);
	}

	if (self.model != nullptr) {
		mem::destroy_at(self.model);
		boost::alignment::aligned_free(self.model);
	}
}

template <typename T>
model<T>::model(model&& other) noexcept : self{other.self} {
	other.self = {nullptr, nullptr, 0, 0, 0};
}

template <typename T>
auto model<T>::acquire_workspace() const -> key {
	for (i64 i = 0; i < self.num_data; ++i) {
		std::atomic_bool& available = self.data[i].available;

		if (available) {
			bool expected = true;
			bool const changed = available.compare_exchange_strong(expected, false);
			if (changed) {
				return key{self.data[i]};
			}
		}
	}
	VEG_ASSERT_ELSE("no workspace available", false);
	std::terminate();
}

template <typename T>
auto model<T>::model_name() const -> ::fmt::string_view {
	return self.model->pin.name;
}

template <typename T>
void model<T>::neutral_configuration(mut_vec out_q) const {

	VEG_DEBUG_ASSERT(out_q.rows() == self.config_dim);
	pin::neutral(self.model->pin, out_q);
}

template <typename T>
void model<T>::random_configuration(mut_vec out_q) const {

	VEG_DEBUG_ASSERT(out_q.rows() == self.config_dim);
	pin::randomConfiguration(               //
			self.model->pin,                    //
			self.model->pin.lowerPositionLimit, //
			self.model->pin.upperPositionLimit, //
			out_q                               //
	);
}

template <typename T>
void model<T>::integrate(
		mut_vec out_q, //
		const_vec q,   //
		const_vec v    //
) const {
	VEG_DEBUG_ASSERT_ALL_OF(
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(out_q.rows() == self.config_dim),
			(not q.hasNaN()),
			(not v.hasNaN()));
	;
	pin::integrate(self.model->pin, q, v, out_q);
}

template <typename T>
void model<T>::d_integrate_dq(
		mut_colmat out_q_dq, //
		const_vec q,         //
		const_vec v          //
) const {
	VEG_DEBUG_ASSERT_ALL_OF(
			(out_q_dq.rows() == self.tangent_dim),
			(out_q_dq.cols() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(not q.hasNaN()),
			(not v.hasNaN()));
	pin::dIntegrate(self.model->pin, q, v, out_q_dq, pin::ArgumentPosition::ARG0);
}

template <typename T>
void model<T>::d_integrate_dv(
		mut_colmat out_q_dv, //
		const_vec q,         //
		const_vec v          //
) const {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_q_dv.rows() == self.tangent_dim),
			(out_q_dv.cols() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(not q.hasNaN()),
			(not v.hasNaN()));
	pin::dIntegrate(self.model->pin, q, v, out_q_dv, pin::ArgumentPosition::ARG1);
}

template <typename T>
void model<T>::difference(
		mut_vec out_v,     //
		const_vec q_start, //
		const_vec q_finish //
) const {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_v.rows() == self.tangent_dim),
			(q_start.rows() == self.config_dim),
			(q_finish.rows() == self.config_dim),
			(not q_start.hasNaN()),
			(not q_finish.hasNaN()));

	pin::difference(self.model->pin, q_start, q_finish, out_v);
}

template <typename T>
void model<T>::d_difference_dq_start(
		mut_colmat out_v_dq_start, //
		const_vec q_start,         //
		const_vec q_finish         //
) const {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_v_dq_start.rows() == self.tangent_dim),
			(out_v_dq_start.cols() == self.tangent_dim),
			(q_start.rows() == self.config_dim),
			(q_finish.rows() == self.config_dim),
			(not q_start.hasNaN()),
			(not q_finish.hasNaN()));

	pin::dDifference(
			self.model->pin,
			q_start,
			q_finish,
			out_v_dq_start,
			pin::ArgumentPosition::ARG0);
}

template <typename T>
void model<T>::d_difference_dq_finish(
		mut_colmat out_v_dq_finish, //
		const_vec q_start,          //
		const_vec q_finish          //
) const {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_v_dq_finish.rows() == self.tangent_dim),
			(out_v_dq_finish.cols() == self.tangent_dim),
			(q_start.rows() == self.config_dim),
			(q_finish.rows() == self.config_dim),
			(not q_start.hasNaN()),
			(not q_finish.hasNaN()));

	pin::dDifference(
			self.model->pin,
			q_start,
			q_finish,
			out_v_dq_finish,
			pin::ArgumentPosition::ARG1);
}

template <typename T>
auto collect(
		i64 njoints, fn::fn_view<fn::nothrow<option<tuple<i64, force<T>>>()>> fext)
		-> pin::container::aligned_vector<pin::ForceTpl<T>> {

	pin::container::aligned_vector<pin::ForceTpl<T>> external_forces(
			narrow<usize>(njoints));
	while (true) {
		auto res = fext();
		if (!res) {
			break;
		}
		VEG_BIND(auto, (i, f), VEG_FWD(res).unwrap());
		auto ui = narrow<usize>(i);

		VEG_ASSERT(i < njoints);
		external_forces[ui].angular() = eigen::slice_to_vec(f.angular);
		external_forces[ui].linear() = eigen::slice_to_vec(f.linear);
	}
	return external_forces;
}

template <typename T>
auto model<T>::dynamics_aba(  //
		mut_vec out_acceleration, //
		const_vec q,              //
		const_vec v,              //
		const_vec tau,            //
		option<fn::fn_view<fn::nothrow<option<tuple<i64, force<T>>>()>>>
				fext_gen, //
		key k         //
) const -> key {
	VEG_DEBUG_ASSERT_ALL_OF(
			(out_acceleration.rows() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(tau.rows() == self.tangent_dim),
			(not q.hasNaN()),
			(not v.hasNaN()),
			(not tau.hasNaN()),
			(k));

	data_impl* data = k.parent;

	VEG_FWD(fext_gen).map_or_else(
			[&](auto fn) {
				auto fext_vec = collect(self.model->pin.njoints, fn);
				pin::aba(self.model->pin, data->pin, q, v, tau, fext_vec);
			},
			[&] { pin::aba(self.model->pin, data->pin, q, v, tau); });
	out_acceleration = data->pin.ddq;

	VEG_DEBUG_ASSERT_ELSE(
			::fmt::format(
					"invalid output acceleration:\n"
					"{}\n"
					"inputs are\n"
					"q  : {}\n"
					"v  : {}\n"
					"tau: {}\n",
					out_acceleration,
					q.transpose(),
					v.transpose(),
					tau.transpose()),
			not out_acceleration.hasNaN());

	return k;
}

template <typename T>
auto model<T>::inverse_dynamics_rnea( //
		mut_vec out_tau,                  //
		const_vec q,                      //
		const_vec v,                      //
		const_vec a,                      //
		option<fn::fn_view<fn::nothrow<option<tuple<i64, force<T>>>()>>>
				fext_gen, //
		key k         //
) const -> key {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_tau.rows() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(a.rows() == self.tangent_dim),
			(not q.hasNaN()),
			(not v.hasNaN()),
			(not a.hasNaN()),
			(k));

	data_impl* data = k.parent;

	VEG_FWD(fext_gen).map_or_else(
			[&](auto fn) {
				auto fext_vec = collect(self.model->pin.njoints, fn);
				pin::rnea(self.model->pin, data->pin, q, v, a, fext_vec);
			},
			[&] { pin::rnea(self.model->pin, data->pin, q, v, a); });
	out_tau = data->pin.ddq;

	VEG_DEBUG_ASSERT_ELSE(
			::fmt::format(
					"invalid output torque:\n"
					"{}\n"
					"inputs are\n"
					"q  : {}\n"
					"v  : {}\n"
					"a  : {}\n",
					out_tau,
					q.transpose(),
					v.transpose(),
					a.transpose()),
			not out_tau.hasNaN());

	return k;
}

template <typename T>
auto model<T>::d_dynamics_aba(        //
		mut_colmat out_acceleration_dq,   //
		mut_colmat out_acceleration_dv,   //
		mut_colmat out_acceleration_dtau, //
		mut_vec out_acceleration,         //
		const_vec q,                      //
		const_vec v,                      //
		const_vec tau,                    //
		option<fn::fn_view<fn::nothrow<option<tuple<i64, force<T>>>()>>>
				fext_gen, //
		key k         //
) const -> key {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_acceleration_dq.rows() == self.tangent_dim),
			(out_acceleration_dv.rows() == self.tangent_dim),
			(out_acceleration_dq.cols() == self.tangent_dim),
			(out_acceleration_dv.cols() == self.tangent_dim),
			(out_acceleration_dtau.rows() == self.tangent_dim),
			(eigen::as_const(out_acceleration_dtau).cols() == self.tangent_dim),

			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(tau.rows() == self.tangent_dim),

			(!q.hasNaN()),
			(!v.hasNaN()),
			(!tau.hasNaN()),

			(static_cast<void*>(k.parent) != nullptr));

	data_impl* data = k.parent;

	VEG_FWD(fext_gen).map_or_else(
			[&](auto fn) {
				auto fext_vec = collect(self.model->pin.njoints, fn);
				pin::computeABADerivatives(
						self.model->pin,
						data->pin,
						q,
						v,
						tau,
						fext_vec,
						out_acceleration_dq,
						out_acceleration_dv,
						out_acceleration_dtau);
			},
			[&] {
				pin::computeABADerivatives(
						self.model->pin,
						data->pin,
						q,
						v,
						tau,
						out_acceleration_dq,
						out_acceleration_dv,
						out_acceleration_dtau);
			});
	out_acceleration = data->pin.ddq;

	VEG_DEBUG_ASSERT_ALL_OF_ELSE(
			(::fmt::format(
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
			(::fmt::format(
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
			(::fmt::format(
					 "invalid output derivative:\n"
					 "{}\n"
					 "inputs are\n"
					 "q  : {}\n"
					 "v  : {}\n"
					 "tau: {}\n",
					 out_acceleration_dtau,
					 q.transpose(),
					 v.transpose(),
					 tau.transpose()),
	     not out_acceleration_dtau.hasNaN()));

	return k;
}

template <typename T>
auto model<T>::d_inverse_dynamics_rnea( //
		mut_colmat out_tau_dq,              //
		mut_colmat out_tau_dv,              //
		mut_colmat out_tau_da,              //
		const_vec q,                        //
		const_vec v,                        //
		const_vec a,                        //
		option<fn::fn_view<fn::nothrow<option<tuple<i64, force<T>>>()>>>
				fext_gen, //
		key k         //
) const -> key {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_tau_dq.rows() == self.tangent_dim),
			(out_tau_dv.rows() == self.tangent_dim),
			(out_tau_dq.cols() == self.tangent_dim),
			(out_tau_dv.cols() == self.tangent_dim),
			(out_tau_da.rows() == self.tangent_dim),
			(out_tau_da.cols() == self.tangent_dim),

			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(a.rows() == self.tangent_dim),

			(!q.hasNaN()),
			(!v.hasNaN()),
			(!a.hasNaN()),

			(k));

	data_impl* data = k.parent;

	VEG_FWD(fext_gen).map_or_else(
			[&](auto fn) {
				auto fext_vec = collect(self.model->pin.njoints, fn);
				pin::computeRNEADerivatives(
						self.model->pin,
						data->pin,
						q,
						v,
						a,
						fext_vec,
						out_tau_dq,
						out_tau_dv,
						out_tau_da);
			},
			[&] {
				pin::computeRNEADerivatives(
						self.model->pin,
						data->pin,
						q,
						v,
						a,
						out_tau_dq,
						out_tau_dv,
						out_tau_da);
			});

	VEG_DEBUG_ASSERT_ALL_OF_ELSE(
			(::fmt::format(
					 "invalid output torque:\n"
					 "{}\n"
					 "inputs are\n"
					 "q  : {}\n"
					 "v  : {}\n"
					 "a  : {}\n",
					 out_tau_dq,
					 q.transpose(),
					 v.transpose(),
					 a.transpose()),
	     !out_tau_dq.hasNaN()),
			(::fmt::format(
					 "invalid output derivative:\n"
					 "{}\n"
					 "inputs are\n"
					 "q  : {}\n"
					 "v  : {}\n"
					 "a  : {}\n",
					 out_tau_dv,
					 q.transpose(),
					 v.transpose(),
					 a.transpose()),
	     !out_tau_dv.hasNaN()),
			(::fmt::format(
					 "invalid output derivative:\n"
					 "{}\n"
					 "inputs are\n"
					 "q  : {}\n"
					 "v  : {}\n"
					 "a  : {}\n",
					 out_tau_da,
					 q.transpose(),
					 v.transpose(),
					 a.transpose()),
	     !out_tau_da.hasNaN()));

	return k;
}

template <typename T>
auto model<T>::compute_forward_kinematics(
		const_vec q, const_vec v, const_vec tau, key k) const -> key {

	VEG_DEBUG_ASSERT_ALL_OF(
			(tau.rows() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(tau.rows() == self.tangent_dim));

	data_impl* data = k.parent;
	pin::forwardKinematics(self.model->pin, data->pin, q, v, tau);

	return k;
}

template <typename T>
auto model<T>::compute_forward_dynamics(
		const_vec q,
		const_vec v,
		const_vec tau,
		const_colmat J,
		const_vec gamma,
		T inv_damping,
		key k) const -> key {

	VEG_DEBUG_ASSERT_ALL_OF(
			(tau.rows() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(tau.rows() == self.tangent_dim));

	auto& model = self.model->pin;
	auto& data = k.parent->pin;
	pin::forwardDynamics(model, data, q, v, tau, J, gamma, inv_damping);
	return k;
}

template <typename T>
auto model<T>::frame_classical_acceleration(i64 frame_id, key k) const
		-> tuple<key, motion<T>> {

	data_impl* data = k.parent;
	auto acc = pin::getFrameClassicalAcceleration(
			self.model->pin,
			data->pin,
			narrow<pin::FrameIndex>(frame_id),
			pin::ReferenceFrame::LOCAL);

	auto const& ang = acc.angular();
	auto const& lin = acc.linear();

	// sanity check
	VEG_DEBUG_ASSERT_ALL_OF( //
			(lin.rows() == 3),
			(lin.cols() == 1),

			(ang.rows() == 3),
			(ang.cols() == 1));

	return {
			elems,
			VEG_FWD(k),
			{
					{ang[0], ang[1], ang[2]},
					{lin[0], lin[1], lin[2]},
			},
	};
}

template <typename T>
auto model<T>::frame_velocity(i64 frame_id, key k) const
		-> tuple<key, motion<T>> {
	auto& data = k.parent->pin;
	auto const& vel = pin::getFrameVelocity(
			self.model->pin, data, narrow<pin::FrameIndex>(frame_id), pin::LOCAL);

	auto const& lin = vel.linear();
	auto const& ang = vel.angular();

	return {
			elems,
			VEG_FWD(k),
			{
					{ang[0], ang[1], ang[2]},
					{lin[0], lin[1], lin[2]},
			}};
}

template <typename T>
auto model<T>::d_frame_acceleration(
		mut_colmat dv_dq,
		mut_colmat da_dq,
		mut_colmat da_dv,
		mut_colmat da_da,
		i64 frame_id,
		key k) const -> key {

	pin::getFrameAccelerationDerivatives(
			self.model->pin,
			k.parent->pin,
			narrow<pin::FrameIndex>(frame_id),
			pin::LOCAL,
			dv_dq,
			da_dq,
			da_dv,
			da_da);
	return k;
}

template <typename T>
auto model<T>::frames_forward_kinematics(const_vec q, key k) const -> key {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(q.rows() == self.config_dim),
			(not q.hasNaN()),
			(k));

	data_impl* data = k.parent;
	pin::framesForwardKinematics(self.model->pin, data->pin, q);
	return k;
}

template <typename T>
auto model<T>::compute_joint_jacobians(const_vec q, key k) const -> key {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(q.rows() == self.config_dim),
			(not q.hasNaN()),
			(k));

	data_impl* data = k.parent;
	pin::computeJointJacobians(self.model->pin, data->pin, q);
	return k;
}

template <typename T>
auto model<T>::frame_se3(i64 frame_id, key k) const -> tuple<key, se3<T>> {

	VEG_DEBUG_ASSERT_ALL_OF( //
			(frame_id >= 0),
			(frame_id < self.model->pin.nframes),
			(k));

	auto frame_uid = narrow<usize>(frame_id);

	data_impl* data = k.parent;
	auto out = data->pin.oMf[frame_uid].translation();

	auto const& trans = data->pin.oMf[frame_uid].translation();
	auto const& rot = data->pin.oMf[frame_uid].rotation();

	// sanity check
	VEG_DEBUG_ASSERT_ALL_OF( //
			(trans.rows() == 3),
			(trans.cols() == 1),

			(rot.rows() == 3),
			(rot.cols() == 3));

	return {
			elems,
			VEG_FWD(k),
			{
					// clang-format off
          {
            rot(0, 0), rot(1, 0), rot(2, 0),
            rot(0, 1), rot(1, 1), rot(2, 1),
            rot(0, 2), rot(1, 2), rot(2, 2),
          },
					// clang-format on
					{trans[0], trans[1], trans[2]},
			},
	};
};

template <typename T>
auto model<T>::d_frame_se3(mut_colmat out, i64 frame_id, key k) const -> key {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(frame_id >= 0),
			(frame_id < self.model->pin.nframes),

			(out.cols() == self.tangent_dim),
			(k));

	data_impl* data = k.parent;
	out.setZero();

	pin::getFrameJacobian(
			self.model->pin,
			data->pin,
			narrow<pin::FrameIndex>(frame_id),
			pin::LOCAL_WORLD_ALIGNED,
			out);
	return k;
};

template <typename T>
auto model<T>::n_frames() const -> i64 {
	return self.model->pin.nframes;
}

template <typename T>
auto model<T>::frame_name(i64 i) const -> ::fmt::string_view {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(i >= 0),
			(i < self.model->pin.nframes));

	return self.model->pin.frames[narrow<usize>(i)].name;
}

template <typename T>
auto model<T>::kkt_contact_dynamics_matrix_inverse(
		mut_colmat out, const_colmat J, key k) const -> key {
	auto& model = self.model->pin;
	auto& data = k.parent->pin;

	pin::getKKTContactDynamicMatrixInverse(model, data, J, out);

	return k;
}

template <typename T>
auto model<T>::skew(const_vec v) -> skew_mat<T> {
	VEG_DEBUG_ASSERT(v.rows() == 3);
	Eigen::Map<Eigen::Matrix<T, 3, 1, Eigen::ColMajor> const> in{v.data()};
	Eigen::Matrix<T, 3, 3, Eigen::ColMajor> out = pin::skew(in);
	array<T, 9> data;
	for (i64 i = 0; i < 9; ++i) {
		data[i] = out.data()[i];
	}
	return {data};
}

} // namespace pinocchio
} // namespace ddp
#endif /* end of include guard DDP_PINOCCHIO_PINOCCHIO_MODEL_IPP_2DC3RHY7S */
