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

#include <veg/memory/placement.hpp>

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
struct Model<T>::ModelImpl {
	pin::ModelTpl<T> pin;
};
template <typename T>
struct Model<T>::DataImpl {
	pin::DataTpl<T> pin;
	std::atomic_bool available;

	DataImpl(pin::DataTpl<T> pin, bool b) : pin{VEG_FWD(pin)}, available{b} {}
};

template <typename T>
void Model<T>::Key::destroy() {
	if (parent != nullptr) {
		parent->available = true;
		parent = nullptr;
	}
}

namespace internal {

struct BuilderFromUrdf {
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

struct PinocchioImpl {
	template <typename T>
	static void from_model_geom_builder(
			Model<T>& m,
			FnView<void(pin::Model&, bool)> builder,
			i64 n_parallel,
			bool add_freeflyer_base) {
		VEG_ASSERT_ALL_OF( //
				(n_parallel > 0),
				(n_parallel < 256));

		using ModelImpl = typename Model<T>::ModelImpl;
		using DataImpl = typename Model<T>::DataImpl;

		m.self.num_data = n_parallel;
		m.self.model = static_cast<ModelImpl*>(
				boost::alignment::aligned_alloc(alignof(ModelImpl), sizeof(ModelImpl)));

		if (m.self.model == nullptr) {
			throw std::bad_alloc();
		}
		m.self.data =
				mem::launder(static_cast<DataImpl*>(boost::alignment::aligned_alloc(
						alignof(DataImpl),
						narrow<usize>(m.self.num_data) * sizeof(DataImpl))));

		if (m.self.data == nullptr) {
			boost::alignment::aligned_free(m.self.model);
			throw std::bad_alloc();
		}

		pin::Model model_double{};
		builder(model_double, add_freeflyer_base);
		m.self.model = ::new (m.self.model) ModelImpl{model_double.cast<T>()};

		for (i64 i = 0; i < m.self.num_data; ++i) {
			::new (m.self.data + i)
					DataImpl{pin::DataTpl<T>{m.self.model->pin}, true};
		}

		m.self.config_dim = m.self.model->pin.nq;
		m.self.tangent_dim = m.self.model->pin.nv;
	}
};

} // namespace internal

template <typename T>
Model<T>::Model(
		::fmt::string_view urdf_path, i64 n_parallel, bool add_freeflyer_base) {
	using FnView = veg::fn::FnView<void(pin::Model&, bool)>;
	internal::PinocchioImpl::from_model_geom_builder(
			*this,
			FnView{as_ref, internal::BuilderFromUrdf{urdf_path}},
			n_parallel,
			add_freeflyer_base);
}

template <typename T>
auto Model<T>::all_joints_test_model(i64 n_parallel) -> Model {
	using FnView = veg::fn::FnView<void(pin::Model&, bool)>;
	Model m;
	internal::PinocchioImpl::from_model_geom_builder(
			m, FnView{as_ref, &pin::buildAllJointsModel}, n_parallel, false);
	return m;
}

template <typename T>
Model<T>::~Model() {
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
Model<T>::Model(Model&& other) noexcept : self{other.self} {
	other.self = {nullptr, nullptr, 0, 0, 0};
}

template <typename T>
auto Model<T>::acquire_workspace() const -> Key {
	for (i64 i = 0; i < self.num_data; ++i) {
		std::atomic_bool& available = self.data[i].available;

		if (available) {
			bool expected = true;
			bool const changed = available.compare_exchange_strong(expected, false);
			if (changed) {
				return Key{self.data[i]};
			}
		}
	}
	VEG_ASSERT_ELSE("no workspace available", false);
	std::terminate();
}

template <typename T>
auto Model<T>::model_name() const -> ::fmt::string_view {
	return self.model->pin.name;
}

template <typename T>
void Model<T>::neutral_configuration(MutVec out_q) const {

	VEG_DEBUG_ASSERT(out_q.rows() == self.config_dim);
	pin::neutral(self.model->pin, out_q);
}

template <typename T>
void Model<T>::random_configuration(MutVec out_q) const {

	VEG_DEBUG_ASSERT(out_q.rows() == self.config_dim);
	pin::randomConfiguration(               //
			self.model->pin,                    //
			self.model->pin.lowerPositionLimit, //
			self.model->pin.upperPositionLimit, //
			out_q                               //
	);
}

template <typename T>
void Model<T>::integrate(
		MutVec out_q, //
		ConstVec q,   //
		ConstVec v    //
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
void Model<T>::d_integrate_dq(
		MutColMat out_q_dq, //
		ConstVec q,         //
		ConstVec v          //
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
void Model<T>::d_integrate_dv(
		MutColMat out_q_dv, //
		ConstVec q,         //
		ConstVec v          //
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
void Model<T>::difference(
		MutVec out_v,     //
		ConstVec q_start, //
		ConstVec q_finish //
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
void Model<T>::d_difference_dq_start(
		MutColMat out_v_dq_start, //
		ConstVec q_start,         //
		ConstVec q_finish         //
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
void Model<T>::d_difference_dq_finish(
		MutColMat out_v_dq_finish, //
		ConstVec q_start,          //
		ConstVec q_finish          //
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
auto collect(i64 njoints, FnView<Option<Tuple<i64, Force<T>>>()> fext)
		-> pin::container::aligned_vector<pin::ForceTpl<T>> {

	pin::container::aligned_vector<pin::ForceTpl<T>> external_forces(
			narrow<usize>(njoints));
	while (true) {
		auto res = fext();
		if (res.is_none()) {
			break;
		}
		VEG_BIND(auto, (i, f), VEG_FWD(res).unwrap());
		auto ui = narrow<usize>(i);

		VEG_ASSERT(i < njoints);
		{
			auto _ = slice::from_array(f.angular._);
			external_forces[ui].angular() = eigen::slice_to_vec(_);
		}
		{
			auto _ = slice::from_array(f.linear._);
			external_forces[ui].linear() = eigen::slice_to_vec(_);
		}
	}
	return external_forces;
}

template <typename T>
auto Model<T>::dynamics_aba(                                 //
		MutVec out_acceleration,                                 //
		ConstVec q,                                              //
		ConstVec v,                                              //
		ConstVec tau,                                            //
		Option<FnView<Option<Tuple<i64, Force<T>>>()>> fext_gen, //
		Key k                                                    //
) const -> Key {
	VEG_DEBUG_ASSERT_ALL_OF(
			(out_acceleration.rows() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(tau.rows() == self.tangent_dim),
			(not q.hasNaN()),
			(not v.hasNaN()),
			(not tau.hasNaN()),
			(k));

	DataImpl* data = k.parent;

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
auto Model<T>::inverse_dynamics_rnea(                        //
		MutVec out_tau,                                          //
		ConstVec q,                                              //
		ConstVec v,                                              //
		ConstVec a,                                              //
		Option<FnView<Option<Tuple<i64, Force<T>>>()>> fext_gen, //
		Key k                                                    //
) const -> Key {

	VEG_DEBUG_ASSERT_ALL_OF(
			(out_tau.rows() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(a.rows() == self.tangent_dim),
			(not q.hasNaN()),
			(not v.hasNaN()),
			(not a.hasNaN()),
			(k));

	DataImpl* data = k.parent;

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
auto Model<T>::d_dynamics_aba(                               //
		MutColMat out_acceleration_dq,                           //
		MutColMat out_acceleration_dv,                           //
		MutColMat out_acceleration_dtau,                         //
		MutVec out_acceleration,                                 //
		ConstVec q,                                              //
		ConstVec v,                                              //
		ConstVec tau,                                            //
		Option<FnView<Option<Tuple<i64, Force<T>>>()>> fext_gen, //
		Key k                                                    //
) const -> Key {

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

	DataImpl* data = k.parent;

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
auto Model<T>::d_inverse_dynamics_rnea(                      //
		MutColMat out_tau_dq,                                    //
		MutColMat out_tau_dv,                                    //
		MutColMat out_tau_da,                                    //
		ConstVec q,                                              //
		ConstVec v,                                              //
		ConstVec a,                                              //
		Option<FnView<Option<Tuple<i64, Force<T>>>()>> fext_gen, //
		Key k                                                    //
) const -> Key {

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

	DataImpl* data = k.parent;

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
auto Model<T>::compute_forward_kinematics(
		ConstVec q, ConstVec v, ConstVec tau, Key k) const -> Key {

	VEG_DEBUG_ASSERT_ALL_OF(
			(tau.rows() == self.tangent_dim),
			(q.rows() == self.config_dim),
			(v.rows() == self.tangent_dim),
			(tau.rows() == self.tangent_dim));

	DataImpl* data = k.parent;
	pin::forwardKinematics(self.model->pin, data->pin, q, v, tau);

	return k;
}

template <typename T>
auto Model<T>::compute_forward_dynamics(
		ConstVec q,
		ConstVec v,
		ConstVec tau,
		ConstColMat J,
		ConstVec gamma,
		T inv_damping,
		Key k) const -> Key {

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
auto Model<T>::set_ddq(MutVec a, Key k) const -> Key {

	VEG_DEBUG_ASSERT(a.rows() == self.tangent_dim);

	auto& data = k.parent->pin;
	eigen::assign(a, data.ddq);

	return k;
}

template <typename T>
auto Model<T>::frame_classical_acceleration(i64 frame_id, Key k) const
		-> Tuple<Key, Motion<T>> {

	DataImpl* data = k.parent;
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
			direct,
			VEG_FWD(k),
			Motion<T>{
					{ang[0], ang[1], ang[2]},
					{lin[0], lin[1], lin[2]},
			},
	};
}

template <typename T>
auto Model<T>::frame_velocity(i64 frame_id, Key k) const
		-> Tuple<Key, Motion<T>> {
	auto& data = k.parent->pin;
	auto const& vel = pin::getFrameVelocity(
			self.model->pin, data, narrow<pin::FrameIndex>(frame_id), pin::LOCAL);

	auto const& lin = vel.linear();
	auto const& ang = vel.angular();

	return {
			direct,
			VEG_FWD(k),
			Motion<T>{
					{ang[0], ang[1], ang[2]},
					{lin[0], lin[1], lin[2]},
			}};
}

template <typename T>
auto Model<T>::d_frame_acceleration(
		MutColMat dv_dq,
		MutColMat da_dq,
		MutColMat da_dv,
		MutColMat da_da,
		i64 frame_id,
		Key k) const -> Key {

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
auto Model<T>::frames_forward_kinematics(ConstVec q, Key k) const -> Key {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(q.rows() == self.config_dim),
			(not q.hasNaN()),
			(k));

	DataImpl* data = k.parent;
	pin::framesForwardKinematics(self.model->pin, data->pin, q);
	return k;
}

template <typename T>
auto Model<T>::compute_joint_jacobians(ConstVec q, Key k) const -> Key {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(q.rows() == self.config_dim),
			(not q.hasNaN()),
			(k));

	DataImpl* data = k.parent;
	pin::computeJointJacobians(self.model->pin, data->pin, q);
	return k;
}

template <typename T>
auto Model<T>::frame_se3(i64 frame_id, Key k) const -> Tuple<Key, Se3<T>> {

	VEG_DEBUG_ASSERT_ALL_OF( //
			(frame_id >= 0),
			(frame_id < self.model->pin.nframes),
			(k));

	auto frame_uid = narrow<usize>(frame_id);

	DataImpl* data = k.parent;
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
			direct,
			VEG_FWD(k),
			Se3<T>{
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
auto Model<T>::d_frame_se3(MutColMat out, i64 frame_id, Key k) const -> Key {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(frame_id >= 0),
			(frame_id < self.model->pin.nframes),

			(out.cols() == self.tangent_dim),
			(k));

	DataImpl* data = k.parent;
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
auto Model<T>::n_frames() const -> i64 {
	return self.model->pin.nframes;
}

template <typename T>
auto Model<T>::frame_name(i64 i) const -> ::fmt::string_view {
	VEG_DEBUG_ASSERT_ALL_OF( //
			(i >= 0),
			(i < self.model->pin.nframes));

	return self.model->pin.frames[narrow<usize>(i)].name;
}

template <typename T>
auto Model<T>::kkt_contact_dynamics_matrix_inverse(
		MutColMat out, ConstColMat J, Key k) const -> Key {
	auto& model = self.model->pin;
	auto& data = k.parent->pin;

	pin::getKKTContactDynamicMatrixInverse(model, data, J, out);

	return k;
}

template <typename T>
auto Model<T>::skew(ConstVec v) -> SkewMat<T> {
	VEG_DEBUG_ASSERT(v.rows() == 3);
	Eigen::Map<Eigen::Matrix<T, 3, 1, Eigen::ColMajor> const> in{v.data()};
	Eigen::Matrix<T, 3, 3, Eigen::ColMajor> out = pin::skew(in);
	Array<T, 9> data;
	for (i64 i = 0; i < 9; ++i) {
		data._[i] = out.data()[i];
	}
	return {VEG_FWD(data)};
}

} // namespace pinocchio
} // namespace ddp
#endif /* end of include guard DDP_PINOCCHIO_PINOCCHIO_MODEL_IPP_2DC3RHY7S */
