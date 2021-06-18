#ifndef DDP_PINOCCHIO_PINOCCHIO_MODEL_HPP_IRYFG86GS
#define DDP_PINOCCHIO_PINOCCHIO_MODEL_HPP_IRYFG86GS

#include "ddp/internal/utils.hpp"
#include "ddp/internal/eigen.hpp"
#include <memory>

namespace ddp {
namespace pinocchio {

template <typename T>
struct SkewMat {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 9> data;
};

template <typename T>
struct Se3 {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 9> rot /* 3x3 col major */;
	Array<T, 3> trans;
};
template <typename T>
struct Motion {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 3> angular;
	Array<T, 3> linear;
};
template <typename T>
struct Force {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 3> angular;
	Array<T, 3> linear;
};

namespace internal {
struct PinocchioImpl;
} // namespace internal

template <typename T>
struct Model {
	DDP_CHECK_CONCEPT(scalar<T>);

	using Scalar = T;

	using MutVec = View<T, colvec>;
	using ConstVec = View<T const, colvec>;

	using MutColMat = View<T, colmat>;
	using ConstColMat = View<T const, colmat>;

private:
	struct ModelImpl;
	struct DataImpl;
	friend struct internal::PinocchioImpl;
	Model() = default;

	struct layout {
		ModelImpl* model = nullptr;
		DataImpl* data = nullptr;
		i64 num_data = 0;
		i64 config_dim = 0;
		i64 tangent_dim = 0;
	} self;

public:
	struct Key {
		Key() = default;

		Key(Key const&) = delete;
		Key(Key&& other) noexcept : parent{other.parent} { other.parent = nullptr; }
		auto operator=(Key const&) -> Key& = delete;
		auto operator=(Key&& other) noexcept -> Key& {
			VEG_DEBUG_ASSERT("no wasting keys :<", !(*this));

			destroy();
			parent = other.parent;
			other.parent = nullptr;
			return *this;
		}
		~Key() { destroy(); }

		explicit operator bool() const { return parent != nullptr; }

	private:
		void destroy();
		friend struct Model;
		explicit Key(DataImpl& ref) : parent{&ref} {};
		DataImpl* parent = nullptr;
	};

	~Model();
	Model(Model const&) = delete;
	Model(Model&&) noexcept;

	auto operator=(Model const&) -> Model& = delete;
	auto operator=(Model&&) -> Model& = delete;

	auto model_name() const -> ::fmt::string_view;

	explicit Model(
			::fmt::string_view urdf_path,
			i64 n_parallel = 1,
			bool add_freeflyer_base = false);
	static auto all_joints_test_model(i64 n_parallel = 1) -> Model;

	auto config_dim() const -> i64 { return self.config_dim; }
	auto tangent_dim() const -> i64 { return self.tangent_dim; }

	auto acquire_workspace() const -> Key;

	void neutral_configuration(MutVec out_q) const;
	void random_configuration(MutVec out_q) const;

	void integrate(   //
			MutVec out_q, //
			ConstVec q,   //
			ConstVec v    //
	) const;

	void d_integrate_dq(    //
			MutColMat out_q_dq, //
			ConstVec q,         //
			ConstVec v          //
	) const;

	void d_integrate_dv(    //
			MutColMat out_q_dv, //
			ConstVec q,         //
			ConstVec v          //
	) const;

	void difference(      //
			MutVec out_v,     //
			ConstVec q_start, //
			ConstVec q_finish //
	) const;

	void d_difference_dq_start(   //
			MutColMat out_v_dq_start, //
			ConstVec q_start,         //
			ConstVec q_finish         //
	) const;

	void d_difference_dq_finish(   //
			MutColMat out_v_dq_finish, //
			ConstVec q_start,          //
			ConstVec q_finish          //
	) const;

	// empty fext_gen means no forces
	// clang-format off
  auto dynamics_aba(                                                //
      MutVec out_acceleration,                                     //
      ConstVec q,                                                  //
      ConstVec v,                                                  //
      ConstVec tau,                                                // use a generator for `fext' because
      Option<FnView<Option<Tuple<i64, Force<T>>>()>> fext_gen, // pinocchio's aligned vector, and ForceTpl
      Key k                                                         // are not visible
  ) const -> Key;
	// clang-format on

	auto inverse_dynamics_rnea( //
			MutVec out_tau,         //
			ConstVec q,             //
			ConstVec v,             //
			ConstVec a,             //
			Option<FnView<Option<Tuple<i64, Force<T>>>()>>
					fext_gen, // pinocchio's aligned vector, and ForceTpl
			Key k         //
	) const -> Key;

	auto d_dynamics_aba(                 //
			MutColMat out_acceleration_dq,   //
			MutColMat out_acceleration_dv,   //
			MutColMat out_acceleration_dtau, //
			MutVec out_acceleration,         //
			ConstVec q,                      //
			ConstVec v,                      //
			ConstVec tau,                    //
			Option<FnView<Option<Tuple<i64, Force<T>>>()>>
					fext_gen, // pinocchio's aligned vector, and ForceTpl
			Key k         //
	) const -> Key;

	auto d_inverse_dynamics_rnea( //
			MutColMat out_tau_dq,     //
			MutColMat out_tau_dv,     //
			MutColMat out_tau_da,     //
			ConstVec q,               //
			ConstVec v,               //
			ConstVec a,               //
			Option<FnView<Option<Tuple<i64, Force<T>>>()>>
					fext_gen, // pinocchio's aligned vector, and ForceTpl
			Key k         //
	) const -> Key;

	auto
	compute_forward_kinematics(ConstVec q, ConstVec v, ConstVec tau, Key k) const
			-> Key;

	auto compute_forward_dynamics(
			ConstVec q,
			ConstVec v,
			ConstVec tau,
			ConstColMat J,
			ConstVec gamma,
			T inv_damping,
			Key k) const -> Key;

	auto set_ddq(MutVec a, Key k) const -> Key;

	auto compute_frames_forward_kinematics(ConstVec q, Key k) const -> Key;

	auto compute_joint_jacobians(ConstVec q, Key k) const -> Key;
	auto frame_se3(i64 frame_id, Key k) const -> Tuple<Key, Se3<T>>;
	auto d_frame_se3(MutColMat out, i64 frame_id, Key k) const -> Key;

	auto frame_classical_acceleration(i64 frame_id, Key k) const
			-> Tuple<Key, Motion<T>>;

	auto frame_velocity(i64 frame_id, Key k) const -> Tuple<Key, Motion<T>>;
	auto d_frame_acceleration(
			MutColMat dv_dq,
			MutColMat da_dq,
			MutColMat da_dv,
			MutColMat da_da,
			i64 frame_id,
			Key k) const -> Key;

	auto frames_forward_kinematics(ConstVec q, Key k) const -> Key;

	auto n_frames() const -> i64;
	auto frame_name(i64 i) const -> ::fmt::string_view;

	auto
	kkt_contact_dynamics_matrix_inverse(MutColMat out, ConstColMat J, Key k) const
			-> Key;

	static auto skew(ConstVec v) -> SkewMat<T>;
};

} // namespace pinocchio
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_PINOCCHIO_MODEL_HPP_IRYFG86GS */
