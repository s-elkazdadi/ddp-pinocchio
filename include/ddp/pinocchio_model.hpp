#ifndef DDP_PINOCCHIO_PINOCCHIO_MODEL_HPP_IRYFG86GS
#define DDP_PINOCCHIO_PINOCCHIO_MODEL_HPP_IRYFG86GS

#include "ddp/internal/utils.hpp"
#include "ddp/internal/eigen.hpp"
#include <memory>

namespace ddp {
namespace pinocchio {

template <typename T>
struct skew_mat {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 9> data;
};

template <typename T>
struct se3 {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 9> rot /* 3x3 col major */;
	Array<T, 3> trans;
};
template <typename T>
struct motion {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 3> angular;
	Array<T, 3> linear;
};
template <typename T>
struct force {
	DDP_CHECK_CONCEPT(scalar<T>);
	Array<T, 3> angular;
	Array<T, 3> linear;
};

namespace internal {
struct pinocchio_impl;
} // namespace internal

template <typename T>
struct model {
	DDP_CHECK_CONCEPT(scalar<T>);

	using scalar = T;

	using mut_vec = view<T, colvec>;
	using const_vec = view<T const, colvec>;

	using mut_colmat = view<T, colmat>;
	using const_colmat = view<T const, colmat>;

private:
	struct model_impl;
	struct data_impl;
	friend struct internal::pinocchio_impl;
	model() = default;

	struct layout {
		model_impl* model = nullptr;
		data_impl* data = nullptr;
		i64 num_data = 0;
		i64 config_dim = 0;
		i64 tangent_dim = 0;
	} self;

public:
	struct key {
		key() = default;

		key(key const&) = delete;
		key(key&& other) noexcept : parent{other.parent} { other.parent = nullptr; }
		auto operator=(key const&) -> key& = delete;
		auto operator=(key&& other) noexcept -> key& {
			VEG_DEBUG_ASSERT("no wasting keys :<", !(*this));

			destroy();
			parent = other.parent;
			other.parent = nullptr;
			return *this;
		}
		~key() { destroy(); }

		explicit operator bool() const { return parent != nullptr; }

	private:
		void destroy();
		friend struct model;
		explicit key(data_impl& ref) : parent{&ref} {};
		data_impl* parent = nullptr;
	};

	~model();
	model(model const&) = delete;
	model(model&&) noexcept;

	auto operator=(model const&) -> model& = delete;
	auto operator=(model&&) -> model& = delete;

	auto model_name() const -> ::fmt::string_view;

	explicit model(
			::fmt::string_view urdf_path,
			i64 n_parallel = 1,
			bool add_freeflyer_base = false);
	static auto all_joints_test_model(i64 n_parallel = 1) -> model;

	auto config_dim() const -> i64 { return self.config_dim; }
	auto tangent_dim() const -> i64 { return self.tangent_dim; }

	auto acquire_workspace() const -> key;

	void neutral_configuration(mut_vec out_q) const;
	void random_configuration(mut_vec out_q) const;

	void integrate(    //
			mut_vec out_q, //
			const_vec q,   //
			const_vec v    //
	) const;

	void d_integrate_dq(     //
			mut_colmat out_q_dq, //
			const_vec q,         //
			const_vec v          //
	) const;

	void d_integrate_dv(     //
			mut_colmat out_q_dv, //
			const_vec q,         //
			const_vec v          //
	) const;

	void difference(       //
			mut_vec out_v,     //
			const_vec q_start, //
			const_vec q_finish //
	) const;

	void d_difference_dq_start(    //
			mut_colmat out_v_dq_start, //
			const_vec q_start,         //
			const_vec q_finish         //
	) const;

	void d_difference_dq_finish(    //
			mut_colmat out_v_dq_finish, //
			const_vec q_start,          //
			const_vec q_finish          //
	) const;

	// empty fext_gen means no forces
	// clang-format off
  auto dynamics_aba(                                                //
      mut_vec out_acceleration,                                     //
      const_vec q,                                                  //
      const_vec v,                                                  //
      const_vec tau,                                                // use a generator for `fext' because
      Option<fn::FnView<Option<Tuple<i64, force<T>>>()>> fext_gen, // pinocchio's aligned vector, and ForceTpl
      key k                                                         // are not visible
  ) const -> key;
	// clang-format on

	auto inverse_dynamics_rnea( //
			mut_vec out_tau,        //
			const_vec q,            //
			const_vec v,            //
			const_vec a,            //
			Option<fn::FnView<Option<Tuple<i64, force<T>>>()>>
					fext_gen, // pinocchio's aligned vector, and ForceTpl
			key k         //
	) const -> key;

	auto d_dynamics_aba(                  //
			mut_colmat out_acceleration_dq,   //
			mut_colmat out_acceleration_dv,   //
			mut_colmat out_acceleration_dtau, //
			mut_vec out_acceleration,         //
			const_vec q,                      //
			const_vec v,                      //
			const_vec tau,                    //
			Option<fn::FnView<Option<Tuple<i64, force<T>>>()>>
					fext_gen, // pinocchio's aligned vector, and ForceTpl
			key k         //
	) const -> key;

	auto d_inverse_dynamics_rnea( //
			mut_colmat out_tau_dq,    //
			mut_colmat out_tau_dv,    //
			mut_colmat out_tau_da,    //
			const_vec q,              //
			const_vec v,              //
			const_vec a,              //
			Option<fn::FnView<Option<Tuple<i64, force<T>>>()>>
					fext_gen, // pinocchio's aligned vector, and ForceTpl
			key k         //
	) const -> key;

	auto compute_forward_kinematics(
			const_vec q, const_vec v, const_vec tau, key k) const -> key;

	auto compute_forward_dynamics(
			const_vec q,
			const_vec v,
			const_vec tau,
			const_colmat J,
			const_vec gamma,
			T inv_damping,
			key k) const -> key;

	auto set_ddq(mut_vec a, key k) const -> key;

	auto compute_frames_forward_kinematics(const_vec q, key k) const -> key;

	auto compute_joint_jacobians(const_vec q, key k) const -> key;
	auto frame_se3(i64 frame_id, key k) const -> Tuple<key, se3<T>>;
	auto d_frame_se3(mut_colmat out, i64 frame_id, key k) const -> key;

	auto frame_classical_acceleration(i64 frame_id, key k) const
			-> Tuple<key, motion<T>>;

	auto frame_velocity(i64 frame_id, key k) const -> Tuple<key, motion<T>>;
	auto d_frame_acceleration(
			mut_colmat dv_dq,
			mut_colmat da_dq,
			mut_colmat da_dv,
			mut_colmat da_da,
			i64 frame_id,
			key k) const -> key;

	auto frames_forward_kinematics(const_vec q, key k) const -> key;

	auto n_frames() const -> i64;
	auto frame_name(i64 i) const -> ::fmt::string_view;

	auto kkt_contact_dynamics_matrix_inverse(
			mut_colmat out, const_colmat J, key k) const -> key;

	static auto skew(const_vec v) -> skew_mat<T>;
};

} // namespace pinocchio
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_PINOCCHIO_MODEL_HPP_IRYFG86GS */
