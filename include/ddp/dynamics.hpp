#ifndef DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS
#define DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS

#include "ddp/internal/tensor.hpp"
#include "ddp/internal/second_order_finite_diff.hpp"
#include "ddp/pinocchio_model.hpp"
#include "ddp/space.hpp"
#include "veg/internal/prologue.hpp"

namespace ddp {

namespace internal {
namespace meta {
template <typename T>
using key_type_t = typename T::Key;
template <typename T>
using acquire_workspace_expr =
		decltype(VEG_DECLVAL(T const&).acquire_workspace());
template <typename T>
using neutral_configuration_expr =
		decltype(VEG_DECLVAL(T const&).neutral_configuration(
				VEG_DECLVAL(View<scalar_type_t<T>, colvec>)));
template <typename T>
using random_configuration_expr =
		decltype(VEG_DECLVAL(T const&).random_configuration(
				VEG_DECLVAL(View<scalar_type_t<T>, colvec>)));

template <typename T>
using state_space_expr = decltype(VEG_DECLVAL(T const&).state_space());
template <typename T>
using control_space_expr = decltype(VEG_DECLVAL(T const&).control_space());
template <typename T>
using output_space_expr = decltype(VEG_DECLVAL(T const&).output_space());

template <typename T>
using eval_to_expr = decltype(VEG_DECLVAL(T const&).eval_to(

		VEG_DECLVAL(View<scalar_type_t<T>, colvec>),
		VEG_DECLVAL(i64),
		VEG_DECLVAL(View<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(View<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(key_type_t<T>&&),
		VEG_DECLVAL(DynStackView)

				));

template <typename T>
using deval_to_expr = decltype(VEG_DECLVAL(T const&).eval_to(

		VEG_DECLVAL(View<scalar_type_t<T>, colmat>),
		VEG_DECLVAL(View<scalar_type_t<T>, colmat>),
		VEG_DECLVAL(View<scalar_type_t<T>, colvec>),
		VEG_DECLVAL(i64),
		VEG_DECLVAL(View<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(View<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(key_type_t<T>&&),
		VEG_DECLVAL(DynStackView)

				));

template <typename T>
using req_eval_to_expr = decltype(VEG_DECLVAL(T const&).eval_to_req());
template <typename T>
using req_deval_to_expr = decltype(VEG_DECLVAL(T const&).d_eval_to_req());
} // namespace meta
} // namespace internal

namespace dynamics {
template <typename T>
struct Lqr {
	eigen::Matrix<T, colmat> a;
	eigen::Matrix<T, colmat> b;
	eigen::Matrix<T, colvec> c;

	struct Key {
		explicit operator bool() const { return true; }
	};
	using Scalar = T;

	auto acquire_workspace() const -> Key { return {}; }
	void neutral_configuration(View<T, colvec> q) const { q.setZero(); }
	void random_configuration(View<T, colvec> q) const { q.setRandom(); }

	auto state_space() const -> VectorSpace<T> {
		return VectorSpace<T>{a.cols()};
	}
	auto output_space() const -> VectorSpace<T> {
		return VectorSpace<T>{a.rows()};
	}
	auto control_space() const -> VectorSpace<T> {
		return VectorSpace<T>{b.cols()};
	}

	auto eval_to_req() const -> MemReq {
		unused(this);
		return {tag<T>, 0};
	}
	auto eval_to(
			View<T, colvec> f_out,
			i64 t,
			View<T const, colvec> x,
			View<T const, colvec> u,
			Key k,
			DynStackView stack) const -> Key {

		unused(t, stack);

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(f_out, x)), //
				(!eigen::aliases(f_out, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(f_out.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		eigen::assign(f_out, c);
		eigen::mul_add_to_noalias(f_out, a, x);
		eigen::mul_add_to_noalias(f_out, b, u);
		return k;
	}

	auto d_eval_to_req() const -> MemReq {
		unused(this);
		return {tag<T>, 0};
	}
	auto d_eval_to(
			View<T, colmat> fx,
			View<T, colmat> fu,
			View<T, colvec> f,
			i64 t,
			View<T const, colvec> x,
			View<T const, colvec> u,
			Key k,
			DynStackView stack) const -> Key {

		unused(t, stack);

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(fx, fu, f, x, u)), //
				(!eigen::aliases(fu, f, x, u)),
				(!eigen::aliases(f, x, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(fx.rows() == state_space().ddim(t)),
				(fx.cols() == state_space().ddim(t)),
				(fu.rows() == state_space().ddim(t)),
				(fu.cols() == control_space().ddim(t)),
				(f.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		eigen::assign(f, c);
		eigen::mul_add_to_noalias(f, a, x);
		eigen::mul_add_to_noalias(f, b, u);
		eigen::assign(fx, a);
		eigen::assign(fu, b);

		return k;
	}
	using Ref = Lqr const&;
};

template <typename T>
struct PinocchioFree {
	pinocchio::Model<T> const& model;
	T dt;
	bool can_use_first_order_diff;

	using Scalar = T;
	using Key = typename pinocchio::Model<T>::Key;

	auto acquire_workspace() const -> Key { return model.acquire_workspace(); }

	void neutral_configuration(View<T, colvec> x) const {
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, model.config_dim()));

		model.neutral_configuration(q);
		v.setZero();
	}
	void random_configuration(View<T, colvec> x) const {
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, model.config_dim()));
		model.random_configuration(q);
		v.setRandom();
	}

	auto state_space() const -> pinocchio_state_space<T> { return {model}; }
	auto output_space() const -> pinocchio_state_space<T> { return {model}; }
	auto control_space() const -> VectorSpace<T> {
		return VectorSpace<T>{model.tangent_dim()};
	}

	auto eval_to_req() const -> MemReq {
		unused(this);
		return {tag<T>, 0};
	}
	auto eval_to(
			View<T, colvec> f_out,
			i64 t,
			View<T const, colvec> x,
			View<T const, colvec> u,
			Key k,
			DynStackView stack) const -> Key {

		unused(t, stack);

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(f_out, x)), //
				(!eigen::aliases(f_out, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(f_out.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		auto nq = model.config_dim();

		VEG_BIND(auto, (q_out, v_out), eigen::split_at_row(f_out, nq));
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));

		// v_out = dt * v_in
		eigen::mul_scalar_to(v_out, v, dt);

		// q_out = q_in + v_out
		//       = q_in + dt * v_in
		model.integrate(q_out, q, eigen::as_const(v_out));

		// v_out = acc
		k = model.dynamics_aba(v_out, q, v, u, none, VEG_FWD(k));

		// v_out = v_in + dt * v_out
		//       = v_in + dt * acc
		eigen::mul_scalar_to(v_out, v_out, dt);
		eigen::add_to(v_out, v_out, v);
		return k;
	}

	auto d_eval_to_req() const -> MemReq {
		unused(this);
		return {tag<T>, 0};
	}
	auto d_eval_to(
			View<T, colmat> fx,
			View<T, colmat> fu,
			View<T, colvec> f,
			i64 t,
			View<T const, colvec> x,
			View<T const, colvec> u,
			Key k,
			DynStackView stack) const -> Key {

		unused(t, stack);

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(fx, fu, f, x, u)), //
				(!eigen::aliases(fu, f, x, u)),
				(!eigen::aliases(f, x, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(fx.rows() == state_space().ddim(t)),
				(fx.cols() == state_space().ddim(t)),
				(fu.rows() == state_space().ddim(t)),
				(fu.cols() == control_space().ddim(t)),
				(f.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		auto nq = model.config_dim();
		auto nv = model.tangent_dim();

		VEG_BIND(auto, (q_out, v_out), eigen::split_at_row(f, nq));
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));
		VEG_BIND(auto, (fx_qq, fx_qv, fx_vq, fx_vv), eigen::split_at(fx, nv, nv));
		VEG_BIND(auto, (fu_q, fu_v), eigen::split_at_row(fu, nv));

		// v_out = dt * v_in;
		eigen::mul_scalar_to(v_out, v, dt);

		// q_out = q_in + dt * v_in
		fx.setZero();
		model.integrate(q_out, q, eigen::as_const(v_out));
		model.d_integrate_dq(fx_qq, q, eigen::as_const(v_out));
		model.d_integrate_dv(fx_qv, q, eigen::as_const(v_out));
		eigen::mul_scalar_to(fx_qv, fx_qv, dt);

		// v_out = acc
		fu_q.setZero();
		k = model.d_dynamics_aba(
				fx_vq, fx_vv, fu_v, v_out, q, v, u, none, VEG_FWD(k));

		// v_out = v_in + dt * v_out
		//       = v_in + dt * acc
		eigen::mul_scalar_to(v_out, v_out, dt);
		eigen::add_to(v_out, v_out, v);

		eigen::mul_scalar_to(fx_vq, fx_vq, dt);
		eigen::mul_scalar_to(fx_vv, fx_vv, dt);
		eigen::add_identity(fx_vv);

		eigen::mul_scalar_to(fu_v, fu_v, dt);

		return k;
	}

	using Ref = PinocchioFree;
};

template <typename T>
struct PinocchioContact { /* NOLINT */
	pinocchio::Model<T> const& model;
	std::vector<i64> frame_ids;
	T dt;
	bool can_use_first_order_diff;

	using Scalar = T;
	using Key = typename pinocchio::Model<T>::Key;

	auto acquire_workspace() const -> Key { return model.acquire_workspace(); }

	void neutral_configuration(View<T, colvec> x) const {
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, model.config_dim()));

		model.neutral_configuration(q);
		v.setZero();
	}
	void random_configuration(View<T, colvec> x) const {
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, model.config_dim()));
		model.random_configuration(q);
		v.setRandom();
	}

	auto state_space() const -> pinocchio_state_space<T> { return {model}; }
	auto output_space() const -> pinocchio_state_space<T> { return {model}; }
	auto control_space() const -> VectorSpace<T> {
		return VectorSpace<T>{model.tangent_dim()};
	}

	auto eval_to_req() const -> MemReq {
		i64 nv = model.tangent_dim();
		return {
				tag<T>,
				nv + i64(3U * frame_ids.size()) * (nv + 1) + 6 * nv,
		};
	}
	auto eval_to(
			View<T, colvec> f_out,
			i64 t,
			View<T const, colvec> x,
			View<T const, colvec> u,
			Key k,
			DynStackView stack) const -> Key {

		unused(t);
		Slice<i64 const> frames = {as_ref, frame_ids};

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(f_out, x)), //
				(!eigen::aliases(f_out, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(f_out.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		auto nq = model.config_dim();
		auto nv = model.tangent_dim();

		VEG_BIND(auto, (q_out, v_out), eigen::split_at_row(f_out, nq));
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));

		// v_out = dt * v_in
		eigen::mul_scalar_to(v_out, v, dt);

		// q_out = q_in + v_out
		//       = q_in + dt * v_in
		model.integrate(q_out, q, eigen::as_const(v_out));

		DDP_TMP_VECTOR(stack, _u_full, T, nv);

		{
			VEG_BIND(auto, (_u_full_0, _u_full_u), eigen::split_at_row(_u_full, 6));
			unused(_u_full_0);
			eigen::assign(_u_full_u, u);
		}

		auto u_full = eigen::as_const(_u_full);
		{
			i64 n_frames = frames.size();
			k = model.compute_forward_kinematics(q, v, u_full, VEG_FWD(k));
			k = model.compute_joint_jacobians(q, VEG_FWD(k));

			DDP_TMP_MATRIX(stack, j_constraint, T, 3 * n_frames, nv);
			DDP_TMP_VECTOR(stack, gamma_constraint, T, 3 * n_frames);

			for (i64 i = 0; i < n_frames; ++i) {
				DDP_TMP_MATRIX(stack, j, T, 6, nv);
				k = model.d_frame_se3(eigen::as_mut(j), frames[i], VEG_FWD(k));
				auto res = model.frame_classical_acceleration(frames[i], VEG_FWD(k));
				k = VEG_FWD(res)[0_c];
				pinocchio::Motion<T> m = VEG_FWD(res)[1_c];

				VEG_BIND(
						auto,
						(ignore_j_top, j_bot),
						eigen::split_at_row(j_constraint, 3 * i));
				VEG_BIND( //
						auto,
						(j_current, ignore_j_bot),
						eigen::split_at_row(j_bot, 3));

				VEG_BIND(
						auto,
						(ignore_g_top, g_bot),
						eigen::split_at_row(gamma_constraint, 3 * i));
				VEG_BIND( //
						auto,
						(g_current, ignore_g_bot),
						eigen::split_at_row(g_bot, 3));

				j_current = eigen::split_at_row(j, 3)[0_c];
				g_current = eigen::slice_to_vec(m.linear);
			}

			k = model.compute_forward_dynamics(
					q,
					v,
					u_full,
					eigen::as_const(j_constraint),
					eigen::as_const(gamma_constraint),
					T(0),
					VEG_FWD(k));

			k = model.set_ddq(v_out, VEG_FWD(k));
		}

		eigen::mul_scalar_to(v_out, v_out, dt);
		eigen::add_to(v_out, v_out, v);
		return k;
	}

	auto d_eval_to_req() const -> MemReq {
		unused(this);
		return {tag<T>, 0};
	}
	auto d_eval_to(
			View<T, colmat> fx,
			View<T, colmat> fu,
			View<T, colvec> f,
			i64 t,
			View<T const, colvec> x,
			View<T const, colvec> u,
			Key k,
			DynStackView stack) const -> Key {

		unused(t);

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(fx, fu, f, x, u)), //
				(!eigen::aliases(fu, f, x, u)),
				(!eigen::aliases(f, x, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(fx.rows() == state_space().ddim(t)),
				(fx.cols() == state_space().ddim(t)),
				(fu.rows() == state_space().ddim(t)),
				(fu.cols() == control_space().ddim(t)),
				(f.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		auto nq = model.config_dim();
		auto nv = model.tangent_dim();

		DDP_TMP_VECTOR(stack, _u_full, T, nv);
		DDP_TMP_MATRIX(stack, _fu_full, T, nv, nv);
		{
			VEG_BIND(auto, (u_full_0, u_full_u), eigen::split_at_row(_u_full, 6));
			VEG_BIND(auto, (fu_full_0, fu_full_u), eigen::split_at_row(_fu_full, 6));
			unused(u_full_0, fu_full_0);
			u_full_u = u;
		}
		auto u_full = eigen::as_const(_u_full);
		auto fu_full = eigen::as_mut(_fu_full);

		VEG_BIND(auto, (q_out, v_out), eigen::split_at_row(f, nq));
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));
		VEG_BIND(auto, (fx_qq, fx_qv, fx_vq, fx_vv), eigen::split_at(fx, nv, nv));
		VEG_BIND(auto, (fu_q, fu_v), eigen::split_at_row(fu, nv));

		// v_out = dt * v_in;
		eigen::mul_scalar_to(v_out, v, dt);

		// q_out = q_in + dt * v_in
		fx.setZero();
		model.integrate(q_out, q, eigen::as_const(v_out));
		model.d_integrate_dq(fx_qq, q, eigen::as_const(v_out));
		model.d_integrate_dv(fx_qv, q, eigen::as_const(v_out));
		eigen::mul_scalar_to(fx_qv, fx_qv, dt);

		// v_out = acc
		fu_q.setZero();
		k = model.d_dynamics_aba(
				fx_vq, fx_vv, fu_v, v_out, q, v, u, none, VEG_FWD(k));

		// v_out = v_in + dt * v_out
		//       = v_in + dt * acc
		eigen::mul_scalar_to(v_out, v_out, dt);
		eigen::add_to(v_out, v_out, v);

		eigen::mul_scalar_to(fx_vq, fx_vq, dt);
		eigen::mul_scalar_to(fx_vv, fx_vv, dt);
		eigen::add_identity(fx_vv);

		eigen::mul_scalar_to(fu_v, fu_v, dt);

		return k;
	}

	using Ref = PinocchioContact const&;
};

namespace nb {
struct pinocchio_free {
	VEG_TEMPLATE(
			(typename T),
			requires(true),
			auto
			operator(),
			(model, pinocchio::Model<T> const&),
			(dt, typename meta::type_identity<T>::type),
			(can_use_first_order_diff = false, bool))
	const->PinocchioFree<T> {
		VEG_ASSERT_ELSE("unimplemented", !can_use_first_order_diff);
		return {model, dt, can_use_first_order_diff};
	}
};

struct lqr {
	VEG_TEMPLATE(
			(typename A, typename B, typename C),
			requires true,
			auto
			operator(),
			(a, A const&),
			(b, B const&),
			(c, C const&))
	const->Lqr<typename A::Scalar> {
		using T = typename A::Scalar;
		return {
				eigen::Matrix<T, colmat>(a),
				eigen::Matrix<T, colmat>(b),
				eigen::Matrix<T, colvec>(c),
		};
	}
};
} // namespace nb
VEG_NIEBLOID(lqr);
VEG_NIEBLOID(pinocchio_free);
} // namespace dynamics

} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS */
