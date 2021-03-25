#ifndef DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS
#define DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS

#include "ddp/internal/tensor.hpp"
#include "ddp/internal/second_order_finite_diff.hpp"
#include "ddp/pinocchio_model.hpp"
#include "ddp/space.hpp"
#include "veg/internal/prologue.hpp"

namespace ddp {

template <typename T>
struct lqr_dynamics {
	eigen::matrix<T, colmat> a;
	eigen::matrix<T, colmat> b;
	eigen::matrix<T, colvec> c;

	struct key {
		explicit operator bool() const { return true; }
	};
	using scalar = T;

	auto acquire_workspace() const -> key { return {}; }
	void neutral_configuration(view<T, colvec> q) const { q.setZero(); }
	void random_configuration(view<T, colvec> q) const { q.setRandom(); }

	auto state_space() const -> vector_space<T> { return {{a.cols()}}; }
	auto output_space() const -> vector_space<T> { return {{a.rows()}}; }
	auto control_space() const -> vector_space<T> { return {{b.cols()}}; }

	auto eval_to_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	auto eval_to(
			view<T, colvec> f_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> u,
			key k,
			dynamic_stack_view stack) const -> key {

		(void)stack;

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(f_out, x)), //
				(!eigen::aliases(f_out, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(f_out.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		(void)t;
		eigen::assign(f_out, c);
		eigen::mul_add_to_noalias(f_out, a, x);
		eigen::mul_add_to_noalias(f_out, b, u);
		return k;
	}

	auto d_eval_to_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	auto d_eval_to(
			view<T, colmat> fx,
			view<T, colmat> fu,
			view<T, colvec> f,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> u,
			key k,
			dynamic_stack_view stack) const -> key {

		(void)t, (void)stack;

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
	using ref_type = lqr_dynamics const&;
};

template <typename T>
struct pinocchio_dynamics_free {
	pinocchio::model<T> const& model;
	T dt;
	bool can_use_first_order_diff;

	using scalar = T;
	using key = typename pinocchio::model<T>::key;

	auto acquire_workspace() const -> key { return model.acquire_workspace(); }

	void neutral_configuration(view<T, colvec> x) const {
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, model.config_dim()));

		model.neutral_configuration(q);
		v.setZero();
	}
	void random_configuration(view<T, colvec> x) const {
		VEG_BIND(auto, (q, v), eigen::split_at_row(x, model.config_dim()));
		model.random_configuration(q);
		v.setRandom();
	}

	auto state_space() const -> pinocchio_state_space<T> { return {model}; }
	auto output_space() const -> pinocchio_state_space<T> { return {model}; }
	auto control_space() const -> vector_space<T> {
		return {model.tangent_dim()};
	}

	auto eval_to_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	auto eval_to(
			view<T, colvec> f_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> u,
			key k,
			dynamic_stack_view stack) const -> key {

		(void)stack;

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(f_out, x)), //
				(!eigen::aliases(f_out, u)),
				(k));

		VEG_DEBUG_ASSERT_ALL_OF(
				(f_out.rows() == state_space().dim(t)),
				(x.rows() == state_space().dim(t)),
				(u.rows() == control_space().dim(t)));

		(void)t;
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

	auto d_eval_to_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	auto d_eval_to(
			view<T, colmat> fx,
			view<T, colmat> fu,
			view<T, colvec> f,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> u,
			key k,
			dynamic_stack_view stack) const -> key {

		(void)t, (void)stack;

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

	using ref_type = pinocchio_dynamics_free;
};

namespace make {
namespace fn {
struct pinocchio_dynamics_free {
	VEG_TEMPLATE(
			(typename T),
			requires(true),
			auto
			operator(),
			(model, pinocchio::model<T> const&),
			(dt, typename meta::type_identity<T>::type),
			(can_use_first_order_diff = false, bool))
	const->ddp::pinocchio_dynamics_free<T> {
		VEG_ASSERT_ELSE("unimplemented", !can_use_first_order_diff);
		return {model, dt, can_use_first_order_diff};
	}
};

struct lqr_dynamics {
	VEG_TEMPLATE(
			(typename A, typename B, typename C),
			requires true,
			auto
			operator(),
			(a, A const&),
			(b, B const&),
			(c, C const&))
	const->ddp::lqr_dynamics<typename A::Scalar> {
		using T = typename A::Scalar;
		return {
				eigen::matrix<T, colmat>(a),
				eigen::matrix<T, colmat>(b),
				eigen::matrix<T, colvec>(c),
		};
	}
};
} // namespace fn
VEG_INLINE_VAR(lqr_dynamics, fn::lqr_dynamics);
VEG_INLINE_VAR(pinocchio_dynamics_free, fn::pinocchio_dynamics_free);
} // namespace make

} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_DYNAMICS_HPP_WJXL0MGOS */
