#ifndef DDP_PINOCCHIO_DERIVATIVE_STORAGE_HPP_LLE0WSK3S
#define DDP_PINOCCHIO_DERIVATIVE_STORAGE_HPP_LLE0WSK3S

#include "ddp/internal/second_order_finite_diff.hpp"
#include "ddp/internal/tensor.hpp"
#include "ddp/cost.hpp"
#include "ddp/dynamics.hpp"
#include "ddp/constraint.hpp"
#include "ddp/trajectory.hpp"
#include "ddp/internal/thread_pool.hpp"
#include "veg/util/timer.hpp"
#include <omp.h>
#include "veg/internal/prologue.hpp"

namespace ddp {
namespace internal {
template <typename T>
struct StorageFor {
	alignas(T) unsigned char buf[sizeof(T)];
};

template <
		typename Out,
		typename XX,
		typename UX,
		typename UU,
		typename DX,
		typename DU>
void add_second_order_term(
		Out&& out,
		XX const& xx,
		UX const& ux,
		UU const& uu,
		DX const& dx,
		DU const& du) {
	VEG_DEBUG_ASSERT_ALL_OF(
			out.cols() == 1,
			out.rows() == xx.outdim(),
			out.rows() == ux.outdim(),
			out.rows() == uu.outdim(),
			dx.rows() == xx.indiml(),
			dx.rows() == xx.indimr(),
			dx.rows() == ux.indimr(),
			du.rows() == uu.indiml(),
			du.rows() == uu.indimr(),
			du.rows() == ux.indiml());

	for (i64 j = 0; j < dx.rows(); ++j) {
		for (i64 i = 0; i < dx.rows(); ++i) {
			for (i64 k = 0; k < out.rows(); ++k) {
				out(k) += 0.5 * dx(i) * xx(k, i, j) * dx(j);
			}
		}
	}

	for (i64 j = 0; j < du.rows(); ++j) {
		for (i64 i = 0; i < du.rows(); ++i) {
			for (i64 k = 0; k < out.rows(); ++k) {
				out(k) += 0.5 * du(i) * uu(k, i, j) * du(j);
			}
		}
	}

	for (i64 j = 0; j < dx.rows(); ++j) {
		for (i64 i = 0; i < du.rows(); ++i) {
			for (i64 k = 0; k < out.rows(); ++k) {
				out(k) += du(i) * ux(k, i, j) * dx(j);
			}
		}
	}
}

#define DDP_ACCESS(name)                                                       \
	auto name(i64 t)& { return self.name[t]; }                                   \
	auto name(i64 t) const& { return self.name[t]; }                             \
	auto name(i64 t)&& = delete

template <typename T>
struct FirstOrderDerivatives {
	using Scalar = T;

	struct layout {
		eigen::HeapMatrix<T, colvec> lfx;

		internal::MatSeq<T, colvec> lx;
		internal::MatSeq<T, colvec> lu;

		internal::MatSeq<T, colvec> f;
		internal::MatSeq<T, colmat> fx;
		internal::MatSeq<T, colmat> fu;

		internal::MatSeq<T, colvec> eq;
		internal::MatSeq<T, colmat> eqx;
		internal::MatSeq<T, colmat> equ;
	} self;

	auto lfx() const& { return self.lfx.get(); }
	auto lfx() & { return self.lfx.mut(); }

	DDP_ACCESS(lx);
	DDP_ACCESS(lu);

	DDP_ACCESS(f);
	DDP_ACCESS(fx);
	DDP_ACCESS(fu);

	DDP_ACCESS(eq);
	DDP_ACCESS(eqx);
	DDP_ACCESS(equ);
};

template <typename T>
struct SecondOrderDerivatives : FirstOrderDerivatives<T> {
	using Scalar = T;

	struct layout {
		eigen::HeapMatrix<T, colmat> lfxx;

		internal::MatSeq<T, colmat> lxx;
		internal::MatSeq<T, colmat> lux;
		internal::MatSeq<T, colmat> luu;

		internal::TensorSeq<T> fxx;
		internal::TensorSeq<T> fux;
		internal::TensorSeq<T> fuu;

		internal::TensorSeq<T> eqxx;
		internal::TensorSeq<T> equx;
		internal::TensorSeq<T> equu;
	} self;

	SecondOrderDerivatives(FirstOrderDerivatives<T> base, layout self_)
			: FirstOrderDerivatives<T>(VEG_FWD(base)), self(VEG_FWD(self_)) {}

	auto lfxx() const& { return self.lfxx.get(); }
	auto lfxx() & { return self.lfxx.mut(); }

	DDP_ACCESS(lxx);
	DDP_ACCESS(lux);
	DDP_ACCESS(luu);

	DDP_ACCESS(fxx);
	DDP_ACCESS(fux);
	DDP_ACCESS(fuu);

	DDP_ACCESS(eqxx);
	DDP_ACCESS(equx);
	DDP_ACCESS(equu);
};

#undef DDP_ACCESS

template <typename Cost, typename Dynamics, typename Constraint>
auto compute_first_derivatives_req(
		Cost const& cost, Dynamics const& dynamics, Constraint const& constraint)
		-> MemReq {
	return MemReq::max_of({
			cost.d_eval_to_req(),
			dynamics.d_eval_to_req(),
			constraint.d_eval_to_req(),
	});
}

// single threaded
template <
		typename T,
		typename Cost,
		typename Dynamics,
		typename Constraint,
		typename Traj>
void compute_first_derivatives(
		FirstOrderDerivatives<T>& derivs,
		Cost const& cost,
		Dynamics const& dynamics,
		Constraint const& constraint,
		Traj const& traj,
		DynStackView stack) {

	cost.d_eval_final_to(derivs.lfx(), traj.x_f(), stack);

	i64 begin = derivs.self.lx.index_begin();
	i64 end = derivs.self.lx.index_end();

	for (i64 t = begin; t < end; ++t) {
		auto k = dynamics.acquire_workspace();
		VEG_BIND(auto, (x, u), traj[t]);

		cost.d_eval_to(derivs.lx(t), derivs.lu(t), t, x, u, stack);

		k = dynamics.d_eval_to(
				derivs.fx(t), derivs.fu(t), derivs.f(t), t, x, u, VEG_FWD(k), stack);

		k = constraint.d_eval_to(
				derivs.eqx(t), derivs.equ(t), derivs.eq(t), t, x, u, VEG_FWD(k), stack);
	}
}

constexpr auto _min2(i64 a, i64 b) noexcept -> i64 {
	return a < b ? a : b;
}

static constexpr i64 max_threads = 16;
inline auto n_threads() -> i64 {
	static const i64 result = (_min2)(max_threads, internal::get_num_procs());
	return result;
}

inline auto threaded_req(MemReq single_thread) noexcept -> MemReq {
	return {single_thread.align, single_thread.size * internal::n_threads()};
}

template <typename Cost, typename Dynamics, typename Constraint>
auto compute_second_derivatives_req(
		Cost const& cost, Dynamics const& dynamics, Constraint const& constraint)
		-> MemReq {
	return MemReq::max_of({
			as_ref,
			{
					cost.d_eval_to_req(),
					cost.dd_eval_to_req(),
					ddp::second_order_deriv_1_req(dynamics),
					ddp::second_order_deriv_1_req(constraint),
					ddp::second_order_deriv_2_req(dynamics),
					ddp::second_order_deriv_2_req(constraint),
			},
	});
}

template <typename Cost, typename Dynamics, typename Constraint>
auto compute_second_derivatives_test_req(
		Cost const& cost, Dynamics const& dynamics, Constraint const& constraint)
		-> MemReq {
	return MemReq::sum_of({
			as_ref,
			{
					{
							tag<typename Dynamics::Scalar>,
							(2 * dynamics.output_space().max_ddim()     // storage_f
	             + 2 * constraint.output_space().max_ddim() // storage_eq
	             + 6 * dynamics.output_space().max_ddim()   // storage_df
	             + 6 * constraint.output_space().max_ddim() // storage_deq
	                                                        //
	             + dynamics.output_space().max_ddim()       // storage_dx
	             + dynamics.control_space().max_ddim()      // storage_du

	             // inside finite_diff_err
	             + dynamics.output_space().max_ddim()  // storage_dx
	             + dynamics.control_space().max_ddim() // storage_du
	                                                   //
	             + dynamics.output_space().max_dim()   // storage_x
	             + dynamics.control_space().max_dim()  // storage_u
	                                                   //
	             + dynamics.output_space().max_dim()   // storage_dotx
	             + dynamics.control_space().max_dim()  // storage_dotu

	             ),
					},

					MemReq::max_of({
							as_ref,
							{
									dynamics.state_space().integrate_req(),     //
									dynamics.control_space().integrate_req(),   //
									dynamics.output_space().difference_req(),   //
									constraint.output_space().difference_req(), //
									cost.eval_req(),                            //
									dynamics.eval_to_req(),                     //
									constraint.eval_to_req(),                   //
							},
					}),
			},
	});
}

// multi threaded
template <typename T, typename Cost, typename Dynamics, typename Constraint>
auto compute_second_derivatives(
		SecondOrderDerivatives<T>& derivs,
		Cost const& cost,
		Dynamics const& dynamics,
		Constraint const& constraint,
		Trajectory<T> const& traj,
		DynStackView stack,
		bool multithread) -> i64 {
	fmt::print("");
	i64 nanosec{};
	using Key = decltype(dynamics.acquire_workspace());
	{
		auto&& _ = veg::time::raii_timer(
				[&](i64 duration) noexcept { nanosec = duration; });
		unused(_);

		auto const& _lx = derivs.FirstOrderDerivatives<T>::self.lx;
		i64 const begin = _lx.index_begin();
		i64 const end = _lx.index_end();

		if (!multithread) {
			auto k = dynamics.acquire_workspace();
			k = cost.d_eval_final_to(derivs.lfx(), traj.x_f(), VEG_FWD(k), stack);
			k = cost.dd_eval_final_to(derivs.lfxx(), traj.x_f(), VEG_FWD(k), stack);
			for (i64 t = begin; t < end; ++t) {
				k = compute_second_derivatives_one_iter(
						derivs, cost, dynamics, constraint, traj, t, VEG_FWD(k), stack);
			}
		} else {
			auto const n_threads = internal::n_threads();

			veg::Array<Option<DynStackArray<StorageFor<T>>>, max_threads>
					stack_buffers;

			{
				i64 stack_len = stack.remaining_bytes() / n_threads / i64(sizeof(T));

				for (i64 i = 0; i < n_threads; ++i) {
					stack_buffers._[i] = stack.make_new(tag<StorageFor<T>>, stack_len);
					VEG_ASSERT(stack_buffers._[i].is_some());
				}
			}

			struct State { /* NOLINT */
				DynStackView thread_stack;
				Key k;
			};

			veg::Array<Option<State>, max_threads> states;
			auto setup = [&](i64 id) -> void* {
				VEG_ASSERT(id < n_threads);
				states._[id] = {
						some,
						{
								{slice::from_range(stack_buffers._[id].as_ref().unwrap())},
								dynamics.acquire_workspace(),
						},
				};
				State& state = states._[id].as_ref().unwrap();
				auto&& k = static_cast<Key&&>(state.k);

				if (id == 0) {
					k = cost.d_eval_final_to(
							derivs.lfx(), traj.x_f(), VEG_FWD(k), state.thread_stack);
					k = cost.dd_eval_final_to(
							derivs.lfxx(), traj.x_f(), VEG_FWD(k), state.thread_stack);
				}

				return mem::addressof(state);
			};
			auto fn = [&](void* vstate, i64 id, i64 i) {
				(void)id;
				State& state = *static_cast<State*>(vstate);
				auto& thread_stack = state.thread_stack;
				auto&& k = VEG_FWD(state.k);

				k = compute_second_derivatives_one_iter(
						derivs,
						cost,
						dynamics,
						constraint,
						traj,
						i,
						VEG_FWD(k),
						thread_stack);
			};
			internal::parallel_for({as_ref, setup}, {as_ref, fn}, begin, end);

			// #ifdef DDP_OPENMP
			// #pragma omp parallel default(none) shared(stack_buffers) shared(derivs)
			// \
// 		shared(cost) shared(dynamics) shared(constraint) shared(traj) \
// 				shared(stack) shared(begin) shared(end) num_threads(n_threads)
			// #endif
			// 			{

			// 				auto k = dynamics.acquire_workspace();

			// 				i64 const thread_num = omp_get_thread_num();
			// 				VEG_ASSERT(thread_num < internal::n_threads());

			// 				DynStackView thread_stack = {slice::from_range(
			// 						stack_buffers._[omp_get_thread_num()].as_ref().unwrap())};

			// 				if (thread_num == 0) {
			// 					k = cost.d_eval_final_to(derivs.lfx(), traj.x_f(), VEG_FWD(k),
			// stack); 					k = cost.dd_eval_final_to( 							derivs.lfxx(),
			// traj.x_f(), VEG_FWD(k), stack);
			// 				}

			// #ifdef DDP_OPENMP
			// #pragma omp for
			// #endif
			// 				for (usize i = 0; i < narrow<usize>(end - begin); ++i) {
			// 					i64 t = begin + narrow<i64>(i);
			// 					k = compute_second_derivatives_one_iter(
			// 							derivs,
			// 							cost,
			// 							dynamics,
			// 							constraint,
			// 							traj,
			// 							t,
			// 							VEG_FWD(k),
			// 							thread_stack);
			// 				}
			// 			}
		}
	}
	veg::time::log_elapsed_time{"computing derivatives"}(nanosec);
	return nanosec;
}

template <typename T, typename Cost, typename Dynamics, typename Constraint>
auto compute_second_derivatives_one_iter(
		SecondOrderDerivatives<T>& derivs,
		Cost const& cost,
		Dynamics const& dynamics,
		Constraint const& constraint,
		Trajectory<T> const& traj,
		i64 t,
		typename Dynamics::Key k,
		DynStackView stack) -> typename Dynamics::Key {

	VEG_BIND(auto, (x, u), traj[t]);

	{
		k = cost.d_eval_to(derivs.lx(t), derivs.lu(t), t, x, u, VEG_FWD(k), stack);
		k = cost.dd_eval_to(
				derivs.lxx(t),
				derivs.lux(t),
				derivs.luu(t),
				t,
				x,
				u,
				VEG_FWD(k),
				stack);
	}
	{
		VEG_ASSERT(!eigen::aliases(derivs.fx(t), x));
		VEG_ASSERT(
				stack.remaining_bytes() >=
				ddp::second_order_deriv_1_req(dynamics).size);
		k = ddp::second_order_deriv_1(
				dynamics,
				derivs.fxx(t),
				derivs.fux(t),
				derivs.fuu(t),
				derivs.fx(t),
				derivs.fu(t),
				derivs.f(t),
				t,
				x,
				u,
				VEG_FWD(k),
				stack);
	}
	{
		k = ddp::second_order_deriv_1(
				constraint,
				derivs.eqxx(t),
				derivs.equx(t),
				derivs.equu(t),
				derivs.eqx(t),
				derivs.equ(t),
				derivs.eq(t),
				t,
				x,
				u,
				VEG_FWD(k),
				stack);
	}
#ifndef NDEBUG
	{
		VEG_ASSERT_ALL_OF( //
				!derivs.eq(t).hasNaN(),
				!derivs.eqx(t).hasNaN(),
				!derivs.equ(t).hasNaN(),
				!derivs.eqxx(t).has_nan(),
				!derivs.equx(t).has_nan(),
				!derivs.equu(t).has_nan(),
				!derivs.f(t).hasNaN(),
				!derivs.fx(t).hasNaN(),
				!derivs.fu(t).hasNaN(),
				!derivs.fxx(t).has_nan(),
				!derivs.fux(t).has_nan(),
				!derivs.fuu(t).has_nan());

		auto ne = derivs.eq(t).rows();
		auto ndx = derivs.fu(t).rows();
		auto ndu = derivs.fu(t).cols();

		T l{};
		T l_{};

		DDP_TMP_VECTOR_UNINIT(stack, f, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, f_, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, eq, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, eq_, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, _df1, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, _ddf1, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, _dddf1, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, _df2, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, _ddf2, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, _dddf2, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, _deq1, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, _ddeq1, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, _dddeq1, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, _deq2, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, _ddeq2, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, _dddeq2, T, ne);
		DDP_TMP_VECTOR_UNINIT(stack, dx_, T, ndx);
		DDP_TMP_VECTOR_UNINIT(stack, du_, T, ndu);

		eigen::assign(f, derivs.f(t));
		eigen::assign(f_, derivs.f(t));
		eigen::assign(eq, derivs.eq(t));
		eigen::assign(eq_, derivs.eq(t));

		T _dl1{};
		T _ddl1{};
		T _dddl1{};
		T _dl2{};
		T _ddl2{};
		T _dddl2{};

		auto* dl = &_dl1;
		auto* df = &_df1;
		auto* deq = &_deq1;

		auto* ddl = &_ddl1;
		auto* ddf = &_ddf1;
		auto* ddeq = &_ddeq1;

		auto* dddl = &_dddl1;
		auto* dddf = &_dddf1;
		auto* dddeq = &_dddeq1;

		dx_.setRandom();
		du_.setRandom();

		auto finite_diff_err = [&](T eps_x, T eps_u) {
			DDP_TMP_VECTOR_UNINIT(stack, dx, T, ndx);
			DDP_TMP_VECTOR_UNINIT(stack, du, T, ndu);
			DDP_TMP_VECTOR_UNINIT(stack, x_, T, x.rows());
			DDP_TMP_VECTOR_UNINIT(stack, u_, T, u.rows());

			eigen::mul_scalar_to(dx, dx_, eps_x);
			eigen::mul_scalar_to(du, du_, eps_u);

			dynamics.state_space().integrate(
					eigen::as_mut(x_), t, eigen::as_const(x), eigen::as_const(dx), stack);
			dynamics.control_space().integrate(
					eigen::as_mut(u_), t, eigen::as_const(u), eigen::as_const(du), stack);

			auto l = [&] {
				VEG_BIND(auto, (_, l), cost.eval(t, x, u, VEG_FWD(k), stack));
				k = VEG_FWD(_);
				return VEG_FWD(l);
			}();
			k = dynamics.eval_to(eigen::as_mut(f), t, x, u, VEG_FWD(k), stack);
			k = constraint.eval_to(eigen::as_mut(eq), t, x, u, VEG_FWD(k), stack);

			auto l_ = [&] {
				VEG_BIND(
						auto,
						(_, l),
						cost.eval(
								t,
								eigen::as_const(x_),
								eigen::as_const(u_),
								VEG_FWD(k),
								stack));
				k = VEG_FWD(_);
				return VEG_FWD(l);
			}();
			k = dynamics.eval_to(
					eigen::as_mut(f_),
					t,
					eigen::as_const(x_),
					eigen::as_const(u_),
					VEG_FWD(k),
					stack);
			k = constraint.eval_to(
					eigen::as_mut(eq_),
					t,
					eigen::as_const(x_),
					eigen::as_const(u_),
					VEG_FWD(k),
					stack);

			*dl = l_ - l;
			dynamics.output_space().difference(
					eigen::as_mut(*df),
					t + 1,
					eigen::as_const(f),
					eigen::as_const(f_),
					stack);
			constraint.output_space().difference(
					eigen::as_mut(*deq),
					t,
					eigen::as_const(eq),
					eigen::as_const(eq_),
					stack);

			*ddl =
					*dl - (eigen::dot(derivs.lx(t), dx) + eigen::dot(derivs.lu(t), du));

			eigen::assign(*ddf, *df);
			eigen::mul_add_to_noalias(*ddf, derivs.fx(t), dx, -1);
			eigen::mul_add_to_noalias(*ddf, derivs.fu(t), du, -1);

			if (ne > 0) {
				eigen::assign(*ddeq, *deq);
				eigen::mul_add_to_noalias(*ddeq, derivs.eqx(t), dx, -1);
				eigen::mul_add_to_noalias(*ddeq, derivs.equ(t), du, -1);
			}

			DDP_TMP_VECTOR(stack, dotx, T, dx.rows());
			DDP_TMP_VECTOR(stack, dotu, T, du.rows());

			eigen::mul_add_to_noalias(dotx, derivs.lxx(t), dx, 1);
			eigen::mul_add_to_noalias(dotu, derivs.luu(t), du, 1);

			*dddl = *ddl - (0.5 * eigen::dot(dx, dotx)     //
			                + 0.5 * eigen::dot(du, dotu)); //

			dotu.setZero();
			eigen::mul_add_to_noalias(dotu, derivs.lux(t), dx, 1);
			*dddl -= eigen::dot(du, dotu);

			eigen::mul_scalar_to(*dddf, *ddf, -1);
			eigen::mul_scalar_to(*dddeq, *ddeq, -1);

			internal::add_second_order_term(
					*dddf, derivs.fxx(t), derivs.fux(t), derivs.fuu(t), dx, du);

			internal::add_second_order_term(
					*dddeq, derivs.eqxx(t), derivs.equx(t), derivs.equu(t), dx, du);

			eigen::mul_scalar_to(*dddf, *dddf, -1);
			eigen::mul_scalar_to(*dddeq, *dddeq, -1);
		};

		T eps_x = 1e-30;
		T eps_u = 1e-30;

		T eps_factor = 0.1;

		finite_diff_err(eps_x, eps_u);

		dl = &_dl2;
		df = &_df2;
		deq = &_deq2;

		ddl = &_ddl2;
		ddf = &_ddf2;
		ddeq = &_ddeq2;

		dddl = &_dddl2;
		dddf = &_dddf2;
		dddeq = &_dddeq2;

		finite_diff_err(eps_x * eps_factor, eps_u * eps_factor);

		auto dl1 = _dl1;
		auto dl2 = _dl2;
		auto ddl1 = _ddl1;
		auto ddl2 = _ddl2;
		auto dddl1 = _dddl1;
		auto dddl2 = _dddl2;

		auto df1 = _df1.array();
		auto df2 = _df2.array();
		auto ddf1 = _ddf1.array();
		auto ddf2 = _ddf2.array();
		auto dddf1 = _dddf1.array();
		auto dddf2 = _dddf2.array();

		auto deq1 = _deq1.array();
		auto deq2 = _deq2.array();
		auto ddeq1 = _ddeq1.array();
		auto ddeq2 = _ddeq2.array();
		auto dddeq1 = _dddeq1.array();
		auto dddeq2 = _dddeq2.array();

		using std::fabs;
		using std::pow;

		auto eps = pow(std::numeric_limits<T>::epsilon(), 0.5);

		VEG_EXPECT_ALL_OF_ELSE(
				(fmt::format("at t = {}\n{}\n{}", t, dl1, dl2),
		     (fabs(dl1) <= eps || fabs(dl2) <= 2 * eps_factor * fabs(dl1))),
				(fmt::format("at t = {}\n{}\n{}", t, ddl1, ddl2),
		     (fabs(ddl1) <= eps ||
		      fabs(ddl2) <= 2 * pow(eps_factor, 2) * fabs(ddl1))),
				(fmt::format("at t = {}\n{}\n{}", t, dddl1, dddl2),
		     (fabs(dddl1) <= eps ||
		      fabs(dddl2) <= 2 * pow(eps_factor, 3) * fabs(dddl1))));

		for (i64 j = 0; j < df1.size(); ++j) {
			VEG_EXPECT_ALL_OF_ELSE(
					(fmt::format("at t = {}, j = {}\n{}\n{}", t, j, df1[j], df2[j]),
			     (fabs(df1[j]) <= eps ||
			      fabs(df2[j]) <= 2 * eps_factor * fabs(df1[j]))),
					(fmt::format("at t = {}, j = {}\n{}\n{}", t, j, ddf1[j], ddf2[j]),
			     (fabs(ddf1[j]) <= eps ||
			      fabs(ddf2[j]) <= 2 * pow(eps_factor, 2) * fabs(ddf1[j]))),
					(fmt::format("at t = {}, j = {}\n{}\n{}", t, j, dddf1[j], dddf2[j]),
			     (fabs(dddf1[j]) <= eps ||
			      fabs(dddf2[j]) <= 2 * pow(eps_factor, 3) * fabs(dddf1[j]))));
		}

		for (i64 j = 0; j < deq1.size(); ++j) {
			VEG_EXPECT_ALL_OF_ELSE(
					(fmt::format("at t = {}\n{}\n{}", t, deq1[j], deq2[j]),
			     (fabs(deq1[j]) <= eps ||
			      fabs(deq2[j]) <= 2 * eps_factor * fabs(deq1[j]))),
					(fmt::format("at t = {}\n{}\n{}", t, ddeq1[j], ddeq2[j]),
			     (fabs(ddeq1[j]) <= eps ||
			      fabs(ddeq2[j]) <= 2 * pow(eps_factor, 2) * fabs(ddeq1[j]))),
					(fmt::format("at t = {}\n{}\n{}", t, dddeq1[j], dddeq2[j]),
			     (fabs(dddeq1[j]) <= eps ||
			      fabs(dddeq2[j]) <= 2 * pow(eps_factor, 3) * fabs(dddeq1[j]))));
		}
	}
#endif
	return k;
}

} // namespace internal
} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_DERIVATIVE_STORAGE_HPP_LLE0WSK3S  \
        */
