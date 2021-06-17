#ifndef DDP_PINOCCHIO_DDP_HPP_CAAKIEJQS
#define DDP_PINOCCHIO_DDP_HPP_CAAKIEJQS

#include "ddp/internal/derivative_storage.hpp"
#include "ddp/internal/function_models.hpp"
#include "ddp/space.hpp"
#include "ddp/internal/idx_transforms.hpp"
#include "veg/util/timer.hpp"
#include <Eigen/Cholesky>
#include <iostream>
#include "veg/internal/prologue.hpp"

namespace ddp {
namespace internal {
struct thread_perf {
	i64 total_duration;
	i64 count;
	bool use_threads;
	auto avg() const -> long double {
		return static_cast<long double>(total_duration) /
		       static_cast<long double>(count);
	}
};
} // namespace internal

enum struct mult_update_attempt_result_e {
	no_update,
	update_success,
	update_failure,
	optimum_attained,
};

enum struct method {
	constant_multipliers,
	affine_multipliers,
};

template <typename Scalar>
struct regularization {

	void increase_reg() {
		m_factor = std::max(Scalar{1}, m_factor) * m_factor_update;

		if (m_reg == 0) {
			m_reg = m_min_value;
		} else {
			m_reg *= m_factor;
		}
	}

	void decrease_reg() {
		m_factor = std::min(Scalar{1}, m_factor) / m_factor_update;
		m_reg *= m_factor;

		if (m_reg <= m_min_value) {
			m_reg = 0;
		}
	}

	auto operator*() const -> Scalar { return m_reg; }

	Scalar m_reg;
	Scalar m_factor;
	Scalar const m_factor_update;
	Scalar const m_min_value;
};

template <typename Dynamics, typename Cost, typename Constraint>
struct ddp {
	Dynamics dynamics;
	Cost cost;
	Constraint constraint;

	using scalar = typename meta::uncvref_t<Dynamics>::scalar;
	using trajectory = ::ddp::trajectory<scalar>;
	using key = decltype(VEG_DECLVAL(Dynamics const&).acquire_workspace());

	using state_space = decltype(dynamics.state_space());
	using control_space = decltype(dynamics.control_space());
	using constraint_output_space =
			decltype(constraint.ref(dynamics).output_space());

	template <method K, typename = void>
	struct multiplier_sequence;
	template <typename Dummy>
	struct multiplier_sequence<method::constant_multipliers, Dummy> {
		using eq_type = internal::constant_function_seq<scalar>;

		struct type {
			eq_type eq;
		};

		static auto zero(
				state_space x_space,
				constraint_output_space eq_space,
				trajectory const& traj) -> type {

			unused(traj, x_space);
			auto begin = traj.index_begin();
			auto end = traj.index_end();
			auto multipliers = type{eq_type{begin, end, eq_space}};
			for (i64 t = begin; t < end; ++t) {
				multipliers.eq.self.val[t].setZero();
			}
			return multipliers;
		}
	};
	template <typename Dummy>
	struct multiplier_sequence<method::affine_multipliers, Dummy> {
		using eq_type = internal::affine_function_seq<scalar, state_space>;

		struct type {
			eq_type eq;
		};

		static auto zero(
				state_space x_space,
				constraint_output_space eq_space,
				trajectory const& traj) -> type {

			auto begin = traj.index_begin();
			auto end = traj.index_end();
			auto multipliers = type{eq_type{begin, end, x_space, eq_space}};

			for (i64 t = begin; t < end; ++t) {
				auto& fn = multipliers.eq.self;
				fn.val[t].setZero();
				fn.jac[t].setZero();
				eigen::assign(fn.origin[t], traj.self.x[t]);
			}
			return multipliers;
		}
	};

	template <method M>
	auto zero_multipliers(trajectory const& traj) const ->
			typename multiplier_sequence<M>::type {
		return multiplier_sequence<M>::zero(
				dynamics.state_space(), constraint.ref(dynamics).output_space(), traj);
	}

	template <typename Control_Gen>
	auto make_trajectory(
			i64 begin,
			i64 end,
			view<scalar const, colvec> x_init,
			Control_Gen it_u) const -> trajectory {

		trajectory traj{
				::ddp::space_to_idx(dynamics.state_space(), begin, end + 1),
				::ddp::space_to_idx(dynamics.control_space(), begin, end),
		};

		auto stack_storage = std::vector<unsigned char>(
				(1U << 25U) +
				narrow<usize>(
						dynamics.eval_to_req().size + dynamics.eval_to_req().align));

		auto stack = DynStackView(slice::from_range(stack_storage));
		auto k = dynamics.acquire_workspace();

		VEG_DEBUG_ASSERT(x_init.rows() == traj[begin][0_c].rows());
		traj[begin][0_c] = x_init;
		for (i64 t = begin; t < end; ++t) {
			VEG_BIND(auto, (x, u), traj[t]);
			auto x_next = traj.x(t + 1);

			eigen::assign(u, it_u(eigen::as_const(x)));
			k = dynamics.eval_to(
					x_next, t, eigen::as_const(x), eigen::as_const(u), VEG_FWD(k), stack);
		}
		return traj;
	}

	using control_feedback = internal::affine_function_seq<scalar, state_space>;
	using derivative_storage = internal::second_order_derivatives<scalar>;
	using derivative_storage_base = internal::first_order_derivatives<scalar>;

	auto make_derivative_storage(trajectory const& traj) const
			-> internal::second_order_derivatives<scalar> {

		i64 begin = traj.index_begin();
		i64 end = traj.index_end();

		using vecseq = internal::mat_seq<scalar, colvec>;
		using matseq = internal::mat_seq<scalar, colmat>;

		idx::idx<colvec> idxx =
				::ddp::space_to_idx(constraint.ref(dynamics).state_space(), begin, end);
		idx::idx<colvec> idxu = ::ddp::space_to_idx(
				constraint.ref(dynamics).control_space(), begin, end);
		idx::idx<colvec> idxe = ::ddp::space_to_idx(
				constraint.ref(dynamics).output_space(), begin, end);

		auto x = idxx.as_view();
		auto u = idxu.as_view();
		auto e = idxe.as_view();

		auto prod2 = [&](idx::idx_view<colvec> l, idx::idx_view<colvec> r) {
			return matseq(idx::prod_idx(l, r));
		};
		auto prod3 = [&](idx::idx_view<colvec> o,
		                 idx::idx_view<colvec> l,
		                 idx::idx_view<colvec> r) {
			return internal::tensor_seq<scalar>(
					idx::tensor_idx(begin, end, [&](i64 t) -> idx::tensor_dims {
						return {o.rows(t), l.rows(t), r.rows(t)};
					}));
		};

		return internal::second_order_derivatives<scalar>{
				{{
						{traj.x_f().rows()},

						vecseq(idxx),
						vecseq(idxu),

						vecseq(idxx),
						prod2(x, x),
						prod2(x, u),

						vecseq(idxe),
						prod2(e, x),
						prod2(e, u),
				}},

				{
						{traj.x_f().rows(), traj.x_f().rows()},

						prod2(x, x),
						prod2(u, x),
						prod2(u, u),

						prod3(x, x, x),
						prod3(x, u, x),
						prod3(x, u, u),

						prod3(e, x, x),
						prod3(e, u, x),
						prod3(e, u, u),
				}};
	}

	auto optimality_constr(derivative_storage const& derivs) const -> scalar {
		using std::max;
		scalar retval(0);
		auto const& eq = derivs.first_order_derivatives::self.eq;
		for (i64 t = eq.index_begin(); t < eq.index_end(); ++t) {
			retval = max(retval, derivs.eq(t).stableNorm());
		}
		return retval;
	}

	template <typename Mult_Seq>
	auto optimality_obj_req(Mult_Seq const& mults) const -> mem_req {
		return mem_req::sum_of({
				as_ref,
				{
						mem_req{
								tag<scalar>, constraint.ref(dynamics).state_space().max_ddim()},
						mem_req{
								tag<scalar>, constraint.ref(dynamics).state_space().max_ddim()},
						mem_req{
								tag<scalar>,
								constraint.ref(dynamics).output_space().max_ddim()},
						mem_req{
								tag<scalar>,
								constraint.ref(dynamics).control_space().max_ddim()},
						mults.eq.eval_to_req(),
				},
		});
	}

	template <typename Mult_Seq>
	auto optimality_obj(
			trajectory const& traj,
			Mult_Seq const& mults,
			scalar const& mu,
			derivative_storage const& derivs,
			DynStackView stack) const -> scalar {

		using std::max;

		scalar retval = 0;

		DDP_TMP_VECTOR_UNINIT(
				stack,
				adj_storage,
				scalar,
				constraint.ref(dynamics).state_space().max_ddim());
		DDP_TMP_VECTOR_UNINIT(
				stack,
				adj_storage_2,
				scalar,
				constraint.ref(dynamics).state_space().max_ddim());

		eigen::assign(
				eigen::slice_to_vec(adj_storage).topRows(derivs.lfx().rows()),
				derivs.lfx());

		DDP_TMP_VECTOR_UNINIT(
				stack,
				pe_storage,
				scalar,
				constraint.ref(dynamics).output_space().max_ddim());
		DDP_TMP_VECTOR_UNINIT(
				stack,
				lu_storage,
				scalar,
				constraint.ref(dynamics).control_space().max_ddim());

		for (i64 t = traj.index_end() - 1; t >= traj.index_begin(); --t) {

			auto nx_next = constraint.ref(dynamics).state_space().ddim(t + 1);

			auto adj = eigen::as_mut(adj_storage.topRows(nx_next).transpose());
			auto pe = eigen::as_mut(
					pe_storage.topRows(constraint.ref(dynamics).output_space().ddim(t)));
			auto lu = eigen::as_mut(
					lu_storage.topRows(constraint.ref(dynamics).control_space().ddim(t)));

			auto eq = derivs.eq(t);
			auto equ = derivs.equ(t);
			auto eqx = derivs.eqx(t);
			auto fu = derivs.fu(t);
			auto fx = derivs.fx(t);

			mults.eq.eval_to(pe, t, traj.x(t), stack);

			eigen::assign(lu, derivs.lu(t).transpose());
			eigen::mul_add_to_noalias(lu, adj, fu);
			if (equ.rows() > 0) {
				eigen::mul_add_to_noalias(lu, pe.transpose(), equ);
				eigen::mul_add_to_noalias(lu, eq.transpose(), equ, mu);
			}

			retval = max(retval, lu.stableNorm());

			if (t > traj.index_begin()) {
				auto nx = constraint.ref(dynamics).state_space().ddim(t);
				auto adj_prev = eigen::as_mut(adj_storage.topRows(nx).transpose());
				auto tmp = eigen::as_mut(adj_storage_2.topRows(nx));
				auto lx = derivs.lx(t).transpose();

				tmp.setZero();

				eigen::mul_add_to_noalias(tmp, adj, fx);
				eigen::add_to(tmp, tmp, lx.transpose());

				if (eq.rows() > 0) {
					eigen::mul_add_to_noalias(tmp, eq.transpose(), eqx, mu);
					eigen::mul_add_to_noalias(tmp, pe.transpose(), eqx);
					eigen::mul_add_to_noalias(tmp, eq.transpose(), mults.eq.self.jac[t]);
				}

				eigen::assign(adj_prev, tmp);
			}
		}
		return retval;
	}
	template <typename Mults>
	auto update_derivatives_req(
			control_feedback const& fb_seq, Mults const& mults) const -> mem_req {
		return mem_req::max_of({
				as_ref,
				{
						internal::compute_second_derivatives_req(
								cost.ref(dynamics), dynamics, constraint.ref(dynamics)),
						mults.eq.update_origin_req(),
						fb_seq.update_origin_req(),
						optimality_obj_req(mults),
				},
		});
	}

	template <typename Mults>
	auto update_derivatives(
			derivative_storage& derivs,
			control_feedback& fb_seq,
			Mults& mults,
			trajectory const& traj,
			scalar mu,
			scalar w,
			scalar n,
			scalar stopping_threshold,
			DynStackView stack,
			internal::thread_perf& threading_method) const
			-> Tuple<mult_update_attempt_result_e, scalar, scalar> {

		threading_method.total_duration += internal::compute_second_derivatives(
				derivs,
				cost.ref(dynamics),
				dynamics,
				constraint.ref(dynamics),
				traj,
				stack,
				threading_method.use_threads);

		mults.eq.update_origin(traj.self.x, stack);
		fb_seq.update_origin(traj.self.x, stack);

		auto opt_obj = optimality_obj(traj, mults, mu, derivs, stack);
		auto opt_constr = optimality_constr(derivs);

		if (opt_constr < stopping_threshold and opt_obj < stopping_threshold) {
			return {
					direct,
					mult_update_attempt_result_e::optimum_attained,
					opt_obj,
					opt_constr,
			};
		}

		if (opt_obj <= w) {
			if (opt_constr <= n) {

				for (i64 t = traj.index_begin(); t < traj.index_end(); ++t) {

					if (derivs.eq(t).size() > 0) {
						eigen::mul_add_to_noalias(
								mults.eq.self.jac[t], derivs.equ(t), fb_seq.self.jac[t], mu);
						eigen::mul_scalar_add_to(mults.eq.self.jac[t], derivs.eqx(t), mu);
						eigen::mul_scalar_add_to(mults.eq.self.val[t], derivs.eq(t), mu);
					}
				}
				return {
						direct,
						mult_update_attempt_result_e::update_success,
						opt_obj,
						opt_constr,
				};
			} else {
				return {
						direct,
						mult_update_attempt_result_e::update_failure,
						opt_obj,
						opt_constr,
				};
			}
		} else {
			return {
					direct,
					mult_update_attempt_result_e::no_update,
					opt_obj,
					opt_constr,
			};
		}
	}

	template <method M>
	auto solve(
			i64 max_iterations,
			scalar optimality_stopping_threshold,
			scalar mu,
			trajectory traj) -> Tuple<trajectory, control_feedback> {

		auto mult = zero_multipliers<M>(traj);
		auto reg = regularization<scalar>{0, 1, 2, 1e-5};
		auto ctrl = control_feedback(
				traj.index_begin(),
				traj.index_end(),
				dynamics.state_space(),
				dynamics.control_space());

		auto derivs = make_derivative_storage(traj);
		auto traj2 = traj;

		using std::pow;

		scalar previous_opt_constr{};
		scalar w = 1 / mu;
		scalar n = 1 / pow(mu, static_cast<scalar>(0.1L));

		mem_req req = //
				mem_req::max_of({
						as_ref,
						{
								mem_req::sum_of({
										as_ref,
										{
												update_derivatives_req(ctrl, mult),
												internal::compute_second_derivatives_test_req(
														cost.ref(dynamics),
														dynamics,
														constraint.ref(dynamics)),
										},
								}),
								fwd_pass_req(traj, mult),
								bwd_pass_req(),
						},
				});
		;
		auto stack_storage =
				std::vector<unsigned char>(narrow<usize>(req.align + req.size));

		DynStackView stack(slice::from_range(stack_storage));

		internal::thread_perf single{0, 0, false};
		internal::thread_perf multi{0, 0, true};
		for (i64 i = 0; i < max_iterations; ++i) {

			internal::thread_perf& threading_method =
					[&]() -> internal::thread_perf& {
				if (multi.count < 5) {
					return multi;
				}
				if ((single.count < 5) || (single.avg() < multi.avg())) {
					return single;
				}

				return multi;
			}();

			++threading_method.count;
			if (i == 0) {
				threading_method.total_duration += internal::compute_second_derivatives(
						derivs,
						cost.ref(dynamics),
						dynamics,
						constraint.ref(dynamics),
						traj,
						stack,
						threading_method.use_threads);

				previous_opt_constr = optimality_constr(derivs);
			} else {
				VEG_BIND(
						auto,
						(mult_update_rv, opt_obj, opt_constr),
						(update_derivatives(
								derivs,
								ctrl,
								mult,
								traj,
								mu,
								w,
								n,
								optimality_stopping_threshold,
								stack,
								threading_method)));
				unused(opt_obj);

				scalar const beta = 0.5;
				switch (mult_update_rv) {
				case mult_update_attempt_result_e::no_update: {
					break;
				}
				case mult_update_attempt_result_e::update_failure: {
					using std::pow;
					::fmt::print(
							"desired new mu {}\n",
							pow(mu / (previous_opt_constr / opt_constr), 1 / (1 - beta)));
					mu = 10 * std::max(     //
												std::min( //
														pow(mu / (previous_opt_constr / opt_constr),
					                      1.0 / (1 - beta)),
														mu * scalar{1e5}),
												mu);
					break;
				}
				case mult_update_attempt_result_e::update_success: {
					using std::pow;
					n = opt_constr / pow(mu, beta / 2);
					w = std::max(w / pow(mu, scalar{1}), n / mu);
					previous_opt_constr = opt_constr;
					break;
				}
				case mult_update_attempt_result_e::optimum_attained:
					return {
							direct,
							VEG_FWD(traj),
							VEG_FWD(ctrl),
					};
				}
			}

			{
				auto&& _ = time::raii_timer(time::log_elapsed_time("bwd pass"));
				veg::unused(_);

				bwd_pass(ctrl, reg, mu, traj, mult, derivs, stack);
			}

			::fmt::print(
					stdout,
					"=================================================="
					"==================================================\n"
					"iter: {:5}   mu: {:13}   reg: {:13}   w: {:13}   n: {:13}\n",
					i,
					mu,
					*reg,
					w,
					n);

			{
				auto&& _ = time::raii_timer(time::log_elapsed_time("fwd pass"));
				veg::unused(_);

				auto res = fwd_pass(
						traj2,
						traj,
						mult,
						ctrl,
						mu,
						dynamics.acquire_workspace(),
						stack,
						true);
				if (res[1_c] >= 0.5) {
					reg.decrease_reg();
				}
			}

			std::swap(traj, traj2);
		}
		return {
				direct,
				VEG_FWD(traj),
				VEG_FWD(ctrl),
		};
	}

	auto bwd_pass_req() const noexcept -> mem_req {
		i64 nf = dynamics.output_space().max_ddim();
		i64 nx = dynamics.state_space().max_ddim();
		i64 nu = constraint.ref(dynamics).control_space().max_ddim();
		i64 ne = constraint.ref(dynamics).output_space().max_ddim();
		return mem_req::sum_of({
				as_ref,
				{
						{tag<scalar>, nf},      // vx
						{tag<scalar>, nf * nf}, // vxx
						{tag<scalar>, ne},      // tmp
						{tag<scalar>, ne * nx}, // tmp2
						{tag<scalar>, nx},      // qx
						{tag<scalar>, nu},      // qu
						{tag<scalar>, nx * nx}, // qxx
						{tag<scalar>, nu * nu}, // quu
						{tag<scalar>, nu * nx}, // qux
						mem_req::max_of({
								as_ref,
								{
										{tag<scalar>, nf * nx}, // tmp (qxx/qux)
										{tag<scalar>, nf * nu}, // tmp (quu)
										{tag<scalar>, nu * nu}, // quu_clone
								},
						}),
				},
		});
	}
	template <typename Mults>
	auto bwd_pass(
			control_feedback& control_feedback,
			regularization<scalar>& reg,
			scalar mu,
			trajectory const& current_traj,
			Mults const& mults,
			derivative_storage const& derivatives,
			DynStackView stack) const -> scalar {

		bool success = false;

		scalar expected_decrease = 0;
		while (!success) {
			::fmt::print("mu: {}\n", mu);

			i64 nxf = derivatives.lfx().rows();

			DDP_TMP_MATRIX_UNINIT(stack, V_xx, scalar, nxf, nxf);
			DDP_TMP_VECTOR_UNINIT(stack, V_x, scalar, nxf);

			eigen::assign(V_xx, derivatives.lfxx());
			eigen::assign(V_x, derivatives.lfx());
			auto const v_x = eigen::as_const(V_x);

			expected_decrease = 0;

			for (i64 t = current_traj.index_end() - 1;
			     t >= current_traj.index_begin();
			     --t) {
				auto xu = current_traj[t];

				auto lx = derivatives.lx(t);
				auto lu = derivatives.lu(t);
				auto lxx = derivatives.lxx(t);
				auto lux = derivatives.lux(t);
				auto luu = derivatives.luu(t);

				auto fx = derivatives.fx(t);
				auto fu = derivatives.fu(t);
				auto fxx = derivatives.fxx(t);
				auto fux = derivatives.fux(t);
				auto fuu = derivatives.fuu(t);

				auto eq_ = derivatives.eq(t);
				auto eqx = derivatives.eqx(t);
				auto equ = derivatives.equ(t);
				auto eqxx = derivatives.eqxx(t);
				auto equx = derivatives.equx(t);
				auto equu = derivatives.equu(t);

				auto pe = mults.eq.self.val[t];
				auto pe_x = mults.eq.self.jac[t];

				DDP_TMP_VECTOR_UNINIT(stack, tmp, scalar, pe.rows());
				DDP_TMP_MATRIX_UNINIT(stack, tmp2, scalar, pe.rows(), pe_x.cols());

				eigen::assign(tmp, pe);
				eigen::mul_scalar_add_to(tmp, eq_, mu);

				eigen::assign(tmp2, pe_x);
				eigen::mul_scalar_add_to(tmp2, eqx, mu);

				bool const has_eq = tmp.rows() > 0;

				{
					using std::isfinite;
					VEG_DEBUG_ASSERT_ALL_OF( //
							isfinite(mu),
							!pe.hasNaN(),
							!pe_x.hasNaN());
				}

				DDP_TMP_VECTOR_UNINIT(stack, Q_x, scalar, lx.rows());
				eigen::assign(Q_x, lx);
				eigen::mul_add_to_noalias(Q_x, fx.transpose(), v_x);
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_x, eqx.transpose(), tmp);
					eigen::mul_add_to_noalias(Q_x, pe_x.transpose(), eq_);
				}

				DDP_TMP_VECTOR_UNINIT(stack, Q_u, scalar, lu.rows());
				eigen::assign(Q_u, lu);
				eigen::mul_add_to_noalias(Q_u, fu.transpose(), v_x);
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_u, equ.transpose(), tmp);
				}

				DDP_TMP_MATRIX_UNINIT(stack, Q_xx, scalar, lxx.rows(), lxx.rows());
				eigen::assign(Q_xx, lxx);
				{
					DDP_TMP_MATRIX(stack, tmp_prod, scalar, V_xx.rows(), fx.cols());
					eigen::mul_add_to_noalias(tmp_prod, V_xx, fx);
					eigen::mul_add_to_noalias(Q_xx, fx.transpose(), tmp_prod);
				}
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_xx, eqx.transpose(), tmp2);
					eigen::mul_add_to_noalias(Q_xx, pe_x.transpose(), eqx);
					eqxx.noalias_contract_add_outdim(
							eigen::as_mut(Q_xx), eigen::as_const(tmp));
				}
				fxx.noalias_contract_add_outdim(eigen::as_mut(Q_xx), v_x);

				DDP_TMP_MATRIX_UNINIT(stack, Q_uu, scalar, luu.rows(), luu.rows());
				eigen::assign(Q_uu, luu);
				{
					DDP_TMP_MATRIX(stack, tmp_prod, scalar, V_xx.rows(), fu.cols());
					eigen::mul_add_to_noalias(tmp_prod, V_xx, fu);
					eigen::mul_add_to_noalias(Q_uu, fu.transpose(), tmp_prod);
				}
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_uu, equ.transpose(), equ, mu);
					equu.noalias_contract_add_outdim(
							eigen::as_mut(Q_uu), eigen::as_const(tmp));
				}
				fuu.noalias_contract_add_outdim(eigen::as_mut(Q_uu), v_x);

				DDP_TMP_MATRIX_UNINIT(stack, Q_ux, scalar, lux.rows(), lux.cols());
				eigen::assign(Q_ux, lux);
				{
					DDP_TMP_MATRIX(stack, tmp_prod, scalar, V_xx.rows(), fx.cols());
					eigen::mul_add_to_noalias(tmp_prod, V_xx, fx);
					eigen::mul_add_to_noalias(Q_ux, fu.transpose(), tmp_prod);
				}
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_ux, equ.transpose(), tmp2);
					equx.noalias_contract_add_outdim(
							eigen::as_mut(Q_ux), eigen::as_const(tmp));
				}
				fux.noalias_contract_add_outdim(eigen::as_mut(Q_ux), v_x);

				auto k = control_feedback.self.val[t];
				auto K = control_feedback.self.jac[t];

				{
					DDP_TMP_MATRIX_UNINIT(
							stack, Q_uu_clone, scalar, Q_uu.rows(), Q_uu.rows());
					eigen::assign(Q_uu_clone, Q_uu);
					eigen::add_identity(Q_uu_clone, *reg);

					Eigen::LLT<eigen::view<scalar, colmat>> llt_res(Q_uu_clone);

					if (!(llt_res.info() == Eigen::ComputationInfo::Success)) {
						reg.increase_reg();
						break;
					}

					{
						eigen::assign(control_feedback.self.origin[t], xu[0_c]);
						k = llt_res.solve(Q_u);
						K = llt_res.solve(Q_ux);
						eigen::mul_scalar_to(k, k, -1);
						eigen::mul_scalar_to(K, K, -1);
					}
				}

				::fmt::print("{}\n", k);

				{
					DDP_TMP_VECTOR(stack, dotk, scalar, Q_uu.rows());
					eigen::mul_add_to_noalias(dotk, Q_uu, k);
					expected_decrease += 0.5 * eigen::dot(k, dotk);
				}

				eigen::assign(V_x, Q_x);
				eigen::mul_add_to_noalias(V_x, k.transpose(), Q_ux);

				eigen::assign(V_xx, Q_xx);
				eigen::mul_add_to_noalias(V_xx, Q_ux.transpose(), K);

				if (t == 0) {
					success = true;
				}
			}
		}
		return expected_decrease;
	}

	template <typename Mults>
	auto fwd_pass_req(trajectory const& traj, Mults const& mults) const
			-> mem_req {
		return mem_req::sum_of({
				as_ref,
				{
						mem_req{
								tag<scalar>, 2 * (traj.index_end() - traj.index_begin() + 1)},
						cost_seq_aug_req(mults),
						mem_req{
								tag<scalar>, constraint.ref(dynamics).state_space().max_ddim()},

						mem_req::max_of({
								as_ref,
								{
										dynamics.state_space().difference_req(),
										dynamics.eval_to_req(),
										cost_seq_aug_req(mults),
								},
						}),
				},
		});
	}

	template <typename Mults>
	auto fwd_pass(
			trajectory& new_traj_storage,
			trajectory const& reference_traj,
			Mults const& old_mults,
			control_feedback const& feedback,
			scalar mu,
			key k,
			DynStackView stack,
			bool do_linesearch = true) const -> Tuple<key, scalar> {

		auto begin = reference_traj.index_begin();
		auto end = reference_traj.index_end();

		DDP_TMP_VECTOR(stack, costs_old_traj, scalar, end - begin + 1);
		DDP_TMP_VECTOR(stack, costs_new_traj, scalar, end - begin + 1);

		if (do_linesearch) {
			k = cost_seq_aug(
					eigen::slice_to_vec(costs_old_traj),
					reference_traj,
					old_mults,
					mu,
					VEG_FWD(k),
					stack);
		}

		scalar step = 1;
		bool success = false;

		DDP_TMP_VECTOR(
				stack, tmp, scalar, constraint.ref(dynamics).state_space().max_ddim());

		while (!success) {
			if (step < 1e-10) {
				// step = 0;
				break;
			}

			for (i64 t = begin; t < end; ++t) {

				auto x_old = reference_traj[t][0_c];
				auto u_old = reference_traj[t][1_c];

				auto x_new = new_traj_storage[t][0_c];
				auto u_new = new_traj_storage[t][1_c];
				auto x_next_new = new_traj_storage.x(t + 1);

				dynamics.state_space().difference(
						eigen::as_mut(tmp),
						t,
						eigen::as_const(x_old),
						eigen::as_const(x_new),
						stack);

				u_new = u_old                         //
				        + step * feedback.self.val[t] //
				        + feedback.self.jac[t] * tmp;
				k = dynamics.eval_to(
						x_next_new,
						t,
						eigen::as_const(x_new),
						eigen::as_const(u_new),
						VEG_FWD(k),
						stack);
			}

			if (do_linesearch) {
				k = cost_seq_aug(
						eigen::as_mut(costs_new_traj),
						new_traj_storage,
						old_mults,
						mu,
						VEG_FWD(k),
						stack);

				if ((costs_new_traj - costs_old_traj).sum() <= 0) {
					success = true;
				} else {
					step *= 0.5;
				}
			} else {
				success = true;
			}
		}

		return {direct, VEG_FWD(k), step};
	}

	template <typename Mults>
	auto cost_seq_aug_req(Mults const& mults) const -> mem_req {
		return mem_req::sum_of({
				as_ref,
				{
						mem_req{
								tag<scalar>, constraint.ref(dynamics).output_space().max_dim()},
						mem_req{
								tag<scalar>, constraint.ref(dynamics).output_space().max_dim()},

						mem_req::max_of({
								as_ref,
								{
										cost.ref(dynamics).eval_req(),
										constraint.ref(dynamics).eval_to_req(),
										mults.eq.eval_to_req(),
								},
						}),
				},
		});
	}

	template <typename Mults>
	auto cost_seq_aug(
			eigen::view<scalar, colvec> out,
			trajectory const& traj,
			Mults const& mults,
			scalar mu,
			key k,
			DynStackView stack) const -> key {

		auto csp = constraint.ref(dynamics).output_space();

		DDP_TMP_VECTOR_UNINIT(stack, ce_storage, scalar, csp.max_dim());
		DDP_TMP_VECTOR_UNINIT(stack, pe_storage, scalar, csp.max_dim());

		for (i64 t = traj.index_begin(); t < traj.index_end(); ++t) {
			VEG_BIND(auto, (x, u), traj[t]);
			auto ce = eigen::as_mut(ce_storage.topRows(csp.dim(t)));
			auto pe = eigen::as_mut(pe_storage.topRows(csp.dim(t)));

			VEG_BIND(
					auto, (_k, l), cost.ref(dynamics).eval(t, x, u, VEG_FWD(k), stack));
			k = VEG_FWD(_k);

			k = constraint.ref(dynamics).eval_to(ce, t, x, u, VEG_FWD(k), stack);
			mults.eq.eval_to(pe, t, x, stack);

			out[t - traj.index_begin()] =
					l + eigen::dot(pe, ce) + (mu / 2) * eigen::dot(ce, ce);
		}

		auto x = traj.x_f();
		VEG_BIND(
				auto, (_k, l), cost.ref(dynamics).eval_final(x, VEG_FWD(k), stack));
		k = VEG_FWD(_k);
		out[traj.index_end() - traj.index_begin()] = l;

		return k;
	}
};

namespace make {
namespace fn {
struct ddp_fn {
	template <typename Dynamics, typename Cost, typename Constraint>
	auto
	operator()(Dynamics&& dynamics, Cost&& cost, Constraint&& constraint) const
			-> ::ddp::ddp<Dynamics, Cost, Constraint> {
		return {
				VEG_FWD(dynamics),
				VEG_FWD(cost),
				VEG_FWD(constraint),
		};
	}
};
} // namespace fn
VEG_INLINE_VAR(ddp, fn::ddp_fn);
} // namespace make

} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_DDP_HPP_CAAKIEJQS */
