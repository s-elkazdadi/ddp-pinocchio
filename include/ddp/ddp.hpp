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
struct ThreadPerf {
	i64 total_duration;
	i64 count;
	bool use_threads;
	auto avg() const -> long double {
		return static_cast<long double>(total_duration) /
		       static_cast<long double>(count);
	}
};
} // namespace internal

enum struct MultUpdateAttemptResult_e {
	no_update,
	update_success,
	update_failure,
	optimum_attained,
};

enum struct Method_e {
	constant_multipliers,
	affine_multipliers,
};

template <typename Scalar>
struct Regularization {

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
struct Ddp {
	Dynamics dynamics;
	Cost cost;
	Constraint constraint;

	using Scalar = typename meta::uncvref_t<Dynamics>::Scalar;
	using trajectory = ::ddp::Trajectory<Scalar>;
	using Key = decltype(VEG_DECLVAL(Dynamics const&).acquire_workspace());

	using StateSpace = decltype(dynamics.state_space());
	using ControlSpace = decltype(dynamics.control_space());
	using ConstraintOutSpace = decltype(constraint.ref(dynamics).output_space());

	template <Method_e K, typename = void>
	struct MultiplierSeq;
	template <typename Dummy>
	struct MultiplierSeq<Method_e::constant_multipliers, Dummy> {
		using Eq = internal::ConstantFnSeq<Scalar>;

		struct Type {
			Eq eq;
		};

		static auto zero(
				StateSpace x_space, ConstraintOutSpace eq_space, trajectory const& traj)
				-> Type {

			unused(traj, x_space);
			auto begin = traj.index_begin();
			auto end = traj.index_end();
			auto multipliers = Type{Eq{begin, end, eq_space}};
			for (i64 t = begin; t < end; ++t) {
				multipliers.eq.self.val[t].setZero();
			}
			return multipliers;
		}
	};
	template <typename Dummy>
	struct MultiplierSeq<Method_e::affine_multipliers, Dummy> {
		using Eq = internal::AffineFnSeq<Scalar, StateSpace>;

		struct Type {
			Eq eq;
		};

		static auto zero(
				StateSpace x_space, ConstraintOutSpace eq_space, trajectory const& traj)
				-> Type {

			auto begin = traj.index_begin();
			auto end = traj.index_end();
			auto multipliers = Type{Eq{begin, end, x_space, eq_space}};

			for (i64 t = begin; t < end; ++t) {
				auto& fn = multipliers.eq.self;
				fn.val[t].setZero();
				fn.jac[t].setZero();
				eigen::assign(fn.origin[t], traj.self.x[t]);
			}
			return multipliers;
		}
	};

	template <Method_e M>
	auto zero_multipliers(trajectory const& traj) const ->
			typename MultiplierSeq<M>::Type {
		return MultiplierSeq<M>::zero(
				dynamics.state_space(), constraint.ref(dynamics).output_space(), traj);
	}

	template <typename Control_Gen>
	auto make_trajectory(
			i64 begin,
			i64 end,
			View<Scalar const, colvec> x_init,
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

	using ControlFeedback = internal::AffineFnSeq<Scalar, StateSpace>;
	using DerivativeStorage = internal::SecondOrderDerivatives<Scalar>;
	using DerivativeStorageBase = internal::FirstOrderDerivatives<Scalar>;

	auto make_derivative_storage(trajectory const& traj) const
			-> internal::SecondOrderDerivatives<Scalar> {

		i64 begin = traj.index_begin();
		i64 end = traj.index_end();

		using vecseq = internal::MatSeq<Scalar, colvec>;
		using matseq = internal::MatSeq<Scalar, colmat>;

		idx::Idx<colvec> idxx =
				::ddp::space_to_idx(constraint.ref(dynamics).state_space(), begin, end);
		idx::Idx<colvec> idxu = ::ddp::space_to_idx(
				constraint.ref(dynamics).control_space(), begin, end);
		idx::Idx<colvec> idxe = ::ddp::space_to_idx(
				constraint.ref(dynamics).output_space(), begin, end);

		auto x = idxx.as_view();
		auto u = idxu.as_view();
		auto e = idxe.as_view();

		auto prod2 = [&](idx::IdxView<colvec> l, idx::IdxView<colvec> r) {
			return matseq(idx::prod_idx(l, r));
		};
		auto prod3 = [&](idx::IdxView<colvec> o,
		                 idx::IdxView<colvec> l,
		                 idx::IdxView<colvec> r) {
			return internal::TensorSeq<Scalar>(
					idx::TensorIdx(begin, end, [&](i64 t) -> idx::TensorDims {
						return {o.rows(t), l.rows(t), r.rows(t)};
					}));
		};

		return internal::SecondOrderDerivatives<Scalar>{
				{
						eigen::HeapMatrix<Scalar, colvec>{
								eigen::with_dims,
								traj.x_f().rows(),
						},

						vecseq(idxx),
						vecseq(idxu),

						vecseq(idxx),
						prod2(x, x),
						prod2(x, u),

						vecseq(idxe),
						prod2(e, x),
						prod2(e, u),
				},

				{
						eigen::HeapMatrix<Scalar, colmat>{
								eigen::with_dims,
								traj.x_f().rows(),
								traj.x_f().rows(),
						},

						prod2(x, x),
						prod2(u, x),
						prod2(u, u),

						prod3(x, x, x),
						prod3(x, u, x),
						prod3(x, u, u),

						prod3(e, x, x),
						prod3(e, u, x),
						prod3(e, u, u),
				},
		};
	}

	auto optimality_constr(DerivativeStorage const& derivs) const -> Scalar {
		using std::max;
		Scalar retval(0);
		auto const& eq = derivs.FirstOrderDerivatives::self.eq;
		for (i64 t = eq.index_begin(); t < eq.index_end(); ++t) {
			retval = max(retval, derivs.eq(t).stableNorm());
		}
		return retval;
	}

	template <typename MultSeq>
	auto optimality_obj_req(MultSeq const& mults) const -> MemReq {
		return MemReq::sum_of({
				as_ref,
				{
						MemReq{
								tag<Scalar>, constraint.ref(dynamics).state_space().max_ddim()},
						MemReq{
								tag<Scalar>, constraint.ref(dynamics).state_space().max_ddim()},
						MemReq{
								tag<Scalar>,
								constraint.ref(dynamics).output_space().max_ddim()},
						MemReq{
								tag<Scalar>,
								constraint.ref(dynamics).control_space().max_ddim()},
						mults.eq.eval_to_req(),
				},
		});
	}

	template <typename Mult_Seq>
	auto optimality_obj(
			trajectory const& traj,
			Mult_Seq const& mults,
			Scalar const& mu,
			DerivativeStorage const& derivs,
			DynStackView stack) const -> Scalar {

		using std::max;

		Scalar retval = 0;

		DDP_TMP_VECTOR_UNINIT(
				stack,
				adj_storage,
				Scalar,
				constraint.ref(dynamics).state_space().max_ddim());
		DDP_TMP_VECTOR_UNINIT(
				stack,
				adj_storage_2,
				Scalar,
				constraint.ref(dynamics).state_space().max_ddim());

		eigen::assign(
				eigen::slice_to_vec(adj_storage).topRows(derivs.lfx().rows()),
				derivs.lfx());

		DDP_TMP_VECTOR_UNINIT(
				stack,
				pe_storage,
				Scalar,
				constraint.ref(dynamics).output_space().max_ddim());
		DDP_TMP_VECTOR_UNINIT(
				stack,
				lu_storage,
				Scalar,
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
			ControlFeedback const& fb_seq, Mults const& mults) const -> MemReq {
		return MemReq::max_of({
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
			DerivativeStorage& derivs,
			ControlFeedback& fb_seq,
			Mults& mults,
			trajectory const& traj,
			Scalar mu,
			Scalar w,
			Scalar n,
			Scalar stopping_threshold,
			DynStackView stack,
			internal::ThreadPerf& threading_method) const
			-> Tuple<MultUpdateAttemptResult_e, Scalar, Scalar> {

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
					MultUpdateAttemptResult_e::optimum_attained,
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
						MultUpdateAttemptResult_e::update_success,
						opt_obj,
						opt_constr,
				};
			} else {
				return {
						direct,
						MultUpdateAttemptResult_e::update_failure,
						opt_obj,
						opt_constr,
				};
			}
		} else {
			return {
					direct,
					MultUpdateAttemptResult_e::no_update,
					opt_obj,
					opt_constr,
			};
		}
	}

	template <Method_e M>
	auto solve(
			i64 max_iterations,
			Scalar optimality_stopping_threshold,
			Scalar mu,
			trajectory traj) -> Tuple<trajectory, ControlFeedback> {

		auto mult = zero_multipliers<M>(traj);
		auto reg = Regularization<Scalar>{0, 1, 2, 1e-5};
		auto ctrl = ControlFeedback(
				traj.index_begin(),
				traj.index_end(),
				dynamics.state_space(),
				dynamics.control_space());

		auto derivs = make_derivative_storage(traj);
		auto traj2 = traj;

		using std::pow;

		Scalar previous_opt_constr{};
		Scalar w = 1 / mu;
		Scalar n = 1 / pow(mu, static_cast<Scalar>(0.1L));

		MemReq req = //
				MemReq::max_of({
						as_ref,
						{
								MemReq::sum_of({
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

		internal::ThreadPerf single{0, 0, false};
		internal::ThreadPerf multi{0, 0, true};
		for (i64 i = 0; i < max_iterations; ++i) {

			internal::ThreadPerf& threading_method = [&]() -> internal::ThreadPerf& {
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

				Scalar const beta = 0.5;
				switch (mult_update_rv) {
				case MultUpdateAttemptResult_e::no_update: {
					break;
				}
				case MultUpdateAttemptResult_e::update_failure: {
					using std::pow;
					::fmt::print(
							"desired new mu {}\n",
							pow(mu / (previous_opt_constr / opt_constr), 1 / (1 - beta)));
					mu = 10 * std::max(     //
												std::min( //
														pow(mu / (previous_opt_constr / opt_constr),
					                      1.0 / (1 - beta)),
														mu * Scalar{1e5}),
												mu);
					break;
				}
				case MultUpdateAttemptResult_e::update_success: {
					using std::pow;
					n = opt_constr / pow(mu, beta / 2);
					w = std::max(w / pow(mu, Scalar{1}), n / mu);
					previous_opt_constr = opt_constr;
					break;
				}
				case MultUpdateAttemptResult_e::optimum_attained:
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

	auto bwd_pass_req() const noexcept -> MemReq {
		i64 nf = dynamics.output_space().max_ddim();
		i64 nx = dynamics.state_space().max_ddim();
		i64 nu = constraint.ref(dynamics).control_space().max_ddim();
		i64 ne = constraint.ref(dynamics).output_space().max_ddim();
		return MemReq::sum_of({
				as_ref,
				{
						{tag<Scalar>, nf},      // vx
						{tag<Scalar>, nf * nf}, // vxx
						{tag<Scalar>, ne},      // tmp
						{tag<Scalar>, ne * nx}, // tmp2
						{tag<Scalar>, nx},      // qx
						{tag<Scalar>, nu},      // qu
						{tag<Scalar>, nx * nx}, // qxx
						{tag<Scalar>, nu * nu}, // quu
						{tag<Scalar>, nu * nx}, // qux
						MemReq::max_of({
								as_ref,
								{
										{tag<Scalar>, nf * nx}, // tmp (qxx/qux)
										{tag<Scalar>, nf * nu}, // tmp (quu)
										{tag<Scalar>, nu * nu}, // quu_clone
								},
						}),
				},
		});
	}
	template <typename Mults>
	auto bwd_pass(
			ControlFeedback& control_feedback,
			Regularization<Scalar>& reg,
			Scalar mu,
			trajectory const& current_traj,
			Mults const& mults,
			DerivativeStorage const& derivatives,
			DynStackView stack) const -> Scalar {

		bool success = false;

		Scalar expected_decrease = 0;
		while (!success) {
			::fmt::print("mu: {}\n", mu);

			i64 nxf = derivatives.lfx().rows();

			DDP_TMP_MATRIX_UNINIT(stack, V_xx, Scalar, nxf, nxf);
			DDP_TMP_VECTOR_UNINIT(stack, V_x, Scalar, nxf);

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

				DDP_TMP_VECTOR_UNINIT(stack, tmp, Scalar, pe.rows());
				DDP_TMP_MATRIX_UNINIT(stack, tmp2, Scalar, pe.rows(), pe_x.cols());

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

				DDP_TMP_VECTOR_UNINIT(stack, Q_x, Scalar, lx.rows());
				eigen::assign(Q_x, lx);
				eigen::mul_add_to_noalias(Q_x, fx.transpose(), v_x);
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_x, eqx.transpose(), tmp);
					eigen::mul_add_to_noalias(Q_x, pe_x.transpose(), eq_);
				}

				DDP_TMP_VECTOR_UNINIT(stack, Q_u, Scalar, lu.rows());
				eigen::assign(Q_u, lu);
				eigen::mul_add_to_noalias(Q_u, fu.transpose(), v_x);
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_u, equ.transpose(), tmp);
				}

				DDP_TMP_MATRIX_UNINIT(stack, Q_xx, Scalar, lxx.rows(), lxx.rows());
				eigen::assign(Q_xx, lxx);
				{
					DDP_TMP_MATRIX(stack, tmp_prod, Scalar, V_xx.rows(), fx.cols());
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

				DDP_TMP_MATRIX_UNINIT(stack, Q_uu, Scalar, luu.rows(), luu.rows());
				eigen::assign(Q_uu, luu);
				{
					DDP_TMP_MATRIX(stack, tmp_prod, Scalar, V_xx.rows(), fu.cols());
					eigen::mul_add_to_noalias(tmp_prod, V_xx, fu);
					eigen::mul_add_to_noalias(Q_uu, fu.transpose(), tmp_prod);
				}
				if (has_eq) {
					eigen::mul_add_to_noalias(Q_uu, equ.transpose(), equ, mu);
					equu.noalias_contract_add_outdim(
							eigen::as_mut(Q_uu), eigen::as_const(tmp));
				}
				fuu.noalias_contract_add_outdim(eigen::as_mut(Q_uu), v_x);

				DDP_TMP_MATRIX_UNINIT(stack, Q_ux, Scalar, lux.rows(), lux.cols());
				eigen::assign(Q_ux, lux);
				{
					DDP_TMP_MATRIX(stack, tmp_prod, Scalar, V_xx.rows(), fx.cols());
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
							stack, Q_uu_clone, Scalar, Q_uu.rows(), Q_uu.rows());
					eigen::assign(Q_uu_clone, Q_uu);
					eigen::add_identity(Q_uu_clone, *reg);

					Eigen::LLT<eigen::View<Scalar, colmat>> llt_res(Q_uu_clone);

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
					DDP_TMP_VECTOR(stack, dotk, Scalar, Q_uu.rows());
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
			-> MemReq {
		return MemReq::sum_of({
				as_ref,
				{
						MemReq{
								tag<Scalar>, 2 * (traj.index_end() - traj.index_begin() + 1)},
						cost_seq_aug_req(mults),
						MemReq{
								tag<Scalar>, constraint.ref(dynamics).state_space().max_ddim()},

						MemReq::max_of({
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
			ControlFeedback const& feedback,
			Scalar mu,
			Key k,
			DynStackView stack,
			bool do_linesearch = true) const -> Tuple<Key, Scalar> {

		auto begin = reference_traj.index_begin();
		auto end = reference_traj.index_end();

		DDP_TMP_VECTOR(stack, costs_old_traj, Scalar, end - begin + 1);
		DDP_TMP_VECTOR(stack, costs_new_traj, Scalar, end - begin + 1);

		if (do_linesearch) {
			k = cost_seq_aug(
					eigen::slice_to_vec(costs_old_traj),
					reference_traj,
					old_mults,
					mu,
					VEG_FWD(k),
					stack);
		}

		Scalar step = 1;
		bool success = false;

		DDP_TMP_VECTOR(
				stack, tmp, Scalar, constraint.ref(dynamics).state_space().max_ddim());

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
	auto cost_seq_aug_req(Mults const& mults) const -> MemReq {
		return MemReq::sum_of({
				as_ref,
				{
						MemReq{
								tag<Scalar>, constraint.ref(dynamics).output_space().max_dim()},
						MemReq{
								tag<Scalar>, constraint.ref(dynamics).output_space().max_dim()},

						MemReq::max_of({
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
			View<Scalar, colvec> out,
			trajectory const& traj,
			Mults const& mults,
			Scalar mu,
			Key k,
			DynStackView stack) const -> Key {

		auto csp = constraint.ref(dynamics).output_space();

		DDP_TMP_VECTOR_UNINIT(stack, ce_storage, Scalar, csp.max_dim());
		DDP_TMP_VECTOR_UNINIT(stack, pe_storage, Scalar, csp.max_dim());

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

namespace nb {
struct ddp {
	template <typename Dynamics, typename Cost, typename Constraint>
	auto
	operator()(Dynamics&& dynamics, Cost&& cost, Constraint&& constraint) const
			-> Ddp<Dynamics, Cost, Constraint> {
		return {
				VEG_FWD(dynamics),
				VEG_FWD(cost),
				VEG_FWD(constraint),
		};
	}
};
} // namespace nb
VEG_NIEBLOID(ddp);

} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_DDP_HPP_CAAKIEJQS */
