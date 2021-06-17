#ifndef DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS
#define DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS

#include "ddp/dynamics.hpp"
#include "veg/internal/prologue.hpp"

namespace ddp {

template <typename U>
auto check_eval_to(
		U const& self,
		view<typename U::scalar, colvec> out,
		i64 t,
		view<typename U::scalar const, colvec> x,
		view<typename U::scalar const, colvec> u) noexcept {
	unused(self, out, t, x, u);

	VEG_DEBUG_ASSERT_ALL_OF( //
			(x.rows() == self.state_space().dim(t)),
			(u.rows() == self.control_space().dim(t)),
			(out.rows() == self.output_space().ddim(t)));
}
template <typename U>
auto check_d_eval_to(
		U const& self,
		view<typename U::scalar, colmat> out_x,
		view<typename U::scalar, colmat> out_u,
		view<typename U::scalar, colvec> out,
		i64 t,
		view<typename U::scalar const, colvec> x,
		view<typename U::scalar const, colvec> u) noexcept {
	unused(self, out_x, out_u, out, t, x, u);

	VEG_DEBUG_ASSERT_ALL_OF( //
			(x.rows() == self.state_space().dim(t)),
			(u.rows() == self.control_space().dim(t)),
			(out_x.rows() == self.output_space().ddim(t)),
			(out_x.cols() == self.state_space().ddim(t)),
			(out_u.rows() == self.output_space().ddim(t)),
			(out_u.cols() == self.control_space().ddim(t)),
			(out.rows() == self.output_space().ddim(t)));
}

template <typename Dynamics>
struct no_constraint {
	struct ref_type {
		using key = typename Dynamics::key;
		using scalar = typename Dynamics::scalar;

		typename Dynamics::ref_type dynamics_ref;

		auto dynamics() const -> typename Dynamics::ref_type {
			return dynamics_ref;
		}

		auto output_space() const noexcept -> vector_space<scalar> {
			return vector_space<scalar>{0};
		}
		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> mem_req { return mem_req{tag<scalar>, 0}; }

		auto eval_to(
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {
			unused(k, stack);
			ddp::check_eval_to(*this, out, t, x, u);

			return k;
		}

		auto d_eval_to_req() const -> mem_req { return eval_to_req(); }

		auto d_eval_to(
				view<scalar, colmat> out_x,
				view<scalar, colmat> out_u,
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {

			unused(k, stack);
			ddp::check_d_eval_to(*this, out_x, out_u, out, t, x, u);

			return k;
		}
	};

	auto ref(Dynamics const& dynamics) const noexcept -> ref_type {
		return {dynamics};
	}
};

template <typename Dynamics, typename Fn>
struct config_constraint {
	Fn target_generator;
	i64 max_dim;
	mem_req generator_mem_req;

	struct ref_type {
		using key = typename Dynamics::key;
		using scalar = typename Dynamics::scalar;

		typename Dynamics::ref_type dynamics_ref;
		config_constraint const& constraint_ref;

		auto dynamics() const -> typename Dynamics::ref_type {
			return dynamics_ref;
		}

		struct dim_fn {
			Fn const& gen;
			auto operator()(i64 t) const noexcept -> i64 {
				DynStackView stack{Slice<char>{
						from_raw_parts,
						nullptr,
						0,
						unsafe,
				}};
				return gen(t, stack).rows();
			}
		};

		auto output_space() const noexcept -> basic_vector_space<scalar, dim_fn> {
			return {{{constraint_ref.target_generator}, constraint_ref.max_dim}};
		}

		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> mem_req {
			return constraint_ref.generator_mem_req;
		}

		auto eval_to(
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {
			unused(u);
			ddp::check_eval_to(*this, out, t, x, u);

			auto _target = constraint_ref.target_generator(t, stack);
			auto target = eigen::slice_to_vec(_target);

			if (target.rows() == 0) {
				return k;
			}
			VEG_DEBUG_ASSERT(target.rows() == dynamics().model.config_dim());
			auto nq = dynamics().model.config_dim();
			dynamics().model.difference(out, target, eigen::split_at_row(x, nq)[0_c]);
			return k;
		}

		auto d_eval_to_req() const -> mem_req { return eval_to_req(); }

		auto d_eval_to(
				view<scalar, colmat> out_x,
				view<scalar, colmat> out_u,
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {

			ddp::check_d_eval_to(*this, out_x, out_u, out, t, x, u);

			auto _target = constraint_ref.target_generator(t, stack);
			auto target = eigen::slice_to_vec(_target);

			if (target.rows() == 0) {
				return k;
			}
			VEG_DEBUG_ASSERT(target.rows() == dynamics().model.config_dim());

			auto nq = dynamics().model.config_dim();
			auto nv = dynamics().model.tangent_dim();

			auto xq = eigen::split_at_row(x, nq)[0_c];

			VEG_BIND(auto, (out_xq, out_xv), eigen::split_at_col(out_x, nv));
			unused(u, out_xv);

			out_x.setZero();
			out_u.setZero();

			dynamics().model.difference(out, target, xq);
			dynamics().model.d_difference_dq_finish(out_xq, target, xq);

			return k;
		}
	};

	auto ref(Dynamics const& dynamics) const noexcept -> ref_type {
		return {dynamics, *this};
	}
};

template <typename Dynamics, typename Fn>
struct velocity_constraint {
	Fn target_generator;
	i64 max_dim;
	mem_req generator_mem_req;

	struct ref_type {
		using key = typename Dynamics::key;
		using scalar = typename Dynamics::scalar;

		typename Dynamics::ref_type dynamics_ref;
		velocity_constraint const& constraint_ref;

		auto dynamics() const -> typename Dynamics::ref_type { return dynamics; }

		struct dim_fn {
			Fn const& gen;
			auto operator()(i64 t) const noexcept -> i64 {
				DynStackView stack{Slice<char>{
						from_raw_parts,
						nullptr,
						0,
						unsafe,
				}};
				return gen(t, stack).rows();
			}
		};

		auto output_space() const noexcept -> basic_vector_space<scalar, dim_fn> {
			return {{constraint_ref.target_generator, constraint_ref.max_dim}};
		}

		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> mem_req {
			return constraint_ref.generator_mem_req;
		}

		auto eval_to(
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {
			ddp::check_eval_to(*this, out, t, x, u);

			auto _target = constraint_ref.target_generator(t, stack);
			auto target = eigen::slice_to_vec(_target);

			if (target.rows() == 0) {
				return k;
			}

			VEG_DEBUG_ASSERT(target.rows() == dynamics().model.tangent_dim());

			unused(u);
			auto nq = dynamics().model.config_dim();

			eigen::sub_to(out, target, eigen::split_at_row(x, nq)[1_c]);

			return k;
		}

		auto d_eval_to_req() const -> mem_req { return eval_to_req(); }

		auto d_eval_to(
				view<scalar, colmat> out_x,
				view<scalar, colmat> out_u,
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {
			ddp::check_d_eval_to(*this, out_x, out_u, out, t, x, u);

			auto _target = constraint_ref.target_generator(t, stack);
			auto target = eigen::slice_to_vec(_target);

			if (target.rows() == 0) {
				return k;
			}

			VEG_DEBUG_ASSERT(target.rows() == dynamics().model.tangent_dim());

			unused(u);
			auto nq = dynamics().model.config_dim();
			auto nv = dynamics().model.tangent_dim();

			VEG_BIND(auto, (out_xq, out_xv), eigen::split_at_col(out_x, nv));

			out_xq.setZero();
			out_u.setZero();
			out_xv.setIdentity();

			eigen::sub_to(out, target, eigen::split_at_row(x, nq)[1_c]);

			return k;
		}
	};
};

template <typename Dynamics, typename Fn>
struct spatial_constraint {
	Fn target_generator;
	i64 max_dim;
	mem_req generator_mem_req;
	std::vector<i64> frame_ids;

	struct ref_type {
		using scalar = typename Dynamics::scalar;
		using key = typename Dynamics::key;

		typename Dynamics::ref_type dynamics_ref;
		Fn const& target_generator;
		i64 max_dim;
		mem_req generator_mem_req;
		Slice<i64 const> frame_ids;

		auto dynamics() const -> typename Dynamics::ref_type {
			return dynamics_ref;
		}

		struct dim_fn {
			Fn const& gen;
			Slice<i64 const> frame_ids;
			auto operator()(i64 t) const noexcept -> i64 {
				DynStackView stack{Slice<char>{
						from_raw_parts,
						nullptr,
						0,
						unsafe,
				}};
				i64 dim = 0;
				for (i64 i = 0; i < frame_ids.size(); ++i) {
					dim += gen(i, t, stack).rows();
				}
				return dim;
			}
		};

		auto output_space() const noexcept -> basic_vector_space<scalar, dim_fn> {
			return {{{target_generator, frame_ids}, max_dim}};
		}
		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> mem_req { return generator_mem_req; }
		auto d_eval_to_req() const -> mem_req { return eval_to_req(); }

		auto eval_to(
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {
			ddp::check_eval_to(*this, out, t, x, u);

			auto nq = dynamics().model.config_dim();
			VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));
			bool computed = false;

			for (i64 i = 0; i < frame_ids.size(); ++i) {

				auto const& _target = target_generator(i, t, stack);
				auto target = eigen::slice_to_vec(_target);
				if (target.rows() == 0) {
					continue;
				}

				VEG_BIND(auto, (out_head, out_tail), eigen::split_at_row(out, 3 * i));
				VEG_BIND(auto, (out_3, out_rest), eigen::split_at_row(out_tail, 3));
				unused(u, v, out_head, out_rest);

				if (!computed) {
					k = dynamics().model.frame_coordinates_precompute(q, VEG_FWD(k));
					computed = true;
				}

				k = dynamics()._model.frame_coordinates(
						out_3, frame_ids[i], VEG_FWD(k));
				eigen::sub_to(out_3, out_3, target);
			}

			return k;
		}

		auto d_eval_to(
				view<scalar, colmat> out_x,
				view<scalar, colmat> out_u,
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {

			ddp::check_d_eval_to(*this, out_x, out_u, out, t, x, u);

			auto nq = dynamics().model.config_dim();
			auto nv = dynamics().model.tangent_dim();
			VEG_BIND(auto, (q, v), eigen::split_at_row(x, nq));
			out_u.setZero();

			bool computed = false;
			for (i64 i = 0; i < narrow<i64>(frame_ids.size()); ++i) {
				auto _target = target_generator(i, t, stack);
				auto target = eigen::slice_to_vec(_target);
				if (target.rows() == 0) {
					continue;
				}

				VEG_BIND(auto, (out_head, out_tail), eigen::split_at_row(out, 3 * i));
				VEG_BIND(
						auto, (out_head_x, out_tail_x), eigen::split_at_row(out_x, 3 * i));

				VEG_BIND(
						auto, (out_3x, out_rest_x), eigen::split_at_row(out_tail_x, 3));
				VEG_BIND(auto, (out_3, out_rest), eigen::split_at_row(out_tail, 3));
				unused(u, v, out_head, out_rest, out_head_x, out_rest_x);

				VEG_BIND(auto, (out_3q, out_3v), eigen::split_at_col(out_3x, nv));

				if (!computed) {
					k = dynamics().model.dframe_coordinates_precompute(q, VEG_FWD(k));
					computed = true;
				}

				k = dynamics().model.frame_coordinates(
						out_3, frame_ids[narrow<usize>(i)], VEG_FWD(k));
				eigen::sub_to(out_3, out_3, target);

				k = dynamics().model.d_frame_coordinates(
						out_3q, frame_ids[narrow<usize>(i)], VEG_FWD(k));
				out_3v.setZero();
			}

			return k;
		}
	};

	auto ref(Dynamics const& dynamics) const noexcept -> ref_type {
		return {dynamics, target_generator, max_dim, generator_mem_req, frame_ids};
	}
};

template <typename Constraint>
struct constraint_advance_time {
	Constraint constr;

	struct ref_type {
		using key = typename Constraint::ref_type::key;
		using scalar = typename Constraint::ref_type::scalar;

		typename Constraint::ref_type constr;

		auto dynamics() const VEG_DEDUCE_RET(constr.dynamics());

		struct dim_fn {
			typename Constraint::ref_type constr;
			auto operator()(i64 t) const noexcept -> i64 {
				return constr.output_space().dim(t + 1);
			}
		};
		auto output_space() const noexcept -> basic_vector_space<scalar, dim_fn> {
			return {{{constr}, constr.output_space().max_dim()}};
		}
		auto state_space() const VEG_DEDUCE_RET(constr.state_space());
		auto control_space() const VEG_DEDUCE_RET(constr.control_space());

		auto dim(i64 t) const -> i64 { return constr.dim(t + 1); }
		auto max_dim() const -> i64 { return constr.max_dim(); }

		auto eval_to_req() const -> mem_req {
			return mem_req::sum_of({
					mem_req::max_of({
							constr.eval_to_req(),
							dynamics().eval_to_req(),
					}),

					{tag<scalar>, dynamics().state_space().max_dim()},
			});
		}
		auto eval_to(
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {
			ddp::check_eval_to(*this, out, t, x, u);

			auto nde = output_space().ddim(t);
			unused(nde);
			VEG_DEBUG_ASSERT(out.rows() == nde);
			if (nde == 0) {
				return k;
			}

			DDP_TMP_VECTOR_UNINIT(
					stack, x_n, scalar, dynamics().output_space().dim(t));
			k = dynamics().eval_to(x_n, t, x, u, VEG_FWD(k), stack);
			k = constr.eval_to(
					out, t + 1, eigen::as_const(x_n), u, VEG_FWD(k), stack);
			return k;
		}

		auto d_eval_to_req() const -> mem_req {
			auto nx = dynamics().state_space().max_dim();
			auto ndx = dynamics().state_space().max_ddim();
			auto ndu = dynamics().control_space().max_ddim();
			auto nde = output_space().max_ddim();

			return mem_req::sum_of({

					mem_req::max_of({
							constr.d_eval_to_req(),
							dynamics().d_eval_to_req(),
					}),

					{tag<scalar>,
			     (nx          //
			      + ndx       //
			      + ndx * ndx //
			      + ndx * ndu //
			      + nde * ndx //
			      + nde * ndu //
			      )},
			});
		}

		auto d_eval_to(
				view<scalar, colmat> out_x,
				view<scalar, colmat> out_u,
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {

			auto nx = dynamics().output_space().dim(t);
			auto ndx = dynamics().output_space().ddim(t);
			auto ndu1 = dynamics().control_space().ddim(t);
			auto ndu2 = dynamics().control_space().ddim(t + 1);
			auto nde = output_space().ddim(t);

			VEG_DEBUG_ASSERT_ALL_OF(
					out.rows() == nde,
					out_x.rows() == nde,
					out_u.rows() == nde,
					out_x.cols() == ndx,
					out_u.cols() == ndu1);
			if (nde == 0) {
				return k;
			}

			DDP_TMP_VECTOR_UNINIT(stack, x_n, scalar, nx);
			DDP_TMP_MATRIX_UNINIT(stack, fx_n, scalar, ndx, ndx);
			DDP_TMP_MATRIX_UNINIT(stack, fu_n, scalar, ndx, ndu1);

			k = dynamics().d_eval_to(fx_n, fu_n, x_n, t, x, u, VEG_FWD(k), stack);

			DDP_TMP_MATRIX_UNINIT(stack, eq_n_x, scalar, nde, ndx);
			DDP_TMP_MATRIX_UNINIT(stack, eq_n_u, scalar, nde, ndu2);

			k = constr.d_eval_to(
					eq_n_x,
					eq_n_u,
					out,
					t + 1,
					eigen::as_const(x_n),
					u,
					VEG_FWD(k),
					stack);

			VEG_DEBUG_ASSERT_ELSE(
					(::fmt::print("{}\n", eq_n_u),
			     "control should have no effect on the constraint value"),
					eq_n_u.isConstant(0));

			out_x.setZero();
			out_u.setZero();

			eigen::mul_add_to_noalias(out_x, eq_n_x, fx_n);
			eigen::mul_add_to_noalias(out_u, eq_n_x, fu_n);

			return k;
		}
	};

	template <typename Dynamics>
	auto ref(Dynamics const& dynamics) const noexcept -> ref_type {
		return {constr.ref(dynamics)};
	}
};

template <typename Constr1, typename Constr2>
struct concat_constraint {
	Constr1 constr1;
	Constr2 constr2;

	struct ref_type {
		using key = typename Constr1::ref_type::key;
		using scalar = typename Constr1::ref_type::scalar;

		typename Constr1::ref_type constr1;
		typename Constr2::ref_type constr2;

		auto dynamics() const -> decltype(auto) { return constr1.dynamics(); }

		auto state_space() const { return constr1.state_space(); }
		auto control_space() const { return constr1.control_space(); }
		auto output_space() const { return constr1.output_space(); }

		auto eval_to_req() const -> mem_req {
			return mem_req::max_of({
					constr1.eval_to_req(),
					constr2.eval_to_req(),
			});
		}
		auto eval_to(
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {

			VEG_BIND(
					auto,
					(out1, out2),
					eigen::split_at_row(out, constr1.output_space().dim(t)));

			k = constr1.eval_to(out1, t, x, u, VEG_FWD(k), stack);
			k = constr2.eval_to(out2, t, x, u, VEG_FWD(k), stack);

			return k;
		}

		auto d_eval_to_req() const -> mem_req {
			return mem_req::max_of({
					constr1.d_eval_to_req(),
					constr2.d_eval_to_req(),
			});
		}

		auto d_eval_to(
				view<scalar, colmat> out_x,
				view<scalar, colmat> out_u,
				view<scalar, colvec> out,
				i64 t,
				view<scalar const, colvec> x,
				view<scalar const, colvec> u,
				key k,
				DynStackView stack) const -> key {

			VEG_BIND(
					auto,
					(out1, out2),
					eigen::split_at_row(out, constr1.output_space().dim(t)));
			VEG_BIND(
					auto,
					(out_x1, out_x2),
					eigen::split_at_row(out_x, constr1.output_space().ddim(t)));
			VEG_BIND(
					auto,
					(out_u1, out_u2),
					eigen::split_at_row(out_u, constr1.output_space().ddim(t)));

			k = constr1.d_eval_to(out_x1, out_u1, out1, t, x, u, VEG_FWD(k), stack);
			k = constr2.d_eval_to(out_x2, out_u2, out2, t, x, u, VEG_FWD(k), stack);

			return k;
		}
	};

	template <typename Dynamics>
	auto ref(Dynamics const& dynamics) const noexcept -> ref_type {
		return {constr1.ref(dynamics), constr2.ref(dynamics)};
	}
};

namespace make {
namespace fn {
struct config_constraint {
	template <typename Dynamics, typename Fn>
	auto operator()(
			Dynamics const& /*dynamics*/,
			Fn target_gen,
			i64 max_dim,
			mem_req gen_mem_req) const -> ddp::config_constraint<Dynamics, Fn> {
		return {
				VEG_FWD(target_gen),
				max_dim,
				gen_mem_req,
		};
	}
};
template <i64 N>
struct constraint_advance_time {
	static_assert(N > 0, "");
	template <typename Constraint>
	auto operator()(Constraint&& constr) const //
			-> ddp::constraint_advance_time<
					decltype(constraint_advance_time<N - 1>{}(VEG_FWD(constr)))> {
		return {constraint_advance_time<N - 1>{}(VEG_FWD(constr))};
	}
};

template <>
struct constraint_advance_time<0> {
	template <typename Constraint>
	auto operator()(Constraint&& constr) const -> Constraint {
		return VEG_FWD(constr);
	}
};
} // namespace fn
VEG_INLINE_VAR(config_constraint, fn::config_constraint);
namespace {
template <i64 N>
constexpr auto const& constraint_advance_time =
		meta::static_const<fn::constraint_advance_time<N>>::value;
} // namespace
} // namespace make

} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS */
