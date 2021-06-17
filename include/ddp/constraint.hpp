#ifndef DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS
#define DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS

#include "ddp/dynamics.hpp"
#include "veg/internal/prologue.hpp"

namespace ddp {

template <typename U>
auto check_eval_to(
		U const& self,
		View<typename U::Scalar, colvec> out,
		i64 t,
		View<typename U::Scalar const, colvec> x,
		View<typename U::Scalar const, colvec> u) noexcept {
	unused(self, out, t, x, u);

	VEG_DEBUG_ASSERT_ALL_OF( //
			(x.rows() == self.state_space().dim(t)),
			(u.rows() == self.control_space().dim(t)),
			(out.rows() == self.output_space().ddim(t)));
}
template <typename U>
auto check_d_eval_to(
		U const& self,
		View<typename U::Scalar, colmat> out_x,
		View<typename U::Scalar, colmat> out_u,
		View<typename U::Scalar, colvec> out,
		i64 t,
		View<typename U::Scalar const, colvec> x,
		View<typename U::Scalar const, colvec> u) noexcept {
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

namespace constraint {
template <typename Dynamics>
struct Null {
	struct Ref {
		using Key = typename Dynamics::Key;
		using Scalar = typename Dynamics::Scalar;

		typename Dynamics::Ref dynamics_ref;

		auto dynamics() const -> typename Dynamics::Ref { return dynamics_ref; }

		auto output_space() const noexcept -> VectorSpace<Scalar> {
			return VectorSpace<Scalar>{0};
		}
		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> MemReq { return MemReq{tag<Scalar>, 0}; }

		auto eval_to(
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {
			unused(k, stack);
			ddp::check_eval_to(*this, out, t, x, u);

			return k;
		}

		auto d_eval_to_req() const -> MemReq { return eval_to_req(); }

		auto d_eval_to(
				View<Scalar, colmat> out_x,
				View<Scalar, colmat> out_u,
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

			unused(k, stack);
			ddp::check_d_eval_to(*this, out_x, out_u, out, t, x, u);

			return k;
		}
	};

	auto ref(Dynamics const& dynamics) const noexcept -> Ref {
		return {dynamics};
	}
};

template <typename Dynamics, typename Fn>
struct Config {
	Fn target_generator;
	i64 max_dim;
	MemReq generator_mem_req;

	struct Ref {
		using Key = typename Dynamics::Key;
		using Scalar = typename Dynamics::Scalar;

		typename Dynamics::Ref dynamics_ref;
		Config const& constraint_ref;

		auto dynamics() const -> typename Dynamics::Ref { return dynamics_ref; }

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

		auto output_space() const noexcept -> BasicVectorSpace<Scalar, dim_fn> {
			return {{{constraint_ref.target_generator}, constraint_ref.max_dim}};
		}

		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> MemReq {
			return constraint_ref.generator_mem_req;
		}

		auto eval_to(
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {
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

		auto d_eval_to_req() const -> MemReq { return eval_to_req(); }

		auto d_eval_to(
				View<Scalar, colmat> out_x,
				View<Scalar, colmat> out_u,
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

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

	auto ref(Dynamics const& dynamics) const noexcept -> Ref {
		return {dynamics, *this};
	}
};

template <typename Dynamics, typename Fn>
struct Velocity {
	Fn target_generator;
	i64 max_dim;
	MemReq generator_mem_req;

	struct Ref {
		using Key = typename Dynamics::Key;
		using Scalar = typename Dynamics::Scalar;

		typename Dynamics::Ref dynamics_ref;
		Velocity const& constraint_ref;

		auto dynamics() const -> typename Dynamics::Ref { return dynamics; }

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

		auto output_space() const noexcept -> BasicVectorSpace<Scalar, dim_fn> {
			return {{constraint_ref.target_generator, constraint_ref.max_dim}};
		}

		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> MemReq {
			return constraint_ref.generator_mem_req;
		}

		auto eval_to(
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {
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

		auto d_eval_to_req() const -> MemReq { return eval_to_req(); }

		auto d_eval_to(
				View<Scalar, colmat> out_x,
				View<Scalar, colmat> out_u,
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {
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
struct Spatial {
	Fn target_generator;
	i64 max_dim;
	MemReq generator_mem_req;
	std::vector<i64> frame_ids;

	struct Ref {
		using Scalar = typename Dynamics::Scalar;
		using Key = typename Dynamics::Key;

		typename Dynamics::Ref dynamics_ref;
		Fn const& target_generator;
		i64 max_dim;
		MemReq generator_mem_req;
		Slice<i64 const> frame_ids;

		auto dynamics() const -> typename Dynamics::Ref { return dynamics_ref; }

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

		auto output_space() const noexcept -> BasicVectorSpace<Scalar, dim_fn> {
			return {{{target_generator, frame_ids}, max_dim}};
		}
		auto state_space() const VEG_DEDUCE_RET(dynamics().state_space());
		auto control_space() const VEG_DEDUCE_RET(dynamics().control_space());

		auto eval_to_req() const -> MemReq { return generator_mem_req; }
		auto d_eval_to_req() const -> MemReq { return eval_to_req(); }

		auto eval_to(
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {
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
				View<Scalar, colmat> out_x,
				View<Scalar, colmat> out_u,
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

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

	auto ref(Dynamics const& dynamics) const noexcept -> Ref {
		return {dynamics, target_generator, max_dim, generator_mem_req, frame_ids};
	}
};

template <typename Constraint>
struct AdvanceTime {
	Constraint constr;

	struct Ref {
		using Key = typename Constraint::Ref::Key;
		using Scalar = typename Constraint::Ref::Scalar;

		typename Constraint::Ref constr;

		auto dynamics() const VEG_DEDUCE_RET(constr.dynamics());

		struct dim_fn {
			typename Constraint::Ref constr;
			auto operator()(i64 t) const noexcept -> i64 {
				return constr.output_space().dim(t + 1);
			}
		};
		auto output_space() const noexcept -> BasicVectorSpace<Scalar, dim_fn> {
			return {{{constr}, constr.output_space().max_dim()}};
		}
		auto state_space() const VEG_DEDUCE_RET(constr.state_space());
		auto control_space() const VEG_DEDUCE_RET(constr.control_space());

		auto dim(i64 t) const -> i64 { return constr.dim(t + 1); }
		auto max_dim() const -> i64 { return constr.max_dim(); }

		auto eval_to_req() const -> MemReq {
			return MemReq::sum_of({
					as_ref,
					{
							MemReq::max_of({
									as_ref,
									{
											constr.eval_to_req(),
											dynamics().eval_to_req(),
									},
							}),

							{tag<Scalar>, dynamics().state_space().max_dim()},
					},
			});
		}
		auto eval_to(
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {
			ddp::check_eval_to(*this, out, t, x, u);

			auto nde = output_space().ddim(t);
			unused(nde);
			VEG_DEBUG_ASSERT(out.rows() == nde);
			if (nde == 0) {
				return k;
			}

			DDP_TMP_VECTOR_UNINIT(
					stack, x_n, Scalar, dynamics().output_space().dim(t));
			k = dynamics().eval_to(x_n, t, x, u, VEG_FWD(k), stack);
			k = constr.eval_to(
					out, t + 1, eigen::as_const(x_n), u, VEG_FWD(k), stack);
			return k;
		}

		auto d_eval_to_req() const -> MemReq {
			auto nx = dynamics().state_space().max_dim();
			auto ndx = dynamics().state_space().max_ddim();
			auto ndu = dynamics().control_space().max_ddim();
			auto nde = output_space().max_ddim();

			return MemReq::sum_of({
					as_ref,
					{
							MemReq::max_of({
									as_ref,
									{
											constr.d_eval_to_req(),
											dynamics().d_eval_to_req(),
									},
							}),

							{
									tag<Scalar>,
									(nx          //
			             + ndx       //
			             + ndx * ndx //
			             + ndx * ndu //
			             + nde * ndx //
			             + nde * ndu //
			             ),
							},
					},
			});
		}

		auto d_eval_to(
				View<Scalar, colmat> out_x,
				View<Scalar, colmat> out_u,
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

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

			DDP_TMP_VECTOR_UNINIT(stack, x_n, Scalar, nx);
			DDP_TMP_MATRIX_UNINIT(stack, fx_n, Scalar, ndx, ndx);
			DDP_TMP_MATRIX_UNINIT(stack, fu_n, Scalar, ndx, ndu1);

			k = dynamics().d_eval_to(fx_n, fu_n, x_n, t, x, u, VEG_FWD(k), stack);

			DDP_TMP_MATRIX_UNINIT(stack, eq_n_x, Scalar, nde, ndx);
			DDP_TMP_MATRIX_UNINIT(stack, eq_n_u, Scalar, nde, ndu2);

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
	auto ref(Dynamics const& dynamics) const noexcept -> Ref {
		return {constr.ref(dynamics)};
	}
};

template <typename Constr1, typename Constr2>
struct Concat {
	Constr1 constr1;
	Constr2 constr2;

	struct Ref {
		using Key = typename Constr1::Ref::Key;
		using Scalar = typename Constr1::Ref::Scalar;

		typename Constr1::Ref constr1;
		typename Constr2::Ref constr2;

		auto dynamics() const -> decltype(auto) { return constr1.dynamics(); }

		auto state_space() const { return constr1.state_space(); }
		auto control_space() const { return constr1.control_space(); }
		auto output_space() const { return constr1.output_space(); }

		auto eval_to_req() const -> MemReq {
			return MemReq::max_of({
					constr1.eval_to_req(),
					constr2.eval_to_req(),
			});
		}
		auto eval_to(
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

			VEG_BIND(
					auto,
					(out1, out2),
					eigen::split_at_row(out, constr1.output_space().dim(t)));

			k = constr1.eval_to(out1, t, x, u, VEG_FWD(k), stack);
			k = constr2.eval_to(out2, t, x, u, VEG_FWD(k), stack);

			return k;
		}

		auto d_eval_to_req() const -> MemReq {
			return MemReq::max_of({
					constr1.d_eval_to_req(),
					constr2.d_eval_to_req(),
			});
		}

		auto d_eval_to(
				View<Scalar, colmat> out_x,
				View<Scalar, colmat> out_u,
				View<Scalar, colvec> out,
				i64 t,
				View<Scalar const, colvec> x,
				View<Scalar const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

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
	auto ref(Dynamics const& dynamics) const noexcept -> Ref {
		return {constr1.ref(dynamics), constr2.ref(dynamics)};
	}
};

namespace nb {
struct null {
	template <typename Dynamics>
	auto operator()(Dynamics const& /*dynamics*/
	) const -> Null<Dynamics> {
		return {};
	}
};
struct config {
	template <typename Dynamics, typename Fn>
	auto operator()(
			Dynamics const& /*dynamics*/,
			Fn target_gen,
			i64 max_dim,
			MemReq gen_mem_req) const -> Config<Dynamics, Fn> {
		return {
				VEG_FWD(target_gen),
				max_dim,
				gen_mem_req,
		};
	}
};
template <i64 N>
struct advance_time {
	static_assert(N > 0, "");
	template <typename Constraint>
	auto operator()(Constraint&& constr) const //
			-> AdvanceTime<decltype(advance_time<N - 1>{}(VEG_FWD(constr)))> {
		return {advance_time<N - 1>{}(VEG_FWD(constr))};
	}
};

template <>
struct advance_time<0> {
	template <typename Constraint>
	auto operator()(Constraint&& constr) const -> Constraint {
		return VEG_FWD(constr);
	}
};
} // namespace nb
VEG_NIEBLOID(null);
VEG_NIEBLOID(config);
VEG_NIEBLOID_TEMPLATE(i64 N, advance_time, N);
} // namespace constraint
} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS */
