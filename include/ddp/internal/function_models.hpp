#ifndef DDP_PINOCCHIO_FUNCTION_MODELS_HPP_EPD2SYHQS
#define DDP_PINOCCHIO_FUNCTION_MODELS_HPP_EPD2SYHQS

#include "ddp/internal/matrix_seq.hpp"
#include "ddp/space.hpp"
#include "veg/internal/prologue.hpp"

namespace ddp {
namespace internal {

template <typename Scalar, typename State_Space>
struct affine_function_seq {
	using state_space = State_Space;
	using scalar = Scalar;

	static_assert(VEG_CONCEPT(same<scalar, typename State_Space::scalar>), "");

	struct layout {
		mat_seq<scalar, colvec> origin;
		mat_seq<scalar, colvec> val;
		mat_seq<scalar, colmat> jac;
		state_space in;
	} self;

	explicit affine_function_seq(layout l) : self{VEG_FWD(l)} {}

	template <typename Output_Space>
	affine_function_seq(i64 begin, i64 end, state_space in, Output_Space out)
			: self{
						mat_seq<scalar, colvec>{ddp::space_to_idx(in, begin, end)},
						mat_seq<scalar, colvec>{ddp::space_to_idx(out, begin, end)},
						mat_seq<scalar, colmat>{
								{begin,
	               end,
	               [&](i64 t) {
									 return idx::dims<colmat>{out.dim(t), in.dim(t)};
								 }}},
						VEG_FWD(in),
				} {}

	auto update_origin_req() const -> mem_req {
		return mem_req::sum_of({
				as_ref,
				{
						{
								tag<scalar>,
								(self.in.max_ddim()                                  // diff
		             + self.in.max_ddim() * self.in.max_ddim()           // diff_jac
		             + self.jac.self.idx.max_rows() * self.in.max_ddim() // tmp
		             ),
						},

						mem_req::max_of({
								as_ref,
								{
										self.in.difference_req(),
										self.in.d_difference_d_finish_req(),
								},
						}),
				},
		});
	}

	auto eval_to_req() const -> mem_req {
		return mem_req::sum_of({
				as_ref,
				{
						{tag<scalar>, self.in.max_ddim()},
						self.in.difference_req(),
				},
		});
	}
	void eval_to(
			view<scalar, colvec> out,
			i64 t,
			view<scalar const, colvec> in,
			DynStackView stack) const {
		DDP_TMP_VECTOR_UNINIT(stack, tmp, scalar, self.jac[t].cols());

		self.in.difference(tmp, t, self.origin[t], in, stack);
		if (out.rows() > 0) {
			eigen::assign(out, self.val[t]);
			eigen::mul_add_to_noalias(out, self.jac[t], tmp);
		}
	}

	void
	update_origin(mat_seq<scalar, colvec> const& new_traj, DynStackView stack) {

		auto begin = self.origin.index_begin();
		auto end = self.origin.index_end();

		VEG_DEBUG_ASSERT_ALL_OF(
				(new_traj.index_end() == end + 1), (new_traj.index_begin() == begin));

		for (i64 t = begin; t < end; ++t) {
			auto origin = self.origin[t];
			auto val = self.val[t];
			auto jac = self.jac[t];

			auto new_origin = new_traj[t];

			if (val.rows() > 0) {
				auto ndo = jac.rows();
				auto ndi = jac.cols();

				DDP_TMP_VECTOR_UNINIT(stack, diff, scalar, ndi);
				DDP_TMP_MATRIX_UNINIT(stack, diff_jac, scalar, ndi, ndi);
				DDP_TMP_MATRIX(stack, tmp, scalar, ndo, ndi);

				self.in.difference(diff, t, eigen::as_const(origin), new_origin, stack);
				self.in.d_difference_d_finish(
						diff_jac, t, eigen::as_const(origin), new_origin, stack);

				eigen::mul_add_to_noalias(val, jac, diff);

				eigen::mul_add_to_noalias(tmp, jac, diff_jac);
				eigen::assign(jac, tmp);
			}
			eigen::assign(origin, new_origin);
		}
	}
};

template <typename Scalar>
struct constant_function_seq {
	using scalar = Scalar;

	mat_seq<scalar, colvec> val;

	template <typename Output_Space>
	constant_function_seq(i64 begin, i64 end, Output_Space out)
			: val{ddp::space_to_idx(out, begin, end)} {}

	auto update_origin_req() const -> mem_req { return {tag<scalar>, 0}; }
	auto eval_to_req() const -> mem_req { return {tag<scalar>, 0}; }

	void eval_to(
			view<scalar, colvec> out,
			i64 t,
			view<scalar const, colvec> in,
			DynStackView stack) const {

		unused(stack, in, t);
		eigen::assign(out, val[t]);
	}

	void
	update_origin(mat_seq<scalar, colvec> const& new_traj, DynStackView stack) {
		unused(this, new_traj, stack);
	}
};

} // namespace internal
} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_FUNCTION_MODELS_HPP_EPD2SYHQS */
