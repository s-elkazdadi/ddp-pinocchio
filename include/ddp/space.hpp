#ifndef DDP_PINOCCHIO_SPACE_HPP_3K0MQGEZS
#define DDP_PINOCCHIO_SPACE_HPP_3K0MQGEZS

#include "ddp/pinocchio_model.hpp"
#include "ddp/internal/matrix_seq.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace ddp {

template <typename Space>
auto space_to_idx(Space space, i64 begin, i64 end) -> idx::idx<colvec> {
	return {begin, end, [&](i64 t) { return idx::dims<colvec>{space.dim(t)}; }};
}

template <typename T, typename Parent>
struct vector_space_from_parent {
	Parent parent;
	using scalar = T;

	auto tangent() const { return *this; }

	auto dim(i64 t) const { return parent.dim(t); }
	auto ddim(i64 t) const { return dim(t); }
	auto max_dim() const { return parent.max_dim(); }
	auto max_ddim() const { return max_dim(); }

	auto integrate_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	void integrate(
			view<T, colvec> x_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			dynamic_stack_view stack) const {
		(void)t, (void)stack;
		VEG_DEBUG_ASSERT_ALL_OF( //
				(x_out.rows() == dim(t)),
				(dx.rows() == ddim(t)),
				(x.rows() == dim(t)));
		eigen::add_to(x_out, x, dx);
	}

	auto dintegrate_d_base_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}

	void dintegrate_d_base(
			view<T, colmat> dx_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			dynamic_stack_view stack) const {
		(void)t, (void)x, (void)dx, (void)stack;

		VEG_DEBUG_ASSERT_ALL_OF(
				dx_out.rows() == ddim(t),
				dx_out.cols() == ddim(t),
				dx.rows() == ddim(t),
				x.rows() == dim(t));

		dx_out.setIdentity();
	}

	auto difference_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}

	void difference(
			view<T, colvec> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			dynamic_stack_view stack) const {

		(void)t, (void)stack;
		VEG_DEBUG_ASSERT_ALL_OF( //
				out.rows() == ddim(t),
				start.rows() == dim(t),
				finish.rows() == dim(t));

		eigen::sub_to(out, finish, start);
	}

	auto d_difference_d_finish_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}

	void d_difference_d_finish(
			view<T, colmat> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			dynamic_stack_view stack) const {

		(void)t, (void)start, (void)finish, (void)stack;
		VEG_DEBUG_ASSERT_ALL_OF(
				out.rows() == ddim(t),
				out.cols() == ddim(t),
				start.rows() == dim(t),
				finish.rows() == dim(t));

		out.setIdentity();
	}
};

struct constant_dim {
	i64 dim_;
	auto dim(i64 /*t*/) const -> i64 { return dim_; }
	auto max_dim() const -> i64 { return dim_; }
};

template <typename T>
using vector_space = vector_space_from_parent<T, constant_dim>;

// pinocchio expects a zero'd matrix
// TODO: document upstream
template <typename T>
struct pinocchio_state_space {
	pinocchio::model<T> const& model;
	using scalar = T;

	auto tangent() const { return vector_space<T>{{max_ddim()}}; }

	auto dim(i64 t) const -> i64 {
		return (void)t, model.config_dim() + model.tangent_dim();
	}
	auto ddim(i64 t) const -> i64 { return (void)t, 2 * model.tangent_dim(); }

	auto max_dim() const -> i64 {
		return model.config_dim() + model.tangent_dim();
	}
	auto max_ddim() const -> i64 { return 2 * model.tangent_dim(); }

	auto integrate_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	void integrate(
			view<T, colvec> x_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			dynamic_stack_view stack) const {

		(void)t, (void)stack;

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(x_out, x)), //
				(!eigen::aliases(x_out, dx)));

		VEG_DEBUG_ASSERT_ALL_OF( //
				(x_out.rows() == dim(t)),
				(x.rows() == dim(t)),
				(dx.rows() == ddim(t)));

		VEG_BIND(
				auto, (xw_q, xw_v), eigen::split_at_row(x_out, model.config_dim()));

		VEG_BIND(auto, (x_q, x_v), eigen::split_at_row(x, model.config_dim()));
		VEG_BIND(auto, (dx_q, dx_v), eigen::split_at_row(dx, model.tangent_dim()));

		model.integrate(xw_q, x_q, dx_q);
		eigen::add_to(xw_v, x_v, dx_v);
	}

	auto dintegrate_d_base_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	void dintegrate_d_base(
			view<T, colmat> dx_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			dynamic_stack_view stack) const {

		(void)stack, (void)t;

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(dx_out, dx)), //
				(!eigen::aliases(dx_out, dx)));

		VEG_DEBUG_ASSERT_ALL_OF(
				(dx_out.rows() == ddim()),
				(dx_out.cols() == ddim()),
				(x.rows() == dim()),
				(dx.rows() == ddim()));

		auto nq = model.config_dim();
		auto nv = model.tangent_dim();

		VEG_BIND(
				auto,
				(dxw_qq, dxw_qv, dxw_vq, dxw_vv),
				eigen::split_at(dx_out, nv, nv));
		(void)dxw_qv, (void)dxw_vq;

		VEG_BIND(auto, (x_q, x_v), eigen::split_at_row(x, nq));
		VEG_BIND(auto, (dx_q, dx_v), eigen::split_at_row(dx, nv));

		(void)x_v, (void)dx_v;

		dx_out.setZero();
		model.d_integrate_dq(dxw_qq, x_q, dx_q);
		eigen::add_identity(dxw_vv);
	}

	auto difference_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	void difference(
			view<T, colvec> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			dynamic_stack_view stack) const {

		(void)t, (void)stack;

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(out, start)), //
				(!eigen::aliases(out, finish)));

		VEG_DEBUG_ASSERT_ALL_OF(
				(out.rows() == ddim(t)),
				(start.rows() == dim(t)),
				(finish.rows() == dim(t)));

		auto nq = model.config_dim();
		auto nv = model.tangent_dim();

		VEG_BIND(auto, (out_q, out_v), eigen::split_at_row(out, nv));

		VEG_BIND(auto, (start_q, start_v), eigen::split_at_row(start, nq));
		VEG_BIND(auto, (finish_q, finish_v), eigen::split_at_row(finish, nq));

		eigen::sub_to(out_v, finish_v, start_v);
		model.difference(out_q, start_q, finish_q);
	}

	auto d_difference_d_finish_req() const -> mem_req {
		(void)this;
		return {tag<T>, 0};
	}
	void d_difference_d_finish(
			view<T, colmat> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			dynamic_stack_view stack) const {

		(void)t, (void)stack;

		VEG_DEBUG_ASSERT_ALL_OF(
				(!eigen::aliases(out, start)), //
				(!eigen::aliases(out, finish)));

		VEG_DEBUG_ASSERT_ALL_OF(
				(out.rows() == ddim(t)),
				(out.cols() == ddim(t)),
				(start.rows() == dim(t)),
				(finish.rows() == dim(t)));

		auto nq = model.config_dim();
		auto nv = model.tangent_dim();

		VEG_BIND(
				auto, (out_qq, out_qv, out_vq, out_vv), eigen::split_at(out, nv, nv));
		(void)out_qv, (void)out_vq;

		VEG_BIND(auto, (start_q, start_v), eigen::split_at_row(start, nq));
		VEG_BIND(auto, (finish_q, finish_v), eigen::split_at_row(finish, nq));
		(void)start_v, (void)finish_v;

		out.setZero();
		model.d_difference_dq_finish(out_qq, start_q, finish_q);
		eigen::add_identity(out_vv);
	}
};

} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_SPACE_HPP_3K0MQGEZS */
