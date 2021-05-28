#ifndef DDP_PINOCCHIO_SPACE_HPP_3K0MQGEZS
#define DDP_PINOCCHIO_SPACE_HPP_3K0MQGEZS

#include "ddp/pinocchio_model.hpp"
#include "ddp/internal/matrix_seq.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace ddp {
namespace internal {
namespace meta {

template <typename T>
using dim_expr = decltype(VEG_DECLVAL(T const&).dim(VEG_DECLVAL(i64)));
template <typename T>
using ddim_expr = decltype(VEG_DECLVAL(T const&).ddim(VEG_DECLVAL(i64)));
template <typename T>
using max_dim_expr = decltype(VEG_DECLVAL(T const&).max_dim());
template <typename T>
using max_ddim_expr = decltype(VEG_DECLVAL(T const&).max_ddim());

template <typename T>
using scalar_type_t = typename T::scalar;
template <typename T>
using integrate_expr = decltype(VEG_DECLVAL(T const&).integrate(

		VEG_DECLVAL(view<scalar_type_t<T>, colvec>),
		VEG_DECLVAL(i64),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(DynStackView)

				));
template <typename T>
using difference_expr = decltype(VEG_DECLVAL(T const&).difference(

		VEG_DECLVAL(view<scalar_type_t<T>, colvec>),
		VEG_DECLVAL(i64),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(DynStackView)

				));
template <typename T>
using dintegrate_expr = decltype(VEG_DECLVAL(T const&).dintegrate_d_base(

		VEG_DECLVAL(view<scalar_type_t<T>, colmat>),
		VEG_DECLVAL(i64),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(DynStackView)

				));
template <typename T>
using ddifference_expr = decltype(VEG_DECLVAL(T const&).d_difference_d_finish(

		VEG_DECLVAL(view<scalar_type_t<T>, colmat>),
		VEG_DECLVAL(i64),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(view<scalar_type_t<T> const, colvec>),
		VEG_DECLVAL(DynStackView)

				));

template <typename T>
using req_integrate_expr = decltype(VEG_DECLVAL(T const&).integrate_req());
template <typename T>
using req_dintegrate_expr =
		decltype(VEG_DECLVAL(T const&).dintegrate_d_base_req());
template <typename T>
using req_difference_expr = decltype(VEG_DECLVAL(T const&).difference_req());
template <typename T>
using req_ddifference_expr =
		decltype(VEG_DECLVAL(T const&).d_difference_d_finish_req());

} // namespace meta
} // namespace internal
namespace concepts {
namespace aux {
using namespace internal::meta;

DDP_DEF_CONCEPT(
		typename T,
		has_scalar,
		DDP_CONCEPT(scalar<veg::meta::detected_t<scalar_type_t, T>>));

DDP_DEF_CONCEPT(
		typename T,
		has_dim,
		VEG_CONCEPT(same<i64, veg::meta::detected_t<dim_expr, T>>));
DDP_DEF_CONCEPT(
		typename T,
		has_ddim,
		VEG_CONCEPT(same<i64, veg::meta::detected_t<ddim_expr, T>>));
DDP_DEF_CONCEPT(
		typename T,
		has_max_dim,
		VEG_CONCEPT(same<i64, veg::meta::detected_t<max_dim_expr, T>>));
DDP_DEF_CONCEPT(
		typename T,
		has_max_ddim,
		VEG_CONCEPT(same<i64, veg::meta::detected_t<max_ddim_expr, T>>));

DDP_DEF_CONCEPT(
		typename T, //
		has_integrate,
		VEG_CONCEPT(detected<integrate_expr, T>));
DDP_DEF_CONCEPT(
		typename T, //
		has_dintegrate,
		VEG_CONCEPT(detected<dintegrate_expr, T>));
DDP_DEF_CONCEPT(
		typename T, //
		has_difference,
		VEG_CONCEPT(detected<difference_expr, T>));
DDP_DEF_CONCEPT(
		typename T, //
		has_ddifference,
		VEG_CONCEPT(detected<ddifference_expr, T>));

DDP_DEF_CONCEPT(
		typename T,
		has_req_integrate,
		VEG_CONCEPT(same<mem_req, veg::meta::detected_t<req_integrate_expr, T>>));
DDP_DEF_CONCEPT(
		typename T,
		has_req_dintegrate,
		VEG_CONCEPT(same<mem_req, veg::meta::detected_t<req_dintegrate_expr, T>>));
DDP_DEF_CONCEPT(
		typename T,
		has_req_difference,
		VEG_CONCEPT(same<mem_req, veg::meta::detected_t<req_difference_expr, T>>));
DDP_DEF_CONCEPT(
		typename T,
		has_req_ddifference,
		VEG_CONCEPT(same<mem_req, veg::meta::detected_t<req_ddifference_expr, T>>));
} // namespace aux

DDP_DEF_CONCEPT_CONJUNCTION(
		typename T,
		space,

		((aux::, has_scalar<T>),
     (aux::, has_dim<T>),
     (aux::, has_ddim<T>),
     (aux::, has_max_dim<T>),
     (aux::, has_max_ddim<T>),

     (aux::, has_integrate<T>),
     (aux::, has_dintegrate<T>),
     (aux::, has_difference<T>),
     (aux::, has_ddifference<T>),

     (aux::, has_req_integrate<T>),
     (aux::, has_req_dintegrate<T>),
     (aux::, has_req_difference<T>),
     (aux::, has_req_ddifference<T>)

         ));
} // namespace concepts

template <typename Space>
auto space_to_idx(Space space, i64 begin, i64 end) -> idx::idx<colvec> {
	return {begin, end, [&](i64 t) { return idx::dims<colvec>{space.dim(t)}; }};
}

template <typename T, typename Dim_Fn>
struct basic_vector_space {
	DDP_CHECK_CONCEPT(scalar<T>);
	VEG_CHECK_CONCEPT(invocable_r<Dim_Fn, i64, i64>);

	struct raw_parts {
		Dim_Fn dim;
		i64 max_dim;
	} self;
	using scalar = T;

	auto dim(i64 t) const -> i64 { return self.dim(t); }
	auto ddim(i64 t) const -> i64 { return self.dim(t); }
	auto max_dim() const -> i64 { return self.max_dim; }
	auto max_ddim() const -> i64 { return self.max_dim; }

	auto integrate_req() const -> mem_req {
		unused(this);
		return {tag<T>, 0};
	}
	void integrate(
			view<T, colvec> x_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			DynStackView stack) const {
		unused(t, stack);
		VEG_DEBUG_ASSERT_ALL_OF( //
				(x_out.rows() == dim(t)),
				(dx.rows() == ddim(t)),
				(x.rows() == dim(t)));
		eigen::add_to(x_out, x, dx);
	}

	auto dintegrate_d_base_req() const -> mem_req {
		unused(this);
		return {tag<T>, 0};
	}

	void dintegrate_d_base(
			view<T, colmat> dx_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			DynStackView stack) const {
		unused(t, x, dx, stack);

		VEG_DEBUG_ASSERT_ALL_OF(
				dx_out.rows() == ddim(t),
				dx_out.cols() == ddim(t),
				dx.rows() == ddim(t),
				x.rows() == dim(t));

		dx_out.setIdentity();
	}

	auto difference_req() const -> mem_req {
		unused(this);
		return {tag<T>, 0};
	}

	void difference(
			view<T, colvec> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			DynStackView stack) const {

		unused(t, stack);
		VEG_DEBUG_ASSERT_ALL_OF( //
				out.rows() == ddim(t),
				start.rows() == dim(t),
				finish.rows() == dim(t));

		eigen::sub_to(out, finish, start);
	}

	auto d_difference_d_finish_req() const -> mem_req {
		unused(this);
		return {tag<T>, 0};
	}

	void d_difference_d_finish(
			view<T, colmat> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			DynStackView stack) const {

		unused(t, start, finish, stack);
		VEG_DEBUG_ASSERT_ALL_OF(
				out.rows() == ddim(t),
				out.cols() == ddim(t),
				start.rows() == dim(t),
				finish.rows() == dim(t));

		out.setIdentity();
	}
};

struct constant_dim {
	struct raw_parts {
		i64 dim;
	} self;
	auto operator()(i64 /*t*/) const noexcept -> i64 { return self.dim; }
};

template <typename T>
struct vector_space : basic_vector_space<T, constant_dim> {
	explicit vector_space(i64 dim) noexcept
			: basic_vector_space<T, constant_dim>{{{{dim}}, dim}} {}
};

// pinocchio expects a zero'd matrix
// TODO: document upstream
template <typename T>
struct pinocchio_state_space {
	DDP_CHECK_CONCEPT(scalar<T>);

	pinocchio::model<T> const& model;
	using scalar = T;

	auto dim(i64 t) const -> i64 {
		return unused(t), model.config_dim() + model.tangent_dim();
	}
	auto ddim(i64 t) const -> i64 { return unused(t), 2 * model.tangent_dim(); }

	auto max_dim() const -> i64 {
		return model.config_dim() + model.tangent_dim();
	}
	auto max_ddim() const -> i64 { return 2 * model.tangent_dim(); }

	auto integrate_req() const -> mem_req {
		unused(this);
		return {tag<T>, 0};
	}
	void integrate(
			view<T, colvec> x_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			DynStackView stack) const {

		unused(t, stack);

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
		unused(this);
		return {tag<T>, 0};
	}
	void dintegrate_d_base(
			view<T, colmat> dx_out,
			i64 t,
			view<T const, colvec> x,
			view<T const, colvec> dx,
			DynStackView stack) const {

		unused(t, stack);

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

		VEG_BIND(auto, (x_q, x_v), eigen::split_at_row(x, nq));
		VEG_BIND(auto, (dx_q, dx_v), eigen::split_at_row(dx, nv));

		unused(dxw_qv, dxw_vq, x_v, dx_v);

		dx_out.setZero();
		model.d_integrate_dq(dxw_qq, x_q, dx_q);
		eigen::add_identity(dxw_vv);
	}

	auto difference_req() const -> mem_req {
		unused(this);
		return {tag<T>, 0};
	}
	void difference(
			view<T, colvec> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			DynStackView stack) const {

		unused(t, stack);

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
		unused(this);
		return {tag<T>, 0};
	}
	void d_difference_d_finish(
			view<T, colmat> out,
			i64 t,
			view<T const, colvec> start,
			view<T const, colvec> finish,
			DynStackView stack) const {

		unused(t, stack);

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

		VEG_BIND(auto, (start_q, start_v), eigen::split_at_row(start, nq));
		VEG_BIND(auto, (finish_q, finish_v), eigen::split_at_row(finish, nq));
		unused(out_qv, out_vq, start_v, finish_v);

		out.setZero();
		model.d_difference_dq_finish(out_qq, start_q, finish_q);
		eigen::add_identity(out_vv);
	}
};

} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_SPACE_HPP_3K0MQGEZS */
