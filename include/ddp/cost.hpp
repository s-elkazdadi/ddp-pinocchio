#ifndef DDP_PINOCCHIO_COST_HPP_FVE04HDWS
#define DDP_PINOCCHIO_COST_HPP_FVE04HDWS

#include "ddp/internal/eigen.hpp"
#include "veg/internal/prologue.hpp"

namespace ddp {

template <typename T>
struct quadratic_cost_fixed_size {
	eigen::heap_matrix<T, colvec> q;
	eigen::heap_matrix<T, colmat> Q;
	eigen::heap_matrix<T, colvec> r;
	eigen::heap_matrix<T, colmat> R;
	eigen::heap_matrix<T, colvec> rf;
	eigen::heap_matrix<T, colmat> Rf;

	struct ref_type {
		eigen::view<T const, colvec> q;
		eigen::view<T const, colmat> Q;
		eigen::view<T const, colvec> r;
		eigen::view<T const, colmat> R;
		eigen::view<T const, colvec> rf;
		eigen::view<T const, colmat> Rf;

		auto eval_req() const -> mem_req { return {tag<T>, 0}; }
		auto d_eval_to_req() const -> mem_req {
			return mem_req::max_of({
					{tag<T>, q.rows()},
					{tag<T>, r.rows()},
					{tag<T>, rf.rows()},
			});
		}
		auto dd_eval_to_req() const -> mem_req { return {tag<T>, 0}; }

		template <typename Key>
		auto eval_final(view<T const, colvec> x, Key k, DynStackView stack) const
				-> Tuple<Key, T> {
			auto nx = rf.rows();
			VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx);
			DDP_TMP_VECTOR(stack, tmp, T, nx);
			eigen::mul_add_to_noalias(tmp, Rf, x);
			return {direct, VEG_FWD(k), 0.5 * eigen::dot(tmp, x) + eigen::dot(rf, x)};
		}

		template <typename Key>
		auto d_eval_final_to(
				view<T, colvec> out_x,
				view<T const, colvec> x,
				Key k,
				DynStackView stack) const -> Key {
			auto nx = r.rows();
			unused(nx, stack);

			VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx);

			eigen::assign(out_x, rf);
			eigen::mul_add_to_noalias(out_x, Rf, x);
			return k;
		}

		template <typename Key>
		auto dd_eval_final_to(
				view<T, colmat> out_xx,
				view<T const, colvec> x,
				Key k,
				DynStackView stack) const -> Key {
			auto nx = rf.rows();
			unused(stack, nx, x);

			VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx);
			eigen::assign(out_xx, Rf);
			return k;
		}

		template <typename Key>
		auto eval(
				i64 t,
				view<T const, colvec> x,
				view<T const, colvec> u,
				Key k,
				DynStackView stack) const -> Tuple<Key, T> {

			auto nu = q.rows();
			auto nx = r.rows();
			unused(t, nu, nx);

			VEG_DEBUG_ASSERT_ALL_OF( //
					x.rows() == nx,
					u.rows() == nu);

			T out(int(0));
			{
				DDP_TMP_VECTOR(stack, tmp, T, nx);
				eigen::mul_add_to_noalias(tmp, R, x);

				out += eigen::dot(tmp, x) / 2;
				out += eigen::dot(r, x);
			}
			{
				DDP_TMP_VECTOR(stack, tmp, T, nu);
				eigen::mul_add_to_noalias(tmp, Q, u);

				out += eigen::dot(eigen::as_const(tmp), u) / 2;
				out += eigen::dot(q, u);
			}
			return {direct, VEG_FWD(k), out};
		}

		template <typename Key>
		auto d_eval_to(
				view<T, colvec> out_x,
				view<T, colvec> out_u,
				i64 t,
				view<T const, colvec> x,
				view<T const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

			auto nu = q.rows();
			auto nx = r.rows();
			unused(stack, t, nu, nx);

			VEG_DEBUG_ASSERT_ALL_OF( //
					x.rows() == nx,
					u.rows() == nu,
					out_u.rows() == nu,
					out_x.rows() == nx);

			eigen::assign(out_x, r);
			eigen::assign(out_u, q);
			eigen::mul_add_to_noalias(out_x, R, x);
			eigen::mul_add_to_noalias(out_u, Q, u);
			return k;
		}

		template <typename Key>
		auto dd_eval_to(
				view<T, colmat> out_xx,
				view<T, colmat> out_ux,
				view<T, colmat> out_uu,
				i64 t,
				view<T const, colvec> x,
				view<T const, colvec> u,
				Key k,
				DynStackView stack) const -> Key {

			auto nu = q.rows();
			auto nx = r.rows();

			unused(t, x, u, stack, nu, nx);

			VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx, u.rows() == nu);

			eigen::assign(out_xx, R);
			eigen::assign(out_uu, Q);
			out_ux.setZero();
			return k;
		}
	};

	template <typename Dynamics>
	auto ref(Dynamics const& /*unused*/) noexcept -> ref_type {
		return {
				q.get(),
				Q.get(),
				r.get(),
				R.get(),
				rf.get(),
				Rf.get(),
		};
	}
};

namespace make {
namespace fn {
struct quadratic_cost_fixed_size_fn {
	template <typename T>
	auto operator()(
			eigen::heap_matrix<T, colvec> q,
			eigen::heap_matrix<T, colmat> Q,
			eigen::heap_matrix<T, colvec> r,
			eigen::heap_matrix<T, colmat> R,
			eigen::heap_matrix<T, colvec> rf,
			eigen::heap_matrix<T, colmat> Rf) const -> quadratic_cost_fixed_size<T> {
		auto nq = q.get().rows();
		auto nr = r.get().rows();
		unused(nq, nr);
		VEG_DEBUG_ASSERT_ALL_OF(
				Q.get().rows() == nq,
				Q.get().cols() == nq,
				R.get().rows() == nr,
				R.get().cols() == nr,
				rf.get().rows() == nr,
				Rf.get().rows() == nr);

		return {
				VEG_FWD(q),
				VEG_FWD(Q),
				VEG_FWD(r),
				VEG_FWD(R),
				VEG_FWD(rf),
				VEG_FWD(Rf),
		};
	}
};
} // namespace fn
VEG_INLINE_VAR(quadratic_cost_fixed_size, fn::quadratic_cost_fixed_size_fn);
} // namespace make

} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_COST_HPP_FVE04HDWS */
