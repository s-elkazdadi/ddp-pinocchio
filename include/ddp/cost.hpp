#ifndef DDP_PINOCCHIO_COST_HPP_FVE04HDWS
#define DDP_PINOCCHIO_COST_HPP_FVE04HDWS

#include "ddp/internal/eigen.hpp"
#include "veg/internal/prologue.hpp"

namespace ddp {
namespace cost {
template <typename T>
struct HomogeneousQuadratic {
	eigen::HeapMatrix<T, colvec> q;
	eigen::HeapMatrix<T, colmat> Q;
	eigen::HeapMatrix<T, colvec> r;
	eigen::HeapMatrix<T, colmat> R;
	eigen::HeapMatrix<T, colvec> rf;
	eigen::HeapMatrix<T, colmat> Rf;

	struct Ref {
		eigen::View<T const, colvec> q;
		eigen::View<T const, colmat> Q;
		eigen::View<T const, colvec> r;
		eigen::View<T const, colmat> R;
		eigen::View<T const, colvec> rf;
		eigen::View<T const, colmat> Rf;

		auto eval_req() const -> MemReq { return {tag<T>, 0}; }
		auto d_eval_to_req() const -> MemReq {
			return MemReq::max_of({
					as_ref,
					{
							{tag<T>, q.rows()},
							{tag<T>, r.rows()},
							{tag<T>, rf.rows()},
					},
			});
		}
		auto dd_eval_to_req() const -> MemReq { return {tag<T>, 0}; }

		template <typename Key>
		auto eval_final(View<T const, colvec> x, Key k, DynStackView stack) const
				-> Tuple<Key, T> {
			auto nx = rf.rows();
			VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx);
			DDP_TMP_VECTOR(stack, tmp, T, nx);
			eigen::mul_add_to_noalias(tmp, Rf, x);
			return {direct, VEG_FWD(k), 0.5 * eigen::dot(tmp, x) + eigen::dot(rf, x)};
		}

		template <typename Key>
		auto d_eval_final_to(
				View<T, colvec> out_x,
				View<T const, colvec> x,
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
				View<T, colmat> out_xx,
				View<T const, colvec> x,
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
				View<T const, colvec> x,
				View<T const, colvec> u,
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
				View<T, colvec> out_x,
				View<T, colvec> out_u,
				i64 t,
				View<T const, colvec> x,
				View<T const, colvec> u,
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
				View<T, colmat> out_xx,
				View<T, colmat> out_ux,
				View<T, colmat> out_uu,
				i64 t,
				View<T const, colvec> x,
				View<T const, colvec> u,
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
	auto ref(Dynamics const& /*unused*/) noexcept -> Ref {
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

namespace nb {
struct homogeneous_quadratic {
	template <typename T>
	auto operator()(
			eigen::HeapMatrix<T, colvec> q,
			eigen::HeapMatrix<T, colmat> Q,
			eigen::HeapMatrix<T, colvec> r,
			eigen::HeapMatrix<T, colmat> R,
			eigen::HeapMatrix<T, colvec> rf,
			eigen::HeapMatrix<T, colmat> Rf) const -> HomogeneousQuadratic<T> {
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
} // namespace nb
VEG_NIEBLOID(homogeneous_quadratic);
} // namespace cost

} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_COST_HPP_FVE04HDWS */
