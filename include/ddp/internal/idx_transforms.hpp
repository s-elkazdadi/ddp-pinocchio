#ifndef DDP_PINOCCHIO_IDX_TRANSFORMS_HPP_NS2OY3QES
#define DDP_PINOCCHIO_IDX_TRANSFORMS_HPP_NS2OY3QES

#include "ddp/internal/matrix_seq.hpp"

namespace ddp {
namespace idx {

template <eigen::Kind K>
auto shift_time_idx(Idx<K> i, i64 dt) -> Idx<K> {
	VEG_BIND(auto, (base, self), VEG_FWD(i).into_parts());
	return {
			base.begin - dt,
			base.end - dt,
			[&](i64 t) { return self.dim_data[narrow<usize>(t - base.begin)]; },
	};
}

inline auto prod_idx(IdxView<colvec> l, IdxView<colvec> r) -> Idx<colmat> {
	return {
			l.index_begin(),
			l.index_end(),
			[&](i64 t) -> Dims<colmat> {
				return {l.rows(t), r.rows(t)};
			},
	};
}

} // namespace idx
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_IDX_TRANSFORMS_HPP_NS2OY3QES */
