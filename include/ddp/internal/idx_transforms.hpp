#ifndef DDP_PINOCCHIO_IDX_TRANSFORMS_HPP_NS2OY3QES
#define DDP_PINOCCHIO_IDX_TRANSFORMS_HPP_NS2OY3QES

#include "ddp/internal/matrix_seq.hpp"

namespace ddp {
namespace idx {

template <eigen::kind K>
auto shift_time_idx(idx<K> i, i64 dt) -> idx<K> {
  VEG_BIND(auto, (base, self), VEG_FWD(i).into_parts());
  return {
      base.begin - dt,
      base.end - dt,
      [&](i64 t) { return self.dim_data[veg::narrow<usize>(t - base.begin)]; },
  };
}

inline auto prod_idx(idx_view<colvec> l, idx_view<colvec> r) -> idx<colmat> {
  return {
      l.index_begin(),
      l.index_end(),
      [&](i64 t) -> dims<colmat> {
        return {l.rows(t), r.rows(t)};
      },
  };
}

} // namespace idx
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_IDX_TRANSFORMS_HPP_NS2OY3QES */
