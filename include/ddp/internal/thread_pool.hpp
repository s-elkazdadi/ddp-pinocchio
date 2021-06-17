#ifndef DDP_PINOCCHIO_THREAD_POOL_HPP_KXVNNCOAS
#define DDP_PINOCCHIO_THREAD_POOL_HPP_KXVNNCOAS

#include "ddp/internal/utils.hpp"
#include <veg/fn_view.hpp>

namespace ddp {
namespace internal {

void set_num_threads(i64 n_threads);
auto get_num_procs() noexcept -> i64;

void parallel_for(
		fn::FnView<void(i64 thread_id, i64 begin_index, i64 end_index)> fn,
		i64 begin,
		i64 end);

} // namespace internal
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_THREAD_POOL_HPP_KXVNNCOAS */
