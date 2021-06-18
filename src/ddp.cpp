#include <ddp/internal/eigen.hpp>
#include <omp.h>

template auto ddp::eigen::internal::format_impl(
		fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
		veg::i64 rows,
		veg::i64 cols,
		veg::fn::FnView<float(veg::i64, veg::i64)> getter,
		fmt::formatter<float>& fmt) -> decltype(fc.out());

template auto ddp::eigen::internal::format_impl(
		fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
		veg::i64 rows,
		veg::i64 cols,
		veg::fn::FnView<double(veg::i64, veg::i64)> getter,
		fmt::formatter<double>& fmt) -> decltype(fc.out());

template auto ddp::eigen::internal::format_impl(
		fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
		veg::i64 rows,
		veg::i64 cols,
		veg::fn::FnView<long double(veg::i64, veg::i64)> getter,
		fmt::formatter<long double>& fmt) -> decltype(fc.out());

namespace ddp {
namespace internal {

auto get_num_procs() noexcept -> i64 {
	return omp_get_num_procs();
}

void set_num_threads(i64 n_threads) {
	omp_set_num_threads(narrow<int>(n_threads));
}

void parallel_for(
		FnView<auto(i64 id)->void*> setup,
		FnView<void(void* state, i64 id, i64 i)> fn,
		i64 begin,
		i64 end) {
	auto const len = narrow<usize>(end - begin);

// #pragma omp parallel 
	{
		i64 const thread_num = omp_get_thread_num();
		void* state = setup(thread_num);

// #pragma omp for
		for (usize i = 0; i < len; ++i) {
      fn(state, thread_num, begin + i64(i));
		}
	}
}

} // namespace internal
} // namespace ddp
