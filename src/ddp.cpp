#include <ddp/internal/eigen.hpp>
#include <deque>
#include <thread>
#include <condition_variable>
#include <mutex>

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

using ThreadFn =
		fn::FnView<void(i64 thread_id, i64 begin_index, i64 end_index)>;

struct Consumer {
	std::condition_variable condvar;
	std::mutex mtx;
	Option<ThreadFn> current;

	void operator()() {
    // condvar.wait(mtx);
	}
};

struct WorkThread {
	std::thread thrd;
};

std::deque<std::thread> thread_pool{};

void set_num_threads(i64 n_threads) {
	thread_pool.resize(n_threads);
}
auto get_num_procs() noexcept -> i64;

void parallel_for(
		fn::FnView<void(i64 thread_id, i64 begin_index, i64 end_index)> fn,
		i64 begin,
		i64 end);

} // namespace internal
} // namespace ddp
