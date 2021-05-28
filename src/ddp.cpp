#include <ddp/internal/eigen.hpp>

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
