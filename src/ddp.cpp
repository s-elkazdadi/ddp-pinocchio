#include <ddp/internal/eigen.hpp>

template auto ddp::eigen::internal::format_impl(
    fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
    veg::i64 rows,
    veg::i64 cols,
    veg::fn_ref<float(veg::i64, veg::i64)> getter,
    fmt::formatter<float>& fmt) -> decltype(fc.out());

template auto ddp::eigen::internal::format_impl(
    fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
    veg::i64 rows,
    veg::i64 cols,
    veg::fn_ref<double(veg::i64, veg::i64)> getter,
    fmt::formatter<double>& fmt) -> decltype(fc.out());

template auto ddp::eigen::internal::format_impl(
    fmt::basic_format_context<fmt::v7::detail::buffer_appender<char>, char>& fc,
    veg::i64 rows,
    veg::i64 cols,
    veg::fn_ref<long double(veg::i64, veg::i64)> getter,
    fmt::formatter<long double>& fmt) -> decltype(fc.out());
