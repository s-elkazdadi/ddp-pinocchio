#ifndef ASSERTIONS_HPP_TKWV84LQ
#define ASSERTIONS_HPP_TKWV84LQ

#include <string>
#include <fmt/format.h>
#include ".hedley.h"

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/tuple/pop_front.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/seq/seq.hpp>

namespace ddp {
namespace assertion {

auto inline all_of(std::initializer_list<bool> arr) -> bool {
  for (bool b : arr) {
    if (!b) {
      return false;
    }
  }
  return true;
}

[[noreturn]] void on_assert_fail(long line, char const* file, char const* func);
void on_expect_fail(long line, char const* file, char const* func);

void set_assert_params(          //
    fmt::string_view expression, //
    fmt::string_view message,    //
    fmt::string_view op,         //
    std::string lhs,             //
    std::string rhs              //
);

template <typename T>
struct lhs_all_of_t {
  T const& lhs;
  fmt::string_view expr;
  fmt::string_view msg;

#define DDP_ASSERT_COMPARISON_OP(Op)                                                                                   \
  template <typename U>                                                                                                \
  auto operator Op(U const& rhs)->bool {                                                                               \
    bool res = static_cast<bool>(lhs Op rhs);                                                                          \
    if (not res) {                                                                                                     \
      assertion::set_assert_params(expr, msg, " " #Op " ", fmt::format("{}", lhs), fmt::format("{}", rhs));            \
    }                                                                                                                  \
    return res;                                                                                                        \
  }                                                                                                                    \
  static_assert(true, "")

  DDP_ASSERT_COMPARISON_OP(==);
  DDP_ASSERT_COMPARISON_OP(!=);
  DDP_ASSERT_COMPARISON_OP(<);
  DDP_ASSERT_COMPARISON_OP(>);
  DDP_ASSERT_COMPARISON_OP(<=);
  DDP_ASSERT_COMPARISON_OP(>=);

#undef DDP_ASSERT_COMPARISON_OP

  operator/* NOLINT(hicpp-explicit-conversions) */ bool() {
    bool res = static_cast<bool>(lhs);
    if (not res) {
      assertion::set_assert_params(expr, msg, "", fmt::format("{}", lhs), "");
    }
    return res;
  }
};

struct operand_strings_t {
  std::string l;
  std::string r;
  fmt::string_view op;
};

struct lhs_rhs_any_of_t {
  void const* lhs = {};
  void const* rhs = {};
  auto (*comp)(void const* l, void const* r) noexcept -> bool = {};
  auto (*serializer)(void const* l, void const* r) -> operand_strings_t = {};
  fmt::string_view expr;
  fmt::string_view msg;
};

auto inline any_of(std::initializer_list<lhs_rhs_any_of_t> arr) -> bool {
  for (auto const& x : arr) {
    if (x.comp(x.lhs, x.rhs)) {
      return true;
    }
  }

  for (auto const& x : arr) {
    auto operands = x.serializer(x.lhs, x.rhs);
    assertion::set_assert_params(
        x.expr,
        x.msg,
        operands.op,
        static_cast<std::string&&>(operands.l),
        static_cast<std::string&&>(operands.r));
  }

  return false;
}

template <typename T>
struct lhs_any_of_t {

  T const& lhs;
  fmt::string_view expr;
  fmt::string_view msg;

#define DDP_ASSERT_COMPARISON_OP(Op)                                                                                   \
  template <typename U>                                                                                                \
  auto operator Op(U const& rhs)->lhs_rhs_any_of_t {                                                                   \
    return {                                                                                                           \
        static_cast<void const*>(&lhs),                                                                                \
        static_cast<void const*>(&rhs),                                                                                \
        +[](void const* l, void const* r) noexcept -> bool {                                                           \
          return static_cast<bool>((*static_cast<T const*>(l))Op(*static_cast<U const*>(r)));                          \
        },                                                                                                             \
        +[](void const* l, void const* r) -> operand_strings_t {                                                       \
          return {                                                                                                     \
              fmt::format("{}", *static_cast<T const*>(l)),                                                            \
              fmt::format("{}", *static_cast<U const*>(r)),                                                            \
              " " #Op " ",                                                                                             \
          };                                                                                                           \
        },                                                                                                             \
        expr,                                                                                                          \
        msg,                                                                                                           \
    };                                                                                                                 \
  }                                                                                                                    \
  static_assert(true, "")

  DDP_ASSERT_COMPARISON_OP(==);
  DDP_ASSERT_COMPARISON_OP(!=);
  DDP_ASSERT_COMPARISON_OP(<);
  DDP_ASSERT_COMPARISON_OP(>);
  DDP_ASSERT_COMPARISON_OP(<=);
  DDP_ASSERT_COMPARISON_OP(>=);

#undef DDP_ASSERT_COMPARISON_OP

  operator/* NOLINT(hicpp-explicit-conversions) */ lhs_rhs_any_of_t() {
    return {
        static_cast<void const*>(&lhs),
        nullptr,
        +[](void const* l, void const* r) noexcept -> bool {
          (void)r;
          return static_cast<bool>(*static_cast<T const*>(l));
        },
        +[](void const* l, void const* r) -> operand_strings_t {
          (void)r;
          return {
              fmt::format("{}", *static_cast<T const*>(l)),
              "",
              "",
          };
        },
        expr,
        msg,
    };
  }
};

struct expression_decomposer_any_of_t {
  fmt::string_view expr;
  fmt::string_view msg;

  template <typename T>
  auto operator<<(T const& lhs) -> lhs_any_of_t<T> {
    return {lhs, expr, msg};
  }
};

struct expression_decomposer_all_of_t {
  fmt::string_view expr;
  fmt::string_view msg;

  template <typename T>
  auto operator<<(T const& lhs) -> lhs_all_of_t<T> {
    return {lhs, expr, msg};
  }
};

} // namespace assertion
} // namespace ddp

#if defined(HEDLEY_GNUC_VERSION)
#define DDP_THIS_FUNCTION __PRETTY_FUNCTION__
#elif defined(HEDLEY_MSVC_VERSION)
#define DDP_THIS_FUNCTION __FUNCSIG__
#else
#define DDP_THIS_FUNCTION __func__
#endif

#define DDP_ASSERT_IMPL(Callback, Message, ...)                                                                        \
  (static_cast<bool>(                                                                                                  \
       ::ddp::assertion::expression_decomposer_all_of_t{                                                               \
           #__VA_ARGS__,                                                                                               \
           Message,                                                                                                    \
       }                                                                                                               \
       << __VA_ARGS__)                                                                                                 \
       ? (void)(0)                                                                                                     \
       : ::ddp::assertion::Callback(                                                                                   \
             __LINE__,                                                                                                 \
             static_cast<char const*>(__FILE__),                                                                       \
             static_cast<char const*>(DDP_THIS_FUNCTION)))

#define DDP_ASSERT_MSG_AGGREGATE_IMPL1(Decomposer, Aggregator, Callback, ...)                                          \
  DDP_ASSERT_MSG_AGGREGATE_IMPL2(Decomposer, Aggregator, Callback, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define DDP_ASSERT_MSG_AGGREGATE_IMPL_FTOR(_, Decomposer, Elem)                                                        \
  (::ddp::assertion::Decomposer{                                                                                       \
       BOOST_PP_STRINGIZE(BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_POP_FRONT(Elem))),                                     \
       BOOST_PP_TUPLE_ELEM(0, Elem),                                                                                   \
   }                                                                                                                   \
   << BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_POP_FRONT(Elem))),

#define DDP_ASSERT_MSG_AGGREGATE_IMPL2(Decomposer, Aggregator, Callback, Seq)                                          \
  (::ddp::assertion::Aggregator({BOOST_PP_SEQ_FOR_EACH(DDP_ASSERT_MSG_AGGREGATE_IMPL_FTOR, Decomposer, Seq)})          \
       ? (void)(0)                                                                                                     \
       : ::ddp::assertion::Callback(                                                                                   \
             __LINE__,                                                                                                 \
             static_cast<char const*>(__FILE__),                                                                       \
             static_cast<char const*>(DDP_THIS_FUNCTION)))

#if HEDLEY_HAS_WARNING("-Woverloaded-shift-op-parentheses")
#define DDP_IGNORE_SHIFT_PAREN_WARNING(Macro, ...)                                                                     \
  HEDLEY_DIAGNOSTIC_PUSH _Pragma("clang diagnostic ignored \"-Woverloaded-shift-op-parentheses\" ") Macro(__VA_ARGS__) \
      HEDLEY_DIAGNOSTIC_POP
#else
#define DDP_IGNORE_SHIFT_PAREN_WARNING(Macro, ...) Macro(__VA_ARGS__)
#endif

#define DDP_ASSERT_MSG(...) DDP_IGNORE_SHIFT_PAREN_WARNING(DDP_ASSERT_IMPL, on_assert_fail, __VA_ARGS__)
#define DDP_ASSERT_MSG_ALL_OF(...)                                                                                     \
  DDP_IGNORE_SHIFT_PAREN_WARNING(                                                                                      \
      DDP_ASSERT_MSG_AGGREGATE_IMPL1,                                                                                  \
      expression_decomposer_all_of_t,                                                                                  \
      all_of,                                                                                                          \
      on_assert_fail,                                                                                                  \
      __VA_ARGS__)
#define DDP_ASSERT_MSG_ANY_OF(...)                                                                                     \
  DDP_IGNORE_SHIFT_PAREN_WARNING(                                                                                      \
      DDP_ASSERT_MSG_AGGREGATE_IMPL1,                                                                                  \
      expression_decomposer_any_of_t,                                                                                  \
      any_of,                                                                                                          \
      on_assert_fail,                                                                                                  \
      __VA_ARGS__)

#define DDP_EXPECT_MSG(...) DDP_IGNORE_SHIFT_PAREN_WARNING(DDP_ASSERT_IMPL, on_expect_fail, __VA_ARGS__)
#define DDP_EXPECT_MSG_ALL_OF(...)                                                                                     \
  DDP_IGNORE_SHIFT_PAREN_WARNING(                                                                                      \
      DDP_ASSERT_MSG_AGGREGATE_IMPL1,                                                                                  \
      expression_decomposer_all_of_t,                                                                                  \
      all_of,                                                                                                          \
      on_expect_fail,                                                                                                  \
      __VA_ARGS__)
#define DDP_EXPECT_MSG_ANY_OF(...)                                                                                     \
  DDP_IGNORE_SHIFT_PAREN_WARNING(                                                                                      \
      DDP_ASSERT_MSG_AGGREGATE_IMPL1,                                                                                  \
      expression_decomposer_any_of_t,                                                                                  \
      any_of,                                                                                                          \
      on_expect_fail,                                                                                                  \
      __VA_ARGS__)

#define DDP_ASSERT(...) DDP_ASSERT_MSG("", __VA_ARGS__)
#define DDP_EXPECT(...) DDP_EXPECT_MSG("", __VA_ARGS__)

#ifdef NDEBUG
#define DDP_DEBUG_ASSERT_MSG(Message, ...) ((void)0)
#define DDP_DEBUG_ASSERT_MSG_ALL_OF(...) ((void)0)
#define DDP_DEBUG_ASSERT_MSG_ANY_OF(...) ((void)0)
#define DDP_DEBUG_EXPECT_MSG(Message, ...) ((void)0)
#define DDP_DEBUG_EXPECT_MSG_ALL_OF(...) ((void)0)
#define DDP_DEBUG_EXPECT_MSG_ANY_OF(...) ((void)0)
#else
#define DDP_DEBUG_ASSERT_MSG(Message, ...) DDP_ASSERT_MSG(Message, __VA_ARGS__)
#define DDP_DEBUG_ASSERT_MSG_ALL_OF(...) DDP_ASSERT_MSG_ALL_OF(__VA_ARGS__)
#define DDP_DEBUG_ASSERT_MSG_ANY_OF(...) DDP_ASSERT_MSG_ANY_OF(__VA_ARGS__)
#define DDP_DEBUG_EXPECT_MSG(Message, ...) DDP_EXPECT_MSG(Message, __VA_ARGS__)
#define DDP_DEBUG_EXPECT_MSG_ALL_OF(...) DDP_EXPECT_MSG_ALL_OF(__VA_ARGS__)
#define DDP_DEBUG_EXPECT_MSG_ANY_OF(...) DDP_EXPECT_MSG_ANY_OF(__VA_ARGS__)
#endif

#endif /* end of include guard ASSERTIONS_HPP_TKWV84LQ */
