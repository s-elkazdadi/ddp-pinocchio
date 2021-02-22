#ifndef FMT_HPP_8STTEZY1
#define FMT_HPP_8STTEZY1

#include "mpfr/detail/handle_as_mpfr.hpp"
#include "mpfr/detail/prologue.hpp"

namespace fmt {
inline namespace v7 {
template <typename OutputIt, typename Char> class basic_format_context;
template <typename T, typename Char, typename Enable> struct formatter;
} // namespace v7
} // namespace fmt

namespace mpfr {
namespace _ {
namespace libfmt {

template <int, typename T> struct dependent_type { using type = T; };

struct fill_t {
  char data[4];
  unsigned char size;
  friend constexpr auto operator==(fill_t f, char c) -> bool {
    return f.size == 1 and f.data[0] == c;
  }
};

enum struct align_e : unsigned char { left, right, center };
enum struct sign_e : unsigned char { plus, minus, space };
enum struct float_format_e : unsigned char { general, exp, fixed, hex, binary };

struct error_handler_ref_t {
private:
  void* m_eh;
  void (*m_on_error)(void* ctx, char const* msg);

  template <typename Error_Handler> static void on_error_impl(void* ctx, char const* msg) {
    static_cast<Error_Handler*>(ctx)->on_error(msg);
  }

public:
  template <typename Error_Handler>
  constexpr error_handler_ref_t(Error_Handler* ctx) // NOLINT(hicpp-explicit-conversions)
      : m_eh{ctx}, m_on_error{&on_error_impl<Error_Handler>} {}

  [[noreturn]] void on_error(char const* msg) {
    m_on_error(m_eh, msg);
    std::terminate();
  }
};

template <typename Iter>
MPFR_CXX_CONSTEXPR auto read_next_char(Iter& it, Iter end, error_handler_ref_t eh) -> fill_t {
  static_assert(
      std::is_same<
          typename std::remove_const<typename std::remove_reference<decltype(*it)>::type>::type,
          char>::value,
      "only narrow char is supported");
  static_assert(CHAR_BIT == 8, "char is assumed to be 8 bits wide");

  int leading_ones = count_leading_zeros(static_cast<unsigned char>(~unsigned(it[0])));
  fill_t ret{};
  if (leading_ones == 0) {
    ret.data[0] = *it;
    ret.size = 1;
    ++it;
    return ret;
  }

  ret.size = static_cast<unsigned char>(leading_ones);
  for (int i = 0; i < leading_ones; ++i) {
    if (it == end) {
      eh.on_error("invalid fill character");
    }
    ret.data[i] = *it;
    ++it;
  }
  return ret;
}

constexpr auto is_digit(char c) -> bool {
  return static_cast<unsigned char>(c) >= '0' and static_cast<unsigned char>(c) <= '9';
}

template <typename Iter>
MPFR_CXX_CONSTEXPR auto parse_int(Iter& it, Iter end, error_handler_ref_t eh) -> int {
  int val = *it - '0';
  ++it;
  while (it < end) {
    if (!is_digit(*it)) {
      break;
    };
    int digit = *it - '0';
    val = int(10 * unsigned(val) + unsigned(digit));
    if (val < 0) {
      eh.on_error("number is too big");
    }
    ++it;
  }
  return val;
};

struct mp_float_specs {
  fill_t fill = {{' ', {}, {}, {}}, 1};
  align_e align = align_e::right;
  sign_e sign = sign_e::minus;
  bool alt = false;
  bool pad_zero = false;

  bool dyn_width = false;
  int width_or_id = 1;
  bool dyn_prec = false;
  int prec_or_id = -1;

  float_format_e format = float_format_e::general;
  bool upper = false;
};

template <typename Parse_Context>
MPFR_CXX_CONSTEXPR auto
parse_width_ref(mp_float_specs& specs, typename Parse_Context::iterator it, Parse_Context& ctx) ->
    typename Parse_Context::iterator {
  auto&& _eh = ctx.error_handler();
  auto eh = error_handler_ref_t{&_eh};

  ++it;
  if (it == ctx.end()) {
    eh.on_error("could not parse width");
  }
  if (*it == '}') {
    specs.dyn_width = true;
    specs.width_or_id = ctx.next_arg_id();
  } else {
    if (not is_digit(*it)) {
      eh.on_error("precision arg-id must be a number");
    }
    specs.dyn_width = true;
    specs.width_or_id = parse_int(it, ctx.end(), eh);
    ctx.check_arg_id(specs.width_or_id);
    if (*it != '}') {
      eh.on_error("width arg-id must be followed by a matching '}'");
    }
  }
  ++it;
  return it;
}

template <typename Parse_Context>
MPFR_CXX_CONSTEXPR auto
parse_prec_ref(mp_float_specs& specs, typename Parse_Context::iterator it, Parse_Context& ctx) //
    -> typename Parse_Context::iterator {
  auto&& _eh = ctx.error_handler();
  auto eh = error_handler_ref_t{&_eh};

  ++it;
  if (it == ctx.end()) {
    eh.on_error("could not parse precision");
  }
  if (*it == '}') {
    specs.dyn_prec = true;
    specs.prec_or_id = ctx.next_arg_id();
  } else {
    if (not is_digit(*it)) {
      eh.on_error("precision arg-id must be a number");
    }
    specs.dyn_prec = true;
    specs.prec_or_id = parse_int(it, ctx.end(), eh);
    ctx.check_arg_id(specs.prec_or_id);
    if (it == ctx.end() or *it != '}') {
      eh.on_error("precision arg-id must be followed by a matching '}'");
    }
  }
  ++it;
  return it;
}

template <typename Parse_Context>
MPFR_CXX_CONSTEXPR auto
parse_width_to_end(mp_float_specs& specs, typename Parse_Context::iterator it, Parse_Context& ctx)
    -> typename Parse_Context::iterator {
  auto&& _eh = ctx.error_handler();
  auto eh = error_handler_ref_t{&_eh};

  if (it == ctx.end()) {
    return it;
  }

  if (is_digit(*it)) {
    specs.width_or_id = parse_int(it, ctx.end(), eh);
    if (specs.width_or_id == 0) {
      eh.on_error("width must be a positive number");
    }
  } else if (*it == '{') {
    it = parse_width_ref(specs, it, ctx);
  }

  if (it == ctx.end()) {
    return it;
  }

  if (*it == '.') {
    // parsing precision
    ++it;
    if (is_digit(*it)) {
      specs.prec_or_id = parse_int(it, ctx.end(), eh);
    } else if (*it == '{') {
      it = parse_prec_ref(specs, it, ctx);
    } else {
      eh.on_error("precision must be a dot '.' followed by a non-negative number");
    }
  }

  if (it == ctx.end()) {
    return it;
  }

  if (*it == 'L') {
    ++it;
  }
  if (it == ctx.end()) {
    return it;
  }

  if (it == ctx.end() or *it == '}') {
    return it;
  } else {
    switch (*it) {
    case 'G':
      specs.upper = true;
      HEDLEY_FALL_THROUGH;
    case 'g':
      specs.format = float_format_e::general;
      break;
    case 'E':
      specs.upper = true;
      HEDLEY_FALL_THROUGH;
    case 'e':
      specs.format = float_format_e::exp;
      break;
    case 'F':
      specs.upper = true;
      HEDLEY_FALL_THROUGH;
    case 'f':
      specs.format = float_format_e::fixed;
      break;
    case 'A':
      specs.upper = true;
      HEDLEY_FALL_THROUGH;
    case 'a':
      specs.format = float_format_e::hex;
      break;
    case 'b':
      specs.format = float_format_e::binary;
      break;
    default:
      eh.on_error("invalid type specifier for mp_float_t");
      break;
    }
    ++it;

    if (it == ctx.end() or *it == '}') {
      return it;
    } else {
      eh.on_error("invalid characters after type specifier");
    }
  }
}

template <typename Parse_Context>
MPFR_CXX_CONSTEXPR auto parse_mp_float_type_specs(mp_float_specs& specs, Parse_Context& ctx) ->
    typename Parse_Context::iterator {

  auto&& _eh = ctx.error_handler();
  auto eh = error_handler_ref_t{&_eh};
  specs = {};

  auto it = ctx.begin();

  // parsing optional fill-align
  if (it == ctx.end() or *it == '}') {
    return it;
  }

  if (*it == '{') {
    it = parse_width_to_end(specs, it, ctx);
    return it;
  }

  auto it_old = it;

  fill_t fill = read_next_char(it, ctx.end(), eh);

  if (it == ctx.end()) {
    return it;
  }

  if (*it != '<' and *it != '^' and *it != '>') {
    fill = {{' ', {}, {}, {}}, 1};
    it = it_old;
  }

  if (*it == '<') {
    specs.fill = fill;
    specs.align = align_e::left;
    ++it;
  } else if (*it == '>') {
    specs.fill = fill;
    specs.align = align_e::right;
    ++it;
  } else if (*it == '^') {
    specs.fill = fill;
    specs.align = align_e::center;
    ++it;
  }

  if (it == ctx.end()) {
    return it;
  }

  if (*it == ' ') {
    specs.sign = sign_e::space;
    ++it;
  } else if (*it == '+') {
    specs.sign = sign_e::plus;
    ++it;
  } else if (*it == '-') {
    specs.sign = sign_e::minus;
    ++it;
  }

  if (it == ctx.end()) {
    return it;
  }

  if (*it == '#') {
    specs.alt = true;
    ++it;
  }

  if (it == ctx.end()) {
    return it;
  }

  if (*it == '0') {
    specs.pad_zero = true;
    ++it;
  }

  it = parse_width_to_end(specs, it, ctx);
  return it;
}

template <typename T, typename Enable = void> struct format_arg_to_int_impl {
  static auto
  run(T const& /* val */,
      int /* min_val */,
      error_handler_ref_t /* eh */,
      char const* /* too_small_msg */,
      char const* /* too_large_msg */) -> int {
    std::terminate();
  }
};

template <bool Signed1, bool Signed2> struct cmp_less_impl;

template <> struct cmp_less_impl<true, true> {
  template <typename U, typename V> static constexpr auto run(U u, V v) -> bool { return u < v; }
};
template <> struct cmp_less_impl<false, false> : cmp_less_impl<true, true> {};

template <> struct cmp_less_impl<true, false> {
  template <typename U, typename V> static constexpr auto run(U u, V v) -> bool {
    return u < 0 or static_cast<typename std::make_unsigned<U>::type>(u) < v;
  }
};

template <> struct cmp_less_impl<false, true> {
  template <typename U, typename V> static constexpr auto run(U u, V v) -> bool {
    return v >= 0 and u < static_cast<typename std::make_unsigned<V>::type>(v);
  }
};

template <typename U, typename V> constexpr auto cmp_less(U u, V v) -> bool {
  return cmp_less_impl<std::is_signed<U>::value, std::is_signed<V>::value>::run(u, v);
}

template <typename T>
struct format_arg_to_int_impl<                                            //
    T,                                                                    //
    typename ::mpfr::_::enable_if<std::is_integral<T>::value, void>::type //
    > {
  static auto
  run(T val,
      int min_val,
      error_handler_ref_t ctx,
      char const* too_small_msg,
      char const* too_large_msg) -> int {
    if (cmp_less(val, min_val)) {
      ctx.on_error(too_small_msg);
    }
    if (cmp_less(std::numeric_limits<int>::max(), val)) {
      ctx.on_error(too_large_msg);
    }
    return static_cast<int>(val);
  }
};

struct format_arg_to_int {
  error_handler_ref_t ctx;
  int min_val;
  char const* too_small_msg;
  char const* too_large_msg;

  template <typename T> auto operator()(T const& val) -> int {
    return format_arg_to_int_impl<T>::run(val, min_val, ctx, too_small_msg, too_large_msg);
  }
};

template <typename Format_Arg> struct format_context_ref_t {
private:
  void* m_ctx;
  auto (*m_arg)(void* ctx, int id) -> Format_Arg;

  template <typename Ctx> static auto arg_impl(void* ctx, int id) -> Format_Arg {
    return static_cast<Ctx*>(ctx)->arg(id);
  }

public:
  template <typename Ctx>
  format_context_ref_t(Ctx* ctx) // NOLINT(hicpp-explicit-conversions)
      : m_ctx{ctx}, m_arg{&arg_impl<Ctx>} {}

  auto arg(int id) -> Format_Arg { return m_arg(m_ctx, id); }
};

template <typename Format_Arg>
void parse_dynamic_args(
    mp_float_specs& specs, format_context_ref_t<Format_Arg> ctx, error_handler_ref_t eh) {

  if (specs.dyn_prec) {
    auto arg = ctx.arg(specs.prec_or_id);
    if (not arg.is_integral()) {
      eh.on_error("precision is not integral");
    }
    specs.prec_or_id = visit_format_arg(
        format_arg_to_int{
            eh,
            0,
            "precision must be non negative",
            "precision is too large",
        },
        arg);
  }

  if (specs.dyn_width) {
    auto arg = ctx.arg(specs.width_or_id);
    if (not arg.is_integral()) {
      eh.on_error("width is not integral");
    }
    specs.width_or_id = visit_format_arg(
        format_arg_to_int{
            eh,
            1,
            "width must be positive",
            "width is too large",
        },
        arg);
  }
}

inline char* format_to_buf(
    char* stack_buffer,
    size_t stack_bufsize,
    heap_str_t& heap_buffer,
    long& precision,
    long& width,
    bool& signbit,
    size_t& zero_padding,
    size_t& n_padding,
    size_t& left_padding,
    size_t& right_padding,
    mp_float_specs const& fspecs,
    mpfr_cref_t x) {
  char format[128] = {};
  std::size_t pos = 0;
  format[pos++] = '%';
  format[pos++] = '.';

  bool hex = fspecs.format == float_format_e::hex;
  bool bin = fspecs.format == float_format_e::binary;
  precision = hex   ? mpfr_get_prec(&x.m) / 4
              : bin ? mpfr_get_prec(&x.m)
                    : static_cast<long>(fspecs.prec_or_id);
  width = static_cast<long>(fspecs.width_or_id);
  signbit = mpfr_signbit(&x.m);

  if (precision < 0) {
    precision = 6;
  }
  std::snprintf(format + pos, sizeof(format) - pos, "%ld", precision);
  pos = std::strlen(format);
  format[pos++] = 'R';

  bool u = fspecs.upper;
  if (fspecs.format == float_format_e::hex) {
    format[pos++] = u ? 'A' : 'a';
  } else if (fspecs.format == float_format_e::binary) {
    format[pos++] = 'b';
  } else if (fspecs.format == float_format_e::exp) {
    format[pos++] = u ? 'E' : 'e';
  } else if (fspecs.format == float_format_e::fixed) {
    format[pos++] = u ? 'F' : 'f';
  } else {
    format[pos++] = u ? 'G' : 'g';
  }
  format[pos++] = '\0';

  MPFR_SIGN(&x.m) = 1;
  auto const size_needed = static_cast<std::size_t>(mpfr_snprintf(nullptr, 0, format, &x.m)) + 1;

  bool use_heap = size_needed > stack_bufsize;
  heap_buffer.init(use_heap ? size_needed : 0);

  char* ptr = use_heap ? heap_buffer.p : stack_buffer;

  mpfr_snprintf(ptr, size_needed, format, &x.m);
  if (not hex and fspecs.alt and precision > 0) {
    char* dot_ptr = std::strchr(ptr, '.');
    if (dot_ptr == nullptr) {
      zero_padding = 1 + static_cast<std::size_t>(precision) - size_needed;
    }
  }

  if (static_cast<std::size_t>(width) >=
      size_needed - 1 + ((signbit or fspecs.sign == sign_e::plus) ? 1 : 0)) {

    n_padding = static_cast<std::size_t>(width) - size_needed + 1 -
                ((signbit or fspecs.sign == sign_e::plus) ? 1 : 0) - zero_padding;
  }

  left_padding = (fspecs.align == align_e::right) //
                     ? n_padding
                     : (fspecs.align == align_e::center) //
                           ? (n_padding / 2)
                           : 0;
  right_padding = (fspecs.align == align_e::left) //
                      ? n_padding
                      : (fspecs.align == align_e::center) //
                            ? (n_padding - left_padding)
                            : 0;
  return ptr;
}

struct out_iter_ref_t {
private:
  void* m_out;
  void (*m_copy_to)(void* out, char const* begin, char const* end);

  template <typename Out_Iter>
  static void copy_into_out_impl(void* out, char const* begin, char const* end) {
    Out_Iter& out_concrete = *static_cast<Out_Iter*>(out);
    while (begin != end) {
      *out_concrete = *begin;
      ++out_concrete;
      ++begin;
    }
  }

public:
  template <typename Out_Iter>
  out_iter_ref_t(Out_Iter* out) // NOLINT(hicpp-explicit-conversions)
      : m_out{out}, m_copy_to{&copy_into_out_impl<Out_Iter>} {}

  void copy_into_out(char const* begin, char const* end) const { m_copy_to(m_out, begin, end); }
};

inline void format_impl(
    mpfr_cref_t value,
    out_iter_ref_t out,
    char* stack_buffer,
    std::size_t stack_bufsize,
    mp_float_specs const& specs) {

  using std::size_t;

  long precision = 0;
  long width = 0;
  size_t zero_padding = 0;
  size_t n_padding = 0;
  bool signbit = false;
  size_t left_padding = 0;
  size_t right_padding = 0;

  mpfr::_::heap_str_t heap_buffer{0};
  char const* const ptr = format_to_buf(
      stack_buffer,
      stack_bufsize,
      heap_buffer,
      precision,
      width,
      signbit,
      zero_padding,
      n_padding,
      left_padding,
      right_padding,
      specs,
      value);

  alignas(0x400) char buffer[0x400];

  char* pos = buffer;
  size_t space = sizeof(buffer);

  auto flush_buf = [&] {
    out.copy_into_out(buffer, pos);
    pos = buffer;
    space = sizeof(buffer);
  };

  auto chars_to_buf = [&](char const* begin, std::size_t len) {
    while (len > 0) {
      if (len <= space) {
        std::memcpy(pos, begin, len);
        pos += len;
        space -= len;
        return;
      } else {
        std::memcpy(pos, begin, space);
        len -= space;
        flush_buf();
      }
    }
  };

  auto fill_n = [&](size_t n) {
    for (size_t i = 0; i < n; ++i) {
      chars_to_buf(specs.fill.data, specs.fill.size);
    }
  };

  fill_n(left_padding);
  (signbit ? chars_to_buf("-", 1) : specs.sign == sign_e::plus ? chars_to_buf("+", 1) : (void)0);
  fill_n(0); // inner padding?
  chars_to_buf(ptr, std::strlen(ptr));
  ((zero_padding > 0) ? chars_to_buf(".", 1) : (void)0);
  for (size_t i = 0; i < zero_padding; ++i) {
    chars_to_buf("0", 1);
  }
  fill_n(right_padding);
  flush_buf();
}

struct base_parser {
  template <typename Parse_Context>
  MPFR_CXX_CONSTEXPR auto parse(Parse_Context& ctx) -> typename Parse_Context::iterator {
    return ::mpfr::_::libfmt::parse_mp_float_type_specs(specs, ctx);
  }
  mp_float_specs specs;
};

} // namespace libfmt
} // namespace _
} // namespace mpfr

namespace fmt {

template <mpfr::precision_t P>
struct formatter<mpfr::mp_float_t<P>, char, void> : ::mpfr::_::libfmt::base_parser {
  template <typename Format_Context>
  auto format(mpfr::mp_float_t<P> const& value, Format_Context& ctx) ->
      typename Format_Context::iterator {

    using format_context_ref_t = ::mpfr::_::libfmt::format_context_ref_t<decltype(ctx.arg(int{}))>;
    {
      auto eh = ctx.error_handler();
      ::mpfr::_::libfmt::parse_dynamic_args(specs, format_context_ref_t{&ctx}, {&eh});
    }

    constexpr std::size_t stack_bufsize =
        ::mpfr::_::digits2_to_10(static_cast<std::size_t>(::mpfr::mp_float_t<P>::precision) + 64);
    char stack_buffer[stack_bufsize];

    auto out = ctx.out();
    ::mpfr::_::libfmt::format_impl(
        ::mpfr::_::impl_access::mpfr_cref(value),
        ::mpfr::_::libfmt::out_iter_ref_t(&out),
        stack_buffer,
        stack_bufsize,
        ::mpfr::_::libfmt::base_parser::specs);
    return out;
  }
};

} // namespace fmt

#include "mpfr/detail/epilogue.hpp"

#endif /* end of include guard FMT_HPP_8STTEZY1 */
