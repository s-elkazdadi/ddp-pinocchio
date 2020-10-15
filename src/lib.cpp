#include "ddp/detail/tuple.hpp"
#include "ddp/detail/utils.hpp"

#include <cstdio>
#include <exception>
#include <set>
#include <vector>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <initializer_list>

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <boost/filesystem.hpp>

namespace ddp {
namespace assertion {

struct assertion_data {
  fmt::string_view expr;
  fmt::string_view message;
  fmt::string_view op;
  std::string lhs;
  std::string rhs;
};

thread_local std::vector<assertion_data> failed_asserts = {};

template <typename T>
struct finally_t {
  T callback;
  finally_t(finally_t const&) = delete;
  finally_t(finally_t&&) = delete;
  auto operator=(finally_t const&) -> finally_t& = delete;
  auto operator=(finally_t&&) -> finally_t& = delete;
  ~finally_t() { callback(); }
};

template <typename T>
HEDLEY_WARN_UNUSED_RESULT auto finally(T&& callback) -> finally_t<T> {
  return {callback};
}

void on_fail(long line, char const* file, char const* func, bool is_fatal) {
  using namespace fmt;
  auto&& clear = finally([&] { failed_asserts.clear(); });
  (void)clear;

  std::string output;

  output = DDP_MOVE(output) + format(fg(color::olive), "{:=<79}", "");
  output = DDP_MOVE(output) + format("\n");
  output = DDP_MOVE(output) + format(fg(color::azure), "{}", failed_asserts.size());
  output = DDP_MOVE(output) + format(" assertion{} ", failed_asserts.size() > 1 ? "s" : "");
  output = DDP_MOVE(output) + format(fg(color::orange_red), "failed");
  output = DDP_MOVE(output) + format("\nin function:\n{}\n", func);
  output = DDP_MOVE(output) + format(fg(color::gray), "{}:{}: ", file, line);
  output = DDP_MOVE(output) + format("\n");

  char const* separator = "";

  for (auto const& a : failed_asserts) {
    char const* newline = std::find(a.message.begin(), a.message.end(), '\n');
    bool multiline = newline != a.message.end();

    output = DDP_MOVE(output) + format(separator);

    output = DDP_MOVE(output) + format(fg(color::orange_red), "{}", is_fatal ? "fatal " : "");
    output = DDP_MOVE(output) + format("assertion ");
    output = DDP_MOVE(output) + format("`");
    output = DDP_MOVE(output) + format(fg(color::azure), "{}", a.expr);
    output = DDP_MOVE(output) + format("'");
    output = DDP_MOVE(output) + format(fg(color::orange_red), " failed:");

    if (not multiline) {
      output = DDP_MOVE(output) + format(" {}", a.message);
    } else {
      char const* b = a.message.begin();
      char const* e = newline;

      while (b != nullptr) {

        output = DDP_MOVE(output) + format("\n > {}", string_view{b, static_cast<size_t>(e - b)});

        if (e == a.message.end()) {
          b = nullptr;
        } else {
          b = e + 1;
          e = std::find(b, a.message.end(), '\n');
        }
      }
    }

    output = DDP_MOVE(output) + format("\nassertion expands to: `");
    output = DDP_MOVE(output) + format(fg(color::azure), "{}{}{}", a.lhs, a.op, a.rhs);
    output = DDP_MOVE(output) + format("'\n");
    separator = "\n";
  }

  output = DDP_MOVE(output) + format(fg(color::olive), "{:=<79}", "");
  output = DDP_MOVE(output) + format("\n");

  print(stderr, output);
}

void on_expect_fail(long line, char const* file, char const* func) {
  on_fail(line, file, func, false);
}

[[noreturn]] void on_assert_fail(long line, char const* file, char const* func) {
  on_fail(line, file, func, true);
  std::terminate();
}

void set_assert_params(          //
    fmt::string_view expression, //
    fmt::string_view message,    //
    fmt::string_view op,         //
    std::string lhs,             //
    std::string rhs              //
) {
  failed_asserts.push_back({
      expression,
      message,
      op,
      static_cast<std::string&&>(lhs),
      static_cast<std::string&&>(rhs),
  });
}

} // namespace assertion

[[noreturn]] void fast_fail(fmt::string_view message) noexcept {
  print_msg(message);
  std::terminate();
}

void print_msg(fmt::string_view message) noexcept {
  std::fwrite(message.data(), 1, message.size(), stderr);
  std::fputc('\n', stderr);
}

struct file_t {
  file_t(file_t const&) = delete;
  file_t(file_t&& other) noexcept : m_ptr{other.m_ptr} { other.m_ptr = nullptr; };
  auto operator=(file_t const&) -> file_t = delete;
  auto operator=(file_t&&) -> file_t = delete;

  explicit file_t(char const* path, int buffering_mode = _IONBF) : m_ptr{std::fopen(path, "w")} {
    if (m_ptr == nullptr) {
      fast_fail("could not open file");
    }
    std::setvbuf(m_ptr, nullptr, buffering_mode, 0);
  }

  ~file_t() {
    if (m_ptr != nullptr) {
      std::fclose(m_ptr);
    }
  }

  auto ptr() const noexcept -> std::FILE* { return static_cast<std::FILE*>(m_ptr); }

private:
  gsl::owner<std::FILE*> m_ptr;
};

struct boost_file_t {
  explicit boost_file_t(boost::filesystem::path path_)
      : path_b{static_cast<boost::filesystem::path&&>(path_)}, path_ptr(path_b.string().c_str()) {}

  auto boost() const -> boost::filesystem::path const& { return path_b; }
  auto ptr() const -> std::FILE* { return path_ptr.ptr(); }

  friend auto operator<(boost_file_t const& a, boost_file_t const& b) {
    return std::less<std::FILE*>{}(a.ptr(), b.ptr());
  }

private:
  boost::filesystem::path path_b;
  file_t path_ptr;
};

struct log_file_t::open_file_set_t {
  std::set<boost_file_t> set;
  std::mutex mutex;
};

log_file_t::open_file_set_t log_file_t::open_files;

log_file_t::log_file_t(char const* const path) {
  std::lock_guard<std::mutex> g{open_files.mutex};

  auto it = open_files.set.begin();
  for (; it != open_files.set.end(); ++it) {
    if (boost::filesystem::equivalent(it->boost(), path)) {
      break;
    }
  }

  if (it == open_files.set.end()) {
    auto result = open_files.set.emplace(boost::filesystem::absolute(boost::filesystem::path{path}));
    DDP_ASSERT_MSG("file already exists", result.second);
    it = result.first;
  }
  ptr = it->ptr();
}

chronometer_t::chronometer_t(char const* message, log_file_t file)
    : m_begin{}, m_end{}, m_message{message}, m_file{static_cast<log_file_t&&>(file)} {
  using clock_t = std::chrono::steady_clock;
  m_begin = clock_t::now().time_since_epoch().count();
}

chronometer_t::~chronometer_t() {
  using clock_t = std::chrono::steady_clock;

  m_end = clock_t::now().time_since_epoch().count();

  fmt::print(
      m_file.ptr,
      "finished: {} | {} elapsed\n",
      m_message,
      std::chrono::duration<double, std::milli>(clock_t::duration{m_end - m_begin}));
}

} // namespace ddp
