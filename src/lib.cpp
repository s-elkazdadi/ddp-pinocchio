#include "ddp/detail/utils.hpp"

#include <cstdio>
#include <exception>

namespace ddp {
[[noreturn]] void fast_fail(fmt::string_view message) noexcept {
  std::fwrite(message.data(), 1, message.size(), stderr);
  std::terminate();
}
} // namespace ddp
