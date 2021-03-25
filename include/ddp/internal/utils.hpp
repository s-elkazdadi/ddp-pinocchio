#ifndef DDP_PINOCCHIO_UTILS_HPP_ABWKK0JNS
#define DDP_PINOCCHIO_UTILS_HPP_ABWKK0JNS

#include <veg/assert.hpp>
#include <veg/fn_view.hpp>
#include <veg/dynamic_stack.hpp>
#include <veg/tuple.hpp>
#include <veg/option.hpp>
#include <fmt/ostream.h>

namespace ddp {
using namespace veg;
namespace make {
using namespace veg::make;
} // namespace make
namespace meta {
using namespace veg::meta;
} // namespace meta

struct mem_req {
	i64 align;
	i64 size;
	template <typename T>
	constexpr mem_req(tag_t<T>, i64 n) noexcept
			: align{alignof(T)}, size{narrow<i64>(sizeof(T)) * n} {}

	constexpr mem_req(i64 al, i64 bytes) noexcept : align{al}, size{bytes} {}

	static constexpr auto sum_of(slice<mem_req const> arr) -> mem_req {
		mem_req m = {1, 0};
		for (auto const* p = arr.data(); p < arr.data() + arr.size(); ++p) {
			m.size += p->size;
			m.align = meta::max2(p->align, m.align);
		}
		return m;
	}

	static constexpr auto max_of(slice<mem_req const> arr) -> mem_req {
		mem_req m = {1, 0};
		for (auto const* p = arr.data(); p < arr.data() + arr.size(); ++p) {
			m.size = meta::max2(p->size, m.size);
			m.align = meta::max2(p->align, m.align);
		}
		return m;
	}
};

struct move_only {
	move_only() = default;
	~move_only();
	move_only(move_only&&) noexcept = default;
	auto operator=(move_only&&) noexcept -> move_only& = default;
	move_only(move_only const&) noexcept = delete;
	auto operator=(move_only const&) noexcept -> move_only& = delete;
};
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_UTILS_HPP_ABWKK0JNS */
