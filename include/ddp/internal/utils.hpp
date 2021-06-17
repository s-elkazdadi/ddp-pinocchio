#ifndef DDP_PINOCCHIO_UTILS_HPP_ABWKK0JNS
#define DDP_PINOCCHIO_UTILS_HPP_ABWKK0JNS

#include <veg/util/assert.hpp>
#include <veg/fn_view.hpp>
#include <veg/memory/dynamic_stack.hpp>
#include <veg/tuple.hpp>
#include <veg/option.hpp>
#include <fmt/ostream.h>

#define DDP_CONCEPT(...) VEG_CONCEPT_MACRO(::ddp::concepts, __VA_ARGS__)
#define DDP_CHECK_CONCEPT(...)                                                 \
	VEG_CHECK_CONCEPT_MACRO(::ddp::concepts, __VA_ARGS__)

#define DDP_DEF_CONCEPT VEG_DEF_CONCEPT
#define DDP_DEF_CONCEPT_CONJUNCTION VEG_DEF_CONCEPT_CONJUNCTION
#define DDP_DEF_CONCEPT_DISJUNCTION VEG_DEF_CONCEPT_DISJUNCTION
#define DDP_TEMPLATE VEG_TEMPLATE
#define DDP_ALL_OF VEG_ALL_OF
#define DDP_ANY_OF VEG_ANY_OF
#define DDP_NIEBLOID VEG_NIEBLOID
#define DDP_NODISCARD VEG_NODISCARD

namespace ddp {
using namespace veg;
namespace make {
} // namespace make
namespace meta {
using namespace veg::meta;
} // namespace meta

struct MemReq {
	i64 align;
	i64 size;
	template <typename T>
	constexpr MemReq(Tag<T> /*unused*/, i64 n) noexcept
			: align{alignof(T)}, size{narrow<i64>(sizeof(T)) * n} {}

	constexpr MemReq(i64 al, i64 bytes) noexcept : align{al}, size{bytes} {}

	static constexpr auto sum_of(Slice<MemReq const> arr) -> MemReq {
		MemReq m = {1, 0};
		for (auto const* p = arr.data(); p < arr.data() + arr.size(); ++p) {
			m.size += p->size;
			m.align = (p->align > m.align) ? p->align : m.align;
		}
		return m;
	}

	static constexpr auto max_of(Slice<MemReq const> arr) -> MemReq {
		MemReq m = {1, 0};
		for (auto const* p = arr.data(); p < arr.data() + arr.size(); ++p) {
			m.size = (p->size > m.size) ? p->size : m.size;
			m.align = (p->align > m.align) ? p->align : m.align;
		}
		return m;
	}
};
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_UTILS_HPP_ABWKK0JNS */
