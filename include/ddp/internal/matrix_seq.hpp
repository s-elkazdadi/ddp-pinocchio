#ifndef DDP_PINOCCHIO_MATRIX_SEQ_HPP_SHW3X8TQS
#define DDP_PINOCCHIO_MATRIX_SEQ_HPP_SHW3X8TQS

#include "ddp/internal/eigen.hpp"
#include <memory>
#include <vector>
#include "veg/internal/prologue.hpp"

namespace ddp {
namespace idx {

template <eigen::Kind>
struct Dims;

template <>
struct Dims<colvec> {
	i64 row_;
	auto rows() const -> i64 { return row_; }
	auto cols() const -> i64 { return unused(this), 1; }
};

template <>
struct Dims<colmat> {
	i64 row_;
	i64 col_;
	auto rows() const -> i64 { return row_; }
	auto cols() const -> i64 { return col_; }
};
namespace internal {

struct Layout {
	i64 begin;
	i64 end;
	i64 max_rows;
	i64 max_cols;
};

template <typename Idx, eigen::Kind K>
struct IdxBase {
	auto derived() const -> Idx const& { return static_cast<Idx const&>(*this); }
	auto dim_data() const -> Slice<Dims<K> const> {
		return {
				from_raw_parts,
				derived().dim_data_impl(),
				index_end() - index_begin(),
				unsafe,
		};
	}
	auto offset_data() const -> Slice<i64 const> {
		return {
				from_raw_parts,
				derived().offset_data_impl(),
				index_end() - index_begin() + 1,
				unsafe,
		};
	}

	using Layout = internal::Layout;
	Layout self;

	auto index_begin() const -> i64 { return self.begin; }
	auto index_end() const -> i64 { return self.end; }

	auto required_memory() const -> i64 { return offset(index_end()); }
	auto offset(i64 t) const -> i64 {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t <= index_end()));
		return offset_data()[t - index_begin()];
	}

	auto max_rows() const -> i64 { return self.max_rows; }
	auto max_cols() const -> i64 { return self.max_cols; }

	auto rows(i64 t) const -> i64 {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].rows();
	}
	auto cols(i64 t) const -> i64 {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].cols();
	}
};

} // namespace internal

template <eigen::Kind K>
struct Idx;

template <eigen::Kind K>
struct IdxView : internal::IdxBase<IdxView<K>, K> {
	using Base = internal::IdxBase<IdxView<K>, K>;
	struct Layout {
		Dims<K> const* dim_data;
		i64 const* offset_data;
	} self;

	auto dim_data_impl() const { return self.dim_data; }
	auto offset_data_impl() const { return self.offset_data; }

	IdxView(typename Base::Layout b, Layout s) : Base{b}, self{s} {}

	auto as_view() const -> IdxView { return *this; }

	using Base::index_begin;
	using Base::index_end;
	using Base::required_memory;
	using Base::offset;
	using Base::rows;
	using Base::cols;
};

template <eigen::Kind K>
struct Idx : internal::IdxBase<Idx<K>, K> {
	using Base = internal::IdxBase<Idx<K>, K>;

	auto dim_data_impl() const { return self.dim_data.data(); }
	auto offset_data_impl() const { return self.offset_data.data(); }

	friend struct internal::IdxBase<IdxView<K>, K>;
	struct Layout {
		std::vector<Dims<K>> dim_data;
		std::vector<i64> offset_data;
	} self;

	Idx(typename Base::Layout b, Layout s) : Base{b}, self{VEG_FWD(s)} {}

	template <typename Fn>
	static auto make(i64 begin, i64 end, Fn dim_fn) -> Idx {
		typename Base::Layout b{begin, end, 0, 0};
		Layout self{{}, {}};

		constexpr auto to_usize = narrow<usize>;
		self.offset_data.resize(to_usize(b.end - b.begin + 1));
		self.dim_data.resize(to_usize(b.end - b.begin));
		self.offset_data[0] = 0;
		for (i64 t = begin; t < b.end; ++t) {

			Dims<K> rc = dim_fn(t);
			i64 r = rc.rows();
			i64 c = rc.cols();

			VEG_DEBUG_ASSERT_ALL_OF((r >= 0), (c >= 0));

			self.dim_data[to_usize(t - begin)] = rc;
			self.offset_data[to_usize(t - begin + 1)] =
					self.offset_data[to_usize(t - begin)] + r * c;

			b.max_rows = (r > b.max_rows) ? r : b.max_rows;
			b.max_cols = (c > b.max_cols) ? c : b.max_cols;
		}
		return {b, VEG_FWD(self)};
	}

public:
	DDP_TEMPLATE(
			typename Fn,
			requires(VEG_CONCEPT(invocable_r<Fn&, Dims<K>, i64>)),
			Idx,
			(begin, i64),
			(end, i64),
			(dims, Fn))
			: Idx{make(begin, end, VEG_FWD(dims))} {}

	auto as_view() const -> IdxView<K> {
		return {Base::self, {self.dim_data.data(), self.offset_data.data()}};
	}

	auto into_parts() && -> Tuple<typename Base::Layout, Layout> {
		return {direct, Base::self, self};
	}

	using Base::index_begin;
	using Base::index_end;
	using Base::required_memory;
	using Base::offset;
	using Base::rows;
	using Base::cols;
};

} // namespace idx
namespace internal {

template <typename T, eigen::Kind K>
struct MatSeq {
	DDP_CHECK_CONCEPT(scalar<T>);

	static_assert(!std::is_const<T>::value, "");
	using const_view = View<T const, K>;
	using mut_view = View<T, K>;

	struct Layout {
		std::vector<T> data;
		idx::Idx<K> idx;
	} self;

	explicit MatSeq(Layout l) : self{VEG_FWD(l)} {}

public:
	explicit MatSeq(idx::Idx<K> idx)
			: self{
						std::vector<T>(narrow<usize>(idx.required_memory())),
						VEG_FWD(idx)} {}

	auto index_begin() const -> i64 { return self.idx.index_begin(); }
	auto index_end() const -> i64 { return self.idx.index_end(); }

	auto operator[](i64 t) const -> const_view {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= self.idx.index_begin()),
				(t < self.idx.index_end()));
		return View<T const, K>{
				self.data.data() + (self.idx.offset(t)),
				self.idx.rows(t),
				self.idx.cols(t),
				self.idx.rows(t),
		};
	}
	auto operator[](i64 t) -> mut_view {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= self.idx.index_begin()),
				(t < self.idx.index_end()));
		return View<T, K>{
				self.data.data() + self.idx.offset(t),
				self.idx.rows(t),
				self.idx.cols(t),
				self.idx.rows(t),
		};
	}
};

} // namespace internal
} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_MATRIX_SEQ_HPP_SHW3X8TQS */
