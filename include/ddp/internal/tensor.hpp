#ifndef TENSOR_HPP_PTZWQRYY
#define TENSOR_HPP_PTZWQRYY

#include "ddp/internal/matrix_seq.hpp"
#include "ddp/internal/eigen.hpp"
#include <memory>
#include "veg/internal/prologue.hpp"

namespace ddp {
namespace idx {

struct TensorDims {
	struct Layout {
		i64 out, left, right;
	} self;
	auto out() const -> i64 { return self.out; }
	auto left() const -> i64 { return self.left; }
	auto right() const -> i64 { return self.right; }
};

namespace internal {

struct TensorLayout {
	i64 begin;
	i64 end;
	i64 max_out;
	i64 max_left;
	i64 max_right;
};
template <typename Idx>
struct TensorIdxViewBase {
	auto derived() const -> Idx const& { return static_cast<Idx const&>(*this); }
	auto dim_data() const -> Slice<TensorDims const> {
		return {
				from_raw_parts,
				derived().dim_data(),
				index_end() - index_begin(),
				unsafe,
		};
	}
	auto offset_data() const -> Slice<i64 const> {
		return {
				from_raw_parts,
				derived().offset_data(),
				index_end() - index_begin() + 1,
				unsafe,
		};
	}

	using Layout = internal::TensorLayout;
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
	auto out(i64 t) const -> i64 {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].out();
	}
	auto left(i64 t) const -> i64 {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].left();
	}
	auto right(i64 t) const -> i64 {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].right();
	}
};

} // namespace internal

struct TensorIdxView : internal::TensorIdxViewBase<TensorIdxView> {
	using Base = internal::TensorIdxViewBase<TensorIdxView>;
	struct Layout {
		TensorDims const* const dim_data;
		i64 const* const offset_data;
	} self;

	auto dim_data() const { return self.dim_data; }
	auto offset_data() const { return self.offset_data; }

	TensorIdxView(typename Base::Layout b, Layout s) : Base{b}, self{s} {}

	auto as_view() const -> TensorIdxView { return *this; }

	using Base::index_begin;
	using Base::index_end;
	using Base::required_memory;
	using Base::offset;
	using Base::out;
	using Base::left;
	using Base::right;
};

struct TensorIdx : internal::TensorIdxViewBase<TensorIdx> {
	using Base = internal::TensorIdxViewBase<TensorIdx>;
	friend struct internal::TensorIdxViewBase<TensorIdx>;
	struct Layout {
		std::vector<TensorDims> dim_data;
		std::vector<i64> offset_data;
	} self;

	auto dim_data() const { return self.dim_data.data(); }
	auto offset_data() const { return self.offset_data.data(); }

	TensorIdx(typename Base::Layout b, Layout s) : Base{b}, self{VEG_FWD(s)} {}

	template <typename Fn>
	static auto make(i64 begin, i64 end, Fn dims) -> TensorIdx {
		typename Base::Layout b{begin, end, 0, 0, 0};
		Layout self{{}, {}};

		self.offset_data.resize(narrow<usize>(b.end - b.begin + 1));
		self.dim_data.resize(narrow<usize>(b.end - b.begin));
		self.offset_data[0] = 0;
		i64& maxout = b.max_out;
		i64& maxleft = b.max_left;
		i64& maxright = b.max_right;
		for (i64 t = begin; t < b.end; ++t) {

			TensorDims olr = dims(t);
			i64 o = olr.out();
			i64 l = olr.left();
			i64 r = olr.right();

			VEG_DEBUG_ASSERT_ALL_OF((o >= 0), (l >= 0), (r >= 0));

			self.dim_data[narrow<usize>(t - begin)] = olr;
			self.offset_data[narrow<usize>(t - begin + 1)] =
					self.offset_data[narrow<usize>(t - begin)] + o * l * r;

			maxout = o > maxout ? o : maxout;
			maxleft = l > maxleft ? l : maxleft;
			maxright = r > maxright ? r : maxright;
		}
		return {b, VEG_FWD(self)};
	}

public:
	VEG_TEMPLATE(
			typename Fn,
			requires(VEG_CONCEPT(
					same<TensorDims, meta::detected_t<meta::invoke_result_t, Fn&, i64>>)),
			TensorIdx,
			(begin, i64),
			(end, i64),
			(dims, Fn))
			: TensorIdx(make(begin, end, VEG_FWD(dims))) {}

	auto as_view() const -> TensorIdxView {
		return {
				static_cast<Base const&>(*this).self,
				{self.dim_data.data(), self.offset_data.data()}};
	}

	using Base::index_begin;
	using Base::index_end;
	using Base::required_memory;
	using Base::offset;
	using Base::out;
	using Base::left;
	using Base::right;
};

} // namespace idx

namespace tensor {

template <typename T>
struct TensorView {
	struct Layout {
		T* data;
		i64 outdim;
		i64 indiml;
		i64 indimr;
	} self;

	using value_type = std::remove_const_t<T>;

	auto data() const -> T* { return self.data; }
	auto outdim() const -> i64 { return self.outdim; }
	auto indiml() const -> i64 { return self.indiml; }
	auto indimr() const -> i64 { return self.indimr; }
	auto operator()(i64 i, i64 j, i64 k) const -> T& {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(i < self.outdim),
				(j < self.indiml),
				(k < self.indimr));
		return self.data[i + j * self.outdim + k * self.outdim * self.indiml];
	}

	void assign(TensorView<value_type const> other) {
		static_assert(!std::is_const<T>::value, "");
		VEG_DEBUG_ASSERT_ALL_OF(
				(self.outdim == other.self.outdim),
				(self.indiml == other.self.indiml),
				(self.indimr == other.self.indimr));

		i64 size = self.outdim * self.indiml * self.indimr;
		if (std::is_trivially_copyable<value_type>::value) {
			std::memcpy(
					self.data, other.self.data, sizeof(value_type) * narrow<usize>(size));
		} else {
			if (std::less<value_type const*>{}(self.data, other.self.data)) {
				for (i64 i = 0; i < size; ++i) {
					self.data[i] = other.self.data[i];
				}
			} else if (std::less<value_type const*>{}(other.self.data, self.data)) {
				for (i64 i = size - 1; i >= 0; --i) {
					self.data[i] = other.self.data[i];
				}
			}
		}
	}
	void set_constant(T const& constant) {
		static_assert(!std::is_const<T>::value, "");

		for (i64 i = 0; i < self.outdim * self.indiml * self.indimr; ++i) {
			self.data[i] = constant;
		}
	}

	void noalias_contract_add_outdim( //
			View<value_type, colmat> out,
			View<value_type const, colvec> v) const {

		VEG_DEBUG_ASSERT_ALL_OF( //
				(v.rows() == self.outdim),
				(out.rows() == self.indiml),
				(out.cols() == self.indimr));
		if (out.rows() != 1) {
			VEG_DEBUG_ASSERT_ELSE(
					"non contiguous matrix", out.outerStride() == out.rows());
		}

		View<value_type const, colmat> //
				in_(self.data, self.outdim, self.indiml * self.indimr, self.outdim);
		Eigen::Map<
				Eigen::Matrix<value_type, 1, -1, Eigen::RowMajor>,
				Eigen::Unaligned> //
				out_(out.data(), 1, self.indiml * self.indimr);

		eigen::mul_add_to_noalias(out_, v.transpose(), in_);
	}

	auto has_nan() const -> bool {
		using std::isnan;
		for (i64 i = 0; i < self.outdim * self.indiml * self.indimr; ++i) {
			if (isnan(self.data[i])) {
				return true;
			}
		}
		return false;
	}

	void print() const {
		for (i64 k = 0; k < self.outdim; ++k) {
			for (i64 i = 0; i < self.indiml; ++i) {
				for (i64 j = 0; j < self.indimr; ++j) {
					::fmt::print("{:>15}  ", (*this)(k, i, j));
				}
				::fmt::print("\n");
			}
			::fmt::print("\n");
		}
	}
};

} // namespace tensor
using tensor::TensorView;

namespace internal {

template <typename T>
struct TensorSeqView {

	struct Layout {
		idx::TensorIdxView const idx;
		Slice<T> const data;
	} self;

	auto as_view() const -> TensorSeqView { return *this; }

	auto operator[](i64 t) const -> TensorView<T> {
		VEG_DEBUG_ASSERT_ALL_OF( //
				(t >= self.idx.index_begin()),
				(t < self.idx.index_end()));
		return {
				self.data.data() + self.idx.offset(t),
				self.idx.out(t),
				self.idx.left(t),
				self.idx.right(t),
		};
	}
};

template <typename T>
struct TensorSeq {
	static_assert(!std::is_const<T>::value, "");

	struct Layout {
		std::vector<T> data;
		idx::TensorIdx idx;
	} self;

	explicit TensorSeq(idx::TensorIdx idx)
			: self{
						std::vector<T>(narrow<usize>(idx.required_memory())),
						VEG_FWD(idx)} {}

	auto as_view() const -> TensorSeqView<T const> {
		return {
				self.idx.as_view(),
				{as_ref, self.data},
		};
	}
	auto as_view() -> TensorSeqView<T> {
		return {
				self.idx.as_view(),
				{as_ref, self.data},
		};
		;
	}

	auto operator[](i64 t) const -> TensorView<T const> { return as_view()[t]; }
	auto operator[](i64 t) -> TensorView<T> { return as_view()[t]; }
};
} // namespace internal
} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard TENSOR_HPP_PTZWQRYY */
