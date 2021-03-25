#ifndef TENSOR_HPP_PTZWQRYY
#define TENSOR_HPP_PTZWQRYY

#include "ddp/internal/matrix_seq.hpp"
#include "ddp/internal/eigen.hpp"
#include <memory>
#include "veg/internal/prologue.hpp"

namespace ddp {
namespace idx {

struct tensor_dims {
	i64 out_, left_, right_;
	auto out() const -> i64 { return out_; }
	auto left() const -> i64 { return left_; }
	auto right() const -> i64 { return right_; }
};

namespace internal {

struct tensor_layout {
	i64 begin;
	i64 end;
	i64 max_out;
	i64 max_left;
	i64 max_right;
};
template <typename Idx>
struct tensor_idx_view_base {
	auto derived() const -> Idx const& { return static_cast<Idx const&>(*this); }
	auto dim_data() const -> slice<tensor_dims const> {
		return {derived().dim_data(), index_end() - index_begin()};
	}
	auto offset_data() const -> slice<i64 const> {
		return {derived().offset_data(), index_end() - index_begin() + 1};
	}

	using layout = internal::tensor_layout;
	layout const self;

	auto index_begin() const -> i64 { return self.begin; }
	auto index_end() const -> i64 { return self.end; }

	auto required_memory() const -> i64 { return offset(index_end()); }
	auto offset(i64 t) const -> i64 {
		VEG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t <= index_end()));
		return offset_data()[t - index_begin()];
	}
	auto out(i64 t) const -> i64 {
		VEG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].out();
	}
	auto left(i64 t) const -> i64 {
		VEG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].left();
	}
	auto right(i64 t) const -> i64 {
		VEG_ASSERT_ALL_OF( //
				(t >= index_begin()),
				(t < index_end()));
		return dim_data()[t - index_begin()].right();
	}
};

} // namespace internal

struct tensor_idx_view : internal::tensor_idx_view_base<tensor_idx_view> {
	using base = internal::tensor_idx_view_base<tensor_idx_view>;
	struct layout {
		tensor_dims const* const dim_data;
		i64 const* const offset_data;
	} self;

	auto dim_data() const { return self.dim_data; }
	auto offset_data() const { return self.offset_data; }

	tensor_idx_view(typename base::layout b, layout s) : base{b}, self{s} {}

	auto as_view() const -> tensor_idx_view { return *this; }

	using base::index_begin;
	using base::index_end;
	using base::required_memory;
	using base::offset;
	using base::out;
	using base::left;
	using base::right;
};

struct tensor_idx : internal::tensor_idx_view_base<tensor_idx> {
	using base = internal::tensor_idx_view_base<tensor_idx>;
	friend struct internal::tensor_idx_view_base<tensor_idx>;
	struct layout {
		std::vector<tensor_dims> dim_data;
		std::vector<i64> offset_data;
	} self;

	auto dim_data() const { return self.dim_data.data(); }
	auto offset_data() const { return self.offset_data.data(); }

	tensor_idx(typename base::layout b, layout s) : base{b}, self{VEG_FWD(s)} {}

	template <typename Fn>
	static auto make(i64 begin, i64 end, Fn dims) -> tensor_idx {
		typename base::layout b{begin, end, 0, 0, 0};
		layout self{{}, {}};

		self.offset_data.resize(narrow<usize>(b.end - b.begin + 1));
		self.dim_data.resize(narrow<usize>(b.end - b.begin));
		self.offset_data[0] = 0;
		i64& maxout = b.max_out;
		i64& maxleft = b.max_left;
		i64& maxright = b.max_right;
		for (i64 t = begin; t < b.end; ++t) {

			tensor_dims olr = dims(t);
			i64 o = olr.out();
			i64 l = olr.left();
			i64 r = olr.right();

			VEG_ASSERT_ALL_OF((o >= 0), (l >= 0), (r >= 0));

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
			requires(VEG_CONCEPT(same<
													 tensor_dims,
													 meta::detected_t<meta::invoke_result_t, Fn&, i64>>)),
			tensor_idx,
			(begin, i64),
			(end, i64),
			(dims, Fn))
			: tensor_idx(make(begin, end, VEG_FWD(dims))) {}

	auto as_view() const -> tensor_idx_view {
		return {
				static_cast<base const&>(*this).self,
				{self.dim_data.data(), self.offset_data.data()}};
	}

	using base::index_begin;
	using base::index_end;
	using base::required_memory;
	using base::offset;
	using base::out;
	using base::left;
	using base::right;
};

} // namespace idx

namespace tensor {

template <typename T>
struct tensor_view {
	struct layout {
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
		VEG_ASSERT_ALL_OF( //
				(i < self.outdim),
				(j < self.indiml),
				(k < self.indimr));
		return self.data[i + j * self.outdim + k * self.outdim * self.indiml];
	}

	void assign(tensor_view<value_type const> other) {
		static_assert(!std::is_const<T>::value, "");
		VEG_ASSERT_ALL_OF(
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
			view<value_type, colmat> out,
			view<value_type const, colvec> v) const {

		VEG_ASSERT_ALL_OF( //
				(v.rows() == self.outdim),
				(out.rows() == self.indiml),
				(out.cols() == self.indimr));
		if (out.rows() != 1) {
			VEG_ASSERT_ELSE("non contiguous matrix", out.outerStride() == out.rows());
		}

		view<value_type const, colmat> //
				in_(self.data, self.outdim, self.indiml * self.indimr, self.outdim);
		Eigen::Map<
				Eigen::Matrix<value_type, 1, -1, Eigen::RowMajor>,
				Eigen::Unaligned> //
				out_(out.data(), 1, self.indiml * self.indimr);

		eigen::tmul_add_to_noalias(out_, v, in_);
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
using tensor::tensor_view;

namespace internal {

template <typename T>
struct tensor_seq_view {

	struct layout {
		idx::tensor_idx_view const idx;
		slice<T> const data;
	} self;

	auto as_view() const -> tensor_seq_view { return *this; }

	auto operator[](i64 t) const -> tensor_view<T> {
		VEG_ASSERT_ALL_OF( //
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
struct tensor_seq {
	static_assert(!std::is_const<T>::value, "");

	struct layout {
		std::vector<T> data;
		idx::tensor_idx idx;
	} self;

	explicit tensor_seq(idx::tensor_idx idx)
			: self{
						std::vector<T>(narrow<usize>(idx.required_memory())),
						VEG_FWD(idx)} {}

	auto as_view() const -> tensor_seq_view<T const> {
		return {self.idx.as_view(), self.data};
	}
	auto as_view() -> tensor_seq_view<T> {
		return {self.idx.as_view(), self.data};
		;
	}

	auto operator[](i64 t) const -> tensor_view<T const> { return as_view()[t]; }
	auto operator[](i64 t) -> tensor_view<T> { return as_view()[t]; }
};
} // namespace internal
} // namespace ddp

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard TENSOR_HPP_PTZWQRYY */
