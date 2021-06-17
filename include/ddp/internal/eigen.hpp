#ifndef DDP_PINOCCHIO_EIGEN_HPP_BDOKZRTGS
#define DDP_PINOCCHIO_EIGEN_HPP_BDOKZRTGS

#include <Eigen/Core>
#include <vector>
#include "ddp/internal/utils.hpp"
#include <sys/ioctl.h> //ioctl() and TIOCGWINSZ
#include <unistd.h>    // for STDOUT_FILENO
#include "veg/internal/prologue.hpp"

namespace ddp {

namespace eigen {
namespace internal {
template <typename T>
using mat_scalar_t = typename T::Scalar;
template <typename T>
using mat_rows_t = veg::meta::constant<i64, i64(T::RowsAtCompileTime)>;
template <typename T>
using mat_cols_t = veg::meta::constant<i64, i64(T::ColsAtCompileTime)>;
} // namespace internal

template <typename T>
using scalar_t = veg::meta::detected_t<internal::mat_scalar_t, T>;
template <typename T>
using rows_t =
		veg::meta::detected_or_t<meta::constant<i64, 0>, internal::mat_rows_t, T>;
template <typename T>
using cols_t =
		veg::meta::detected_or_t<meta::constant<i64, 0>, internal::mat_cols_t, T>;
} // namespace eigen

namespace internal {

template <typename A, typename B>
using add_expr = decltype(VEG_DECLVAL(A &&) + VEG_DECLVAL(B &&));
template <typename A, typename B>
using sub_expr = decltype(VEG_DECLVAL(A &&) - VEG_DECLVAL(B &&));
template <typename A, typename B>
using mul_expr = decltype(VEG_DECLVAL(A &&) * VEG_DECLVAL(B &&));
template <typename A, typename B>
using div_expr = decltype(VEG_DECLVAL(A &&) / VEG_DECLVAL(B &&));

template <typename T>
using neg_expr = decltype(-VEG_DECLVAL(T &&));

template <typename T>
using data_expr = decltype(VEG_DECLVAL(T &&).data());
template <typename T>
using size_expr = decltype(VEG_DECLVAL(T &&).size());

} // namespace internal

namespace concepts {
namespace aux {

DDP_DEF_CONCEPT(
		typename T, has_data, VEG_CONCEPT(detected<internal::data_expr, T&>));
DDP_DEF_CONCEPT(
		typename T, has_size, VEG_CONCEPT(detected<internal::size_expr, T&>));

DDP_DEF_CONCEPT(
		(typename A), negatable, VEG_CONCEPT(detected<internal::neg_expr, A&&>));
DDP_DEF_CONCEPT(
		(typename A, typename B),
		addable,
		VEG_CONCEPT(detected<internal::add_expr, A&&, B&&>));
DDP_DEF_CONCEPT(
		(typename A, typename B),
		subtractible,
		VEG_CONCEPT(detected<internal::sub_expr, A&&, B&&>));
DDP_DEF_CONCEPT(
		(typename A, typename B),
		multipliable,
		VEG_CONCEPT(detected<internal::mul_expr, A&&, B&&>));
DDP_DEF_CONCEPT(
		(typename A, typename B),
		divisible,
		VEG_CONCEPT(detected<internal::div_expr, A&&, B&&>));
} // namespace aux

DDP_DEF_CONCEPT_CONJUNCTION(
		typename T,
		scalar,
		(

				(veg::concepts::, constructible<T>),
				(veg::concepts::, copy_constructible<T>),
				(veg::concepts::, move_constructible<T>),
				(veg::concepts::, copy_assignable<T>),
				(veg::concepts::, move_assignable<T>),

				(veg::concepts::, constructible<T, int>),
				(veg::concepts::, constructible<T, double>),
				(veg::concepts::, assignable<T&, int>),
				(veg::concepts::, assignable<T&, double>),

				(aux::, negatable<T const&>),
				(aux::, addable<T const&, T const&>),
				(aux::, subtractible<T const&, T const&>),
				(aux::, multipliable<T const&, T const&>),
				(aux::, divisible<T const&, T const&>),
				(aux::, multipliable<T const&, int>),
				(aux::, divisible<T const&, int>)

						));

namespace eigen {
namespace aux {

DDP_DEF_CONCEPT(
		typename T,
		has_eigen_base,
		VEG_CONCEPT(convertible<T, Eigen::MatrixBase<T> const&>));

DDP_DEF_CONCEPT(
		typename M, has_scalar, DDP_CONCEPT(scalar<typename M::Scalar>));

DDP_DEF_CONCEPT(
		(typename N, typename M),
		maybe_same,
		((M::value == N::value) || (N::value == Eigen::Dynamic) ||
     (M::value == Eigen::Dynamic)));

} // namespace aux

DDP_DEF_CONCEPT_CONJUNCTION(
		typename T, matrix, ((aux::, has_eigen_base<T>), (aux::, has_scalar<T>)));

DDP_DEF_CONCEPT_CONJUNCTION(
		typename T,
		vector,
		((, matrix<T>),
     (veg::concepts::,
      same<ddp::eigen::cols_t<T>, veg::meta::constant<i64, 1>>)));

} // namespace eigen
} // namespace concepts

namespace eigen {

template <typename T>
using member_data_t = meta::detected_t<ddp::internal::data_expr, T>;
template <typename T>
using member_size_t = meta::detected_t<ddp::internal::size_expr, T>;

enum struct kind { colvec, colmat, rowvec, rowmat };

template <typename T, kind K>
struct heap_matrix;

namespace internal {

template <typename T>
using fmt_formatter = meta::conditional_t<
		::fmt::has_formatter<T, ::fmt::format_context>::value,
		::fmt::formatter<T>,
		::fmt::detail::fallback_formatter<T>>;

template <typename T>
struct with_formatter {
	T const& val;
	fmt_formatter<T>& fmt;
};

template <typename T, typename OutIt>
auto format_impl(
		::fmt::basic_format_context<OutIt, char>& fc,
		i64 rows,
		i64 cols,
		fn::FnView<T(i64, i64)> getter,
		fmt_formatter<T>& fmt) -> decltype(fc.out()) {
	auto out = fc.out();

	bool const colmat = cols == 1;
	bool const multi = (cols > 1) && (rows > 1);

	out = ::fmt::format_to(
			out,
			"[{}|{:>2}×{:<2}] {}",
			(rows == 1)   ? "row"
			: (cols == 1) ? "col"
										: "mat",
			rows,
			cols,
			multi ? "{\n" : "{");

	i64 line_len = multi ? 0
	                     : narrow<i64>(::fmt::formatted_size(
														 "[xxx|{:>2}×{:<2}] x", rows, cols));

	char const* row_sep = "";
	char const* row_begin = colmat ? "" : "{";
	char const* row_end = colmat ? "" : "}";

	struct winsize size {};
	ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
	auto win_cols = size.ws_col;

	for (i64 i = 0; i < rows; ++i) {
		char const* col_sep = "";

		out = ::fmt::format_to(out, "{}", row_sep);
		line_len += narrow<i64>(::fmt::formatted_size("{}", row_sep));

		if (line_len >= win_cols - 10) {
			*out++ = '\n';
			*out++ = ' ';
			line_len = 1;
		}
		out = ::fmt::format_to(out, "{}", row_begin);
		line_len += narrow<i64>(::fmt::formatted_size("{}", row_begin));

		for (i64 j = 0; j < cols; ++j) {
			line_len += narrow<i64>(::fmt::formatted_size("{}", col_sep));
			out = ::fmt::format_to(out, "{}", col_sep);
			fc.advance_to(out);

			T v = getter(i, j);

			auto len =
					narrow<i64>(::fmt::formatted_size("{}", with_formatter<T>{v, fmt}));
			line_len += len;

			if (line_len >= win_cols - 10) {
				*out++ = '\n';
				*out++ = ' ';
				line_len = len + 1;
			}
			out = ::fmt::format_to(out, "{}", with_formatter<T>{v, fmt});

			col_sep = ", ";
		}
		line_len += narrow<i64>(::fmt::formatted_size("{}", row_end));
		out = ::fmt::format_to(out, "{}", row_end);
		row_sep = multi ? ",\n" : ", ";

		line_len = multi ? 0 : line_len;
	}
	out = ::fmt::format_to(out, "{}", multi ? "\n}" : "}");
	return out;
}

extern template auto ddp::eigen::internal::format_impl(
		::fmt::basic_format_context<::fmt::v7::detail::buffer_appender<char>, char>&
				fc,
		i64 rows,
		i64 cols,
		fn::FnView<float(i64, i64)> getter,
		fmt_formatter<float>& fmt) -> decltype(fc.out());

extern template auto ddp::eigen::internal::format_impl(
		::fmt::basic_format_context<::fmt::v7::detail::buffer_appender<char>, char>&
				fc,
		i64 rows,
		i64 cols,
		fn::FnView<double(i64, i64)> getter,
		fmt_formatter<double>& fmt) -> decltype(fc.out());

extern template auto ddp::eigen::internal::format_impl(
		::fmt::basic_format_context<::fmt::v7::detail::buffer_appender<char>, char>&
				fc,
		i64 rows,
		i64 cols,
		fn::FnView<long double(i64, i64)> getter,
		fmt_formatter<long double>& fmt) -> decltype(fc.out());

template <typename T>
constexpr auto fail(T* p = nullptr) -> T {
	return *p;
}

template <typename T, kind K>
struct to_matrix;

template <typename T>
struct to_matrix<T, kind::colmat> {
	using type = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
};
template <typename T>
struct to_matrix<T, kind::colvec> {
	using type = Eigen::Matrix<T, -1, 1, Eigen::ColMajor>;
};
template <typename T>
struct to_matrix<T, kind::rowmat> {
	using type = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
};
template <typename T>
struct to_matrix<T, kind::rowvec> {
	using type = Eigen::Matrix<T, 1, -1, Eigen::RowMajor>;
};

template <bool B, typename T>
using add_const_if = meta::conditional_t<B, T const, T>;

template <typename T>
struct kind_of
		: std::integral_constant<
					kind,
					!T::IsRowMajor
							? (T::ColsAtCompileTime == 1 ? kind::colvec : kind::colmat)
							: (T::RowsAtCompileTime == 1 ? kind::rowvec : kind::rowmat)> {};

} // namespace internal

template <typename T, kind K>
using matrix = typename internal::to_matrix<T, K>::type;

template <typename T, kind K>
using view = Eigen::Map<
		internal::add_const_if<
				std::is_const<T>::value,
				typename internal::to_matrix<meta::uncvref_t<T>, K>::type>,
		Eigen::Unaligned,
		Eigen::OuterStride<-1>>;

static_assert(internal::kind_of<Eigen::VectorXd>::value == kind::colvec, "");
static_assert(internal::kind_of<Eigen::MatrixXd>::value == kind::colmat, "");

namespace internal {
template <template <typename, kind> class Tpl, kind K>
struct dyn_cast_impl;

template <kind K>
struct dyn_cast_impl<matrix, K> {
	template <typename T>
	static auto apply(T& mat) -> matrix<typename T::Scalar, K> {
		if (K == kind::colvec) {
			VEG_DEBUG_ASSERT(mat.cols() == 1);
		}
		return mat;
	}
};

template <>
struct dyn_cast_impl<view, kind::colvec> {
	template <typename T>
	static auto apply(T& mat)
			-> view<std::remove_pointer_t<decltype(mat.data())>, kind::colvec> {
		static_assert(!T::IsRowMajor, "");

		VEG_DEBUG_ASSERT_ALL_OF((mat.innerStride() == 1), (mat.cols() == 1));
		return {mat.data(), mat.rows(), 1, mat.outerStride()};
	}
};

template <>
struct dyn_cast_impl<view, kind::rowvec> {
	template <typename T>
	static auto apply(T& mat)
			-> view<std::remove_pointer_t<decltype(mat.data())>, kind::rowvec> {
		static_assert(T::IsRowMajor, "");

		VEG_DEBUG_ASSERT_ALL_OF((mat.innerStride() == 1), (mat.rows() == 1));
		return {mat.data(), 1, mat.cols(), mat.outerStride()};
	}
};

template <>
struct dyn_cast_impl<view, kind::colmat> {
	template <typename T>
	static auto apply(T& mat)
			-> view<std::remove_pointer_t<decltype(mat.data())>, kind::colmat> {
		static_assert(!T::IsRowMajor, "");

		VEG_DEBUG_ASSERT(mat.innerStride() == 1);
		return {mat.data(), mat.rows(), mat.cols(), mat.outerStride()};
	}
};

template <>
struct dyn_cast_impl<view, kind::rowmat> {
	template <typename T>
	static auto apply(T& mat)
			-> view<std::remove_pointer_t<decltype(mat.data())>, kind::rowmat> {
		static_assert(T::IsRowMajor, "");

		VEG_DEBUG_ASSERT(mat.innerStride() == 1);
		return {mat.data(), mat.rows(), mat.cols(), mat.outerStride()};
	}
};

template <typename T>
auto element_ptr_impl(
		T const* data,
		i64 i,
		i64 j,
		i64 outer_stride,
		i64 inner_stride,
		bool row_major) noexcept -> T* {
	if (row_major) {
		return const_cast<T*>(data) + (i * outer_stride + j * inner_stride);
	} else {
		return const_cast<T*>(data) + (j * outer_stride + i * inner_stride);
	}
}

template <typename T>
auto element_ptr(T const& mat, i64 i, i64 j) -> typename T::Scalar* {
	return (
			element_ptr_impl)(mat.data(), i, j, mat.outerStride(), mat.innerStride(), T::IsRowMajor);
}

inline auto aliases_impl(
		i64 size,
		void const* ptr1_,
		i64 inner_stride1,
		i64 outer_stride1,
		i64 inner_size1,
		i64 outer_size1,
		void const* ptr2_,
		i64 inner_stride2,
		i64 outer_stride2,
		i64 inner_size2,
		i64 outer_size2) -> bool {

	if (inner_size1 == 0 || //
	    outer_size1 == 0 || //
	    inner_size2 == 0 || //
	    outer_size2 == 0) {
		return false;
	}

	using std::uintptr_t;
	auto begin1 = uintptr_t(ptr1_);
	auto begin2 = uintptr_t(ptr2_);

	auto before_end1 = uintptr_t(
			static_cast<char const*>(ptr1_) +
			size * (inner_stride1 * (inner_size1 - 1) +
	            outer_stride1 * (outer_size1 - 1)));

	auto before_end2 = uintptr_t(
			static_cast<char const*>(ptr2_) +
			size * (inner_stride2 * (inner_size2 - 1) +
	            outer_stride2 * (outer_size2 - 1)));

	return !(
			((begin1 < begin2) && (before_end1 < begin2)) || //
			((begin2 < begin1) && (before_end2 < begin1)));
}

template <typename T, typename U>
auto aliases_impl_2(T const& t, U const& u) -> bool {
	static_assert(
			std::is_same<typename T::Scalar, typename U::Scalar>::value,
			"no type mixing please v_v");
	return internal::aliases_impl(
			sizeof(typename T::Scalar),
			t.data(),
			t.innerStride(),
			t.outerStride(),
			t.innerSize(),
			t.outerSize(),
			u.data(),
			u.innerStride(),
			u.outerStride(),
			u.innerSize(),
			u.outerSize());
}

constexpr auto any_of(std::initializer_list<bool> lst) -> bool {
	for (bool b : lst) { // NOLINT(readability-use-anyofallof)
		if (b) {
			return true;
		}
	}
	return false;
}

} // namespace internal

template <typename T, kind K>
using view_type_t =
		eigen::view<veg::meta::unptr_t<decltype(VEG_DECLVAL(T &&).data())>, K>;

template <typename T>
using deduce_view_t =
		view_type_t<T, (internal::kind_of<meta::uncvref_t<T>>::value)>;

namespace nb {
struct as_const {
	DDP_TEMPLATE(
			typename T,
			requires(DDP_CONCEPT(eigen::matrix<T>)),
			DDP_NODISCARD HEDLEY_ALWAYS_INLINE auto
			operator(),
			(mat, T const&))
	const noexcept->deduce_view_t<T const&> {
		return internal::dyn_cast_impl<view, internal::kind_of<T>::value>::apply(
				mat);
	}
};
struct as_mut {
	DDP_TEMPLATE(
			typename T,
			requires(DDP_CONCEPT(eigen::matrix<uncvref_t<T>>)),
			DDP_NODISCARD HEDLEY_ALWAYS_INLINE auto
			operator(),
			(mat, T&&))
	const noexcept->deduce_view_t<T> {
		return internal::dyn_cast_impl<
				view,
				internal::kind_of<meta::uncvref_t<T>>::value>::apply(mat);
	}
};

struct split_at_row {
	DDP_TEMPLATE(
			typename T,
			requires(DDP_CONCEPT(eigen::matrix<uncvref_t<T>>)),
			auto
			operator(),
			(mat, T&&),
			(row, i64))
	const noexcept->Tuple<deduce_view_t<T>, deduce_view_t<T>> {

		VEG_DEBUG_ASSERT_ALL_OF((row >= 0), (row <= mat.rows()));
		return {
				direct,
				deduce_view_t<T>{mat.data(), row, mat.cols(), mat.outerStride()},
				deduce_view_t<T>{
						mem::addressof(mat.coeffRef(row, 0)),
						mat.rows() - row,
						mat.cols(),
						mat.outerStride()},
		};
	}
};
struct split_at_col {
	DDP_TEMPLATE(
			typename T,
			requires(DDP_CONCEPT(eigen::matrix<uncvref_t<T>>)),
			auto
			operator(),
			(mat, T&&),
			(col, i64))
	const noexcept->Tuple<deduce_view_t<T>, deduce_view_t<T>> {

		VEG_DEBUG_ASSERT_ALL_OF((col >= 0), (col <= mat.cols()));
		return {
				direct,
				deduce_view_t<T>{mat.data(), mat.rows(), col, mat.outerStride()},
				deduce_view_t<T>{
						mem::addressof(mat.coeffRef(0, col)),
						mat.rows(),
						mat.cols() - col,
						mat.outerStride()},
		};
	}
};
struct split_at {
	DDP_TEMPLATE(
			typename T,
			requires(DDP_CONCEPT(eigen::matrix<uncvref_t<T>>)),
			auto
			operator(),
			(mat, T&&),
			(row, i64),
			(col, i64))
	const noexcept->Tuple<
			deduce_view_t<T>,
			deduce_view_t<T>,
			deduce_view_t<T>,
			deduce_view_t<T>> {

		VEG_DEBUG_ASSERT_ALL_OF( //
				(row >= 0),
				(row <= mat.rows()),
				(col >= 0),
				(col <= mat.cols()));
		auto const os = mat.outerStride();
		return {
				direct,
				deduce_view_t<T>{mat.data(), row, col, os},
				deduce_view_t<T>{
						mem::addressof(mat.coeffRef(0, col)), row, mat.cols() - col, os},
				deduce_view_t<T>{
						mem::addressof(mat.coeffRef(row, 0)), mat.rows() - row, col, os},
				deduce_view_t<T>{
						mem::addressof(mat.coeffRef(row, col)),
						mat.rows() - row,
						mat.cols() - col,
						os},
		};
	}
};
struct aliases {
	DDP_TEMPLATE(
			(typename T, typename... Ts),
			requires(
					DDP_CONCEPT(eigen::matrix<T>) &&
					VEG_ALL_OF(DDP_CONCEPT(eigen::matrix<Ts>))),
			auto
			operator(),
			(t, T const&),
			(... ts, Ts const&))
	const noexcept->bool {
		return (internal::any_of)({(internal::aliases_impl_2)(t, ts)...});
	}
};
struct slice_to_vec {
	DDP_TEMPLATE(
			typename T,
			requires(
					DDP_CONCEPT(aux::has_data<T>) && DDP_CONCEPT(aux::has_size<T>) &&
					VEG_CONCEPT(pointer<member_data_t<T&>>) &&
					VEG_CONCEPT(constructible<i64, member_size_t<T&>>)),
			auto
			operator(),
			(s, T&),
			(/*nrows*/ = 0, i64),
			(/*ncols*/ = 0, i64))
	const noexcept->view<meta::unptr_t<member_data_t<T&>>, kind::colvec> {
		auto size = narrow<i64>(s.size());
		return {s.data(), size, 1, size};
	}
};
struct slice_to_mat {
	DDP_TEMPLATE(
			typename T,
			requires(
					DDP_CONCEPT(aux::has_data<T>) && DDP_CONCEPT(aux::has_size<T>) &&
					VEG_CONCEPT(pointer<member_data_t<T&>>) &&
					VEG_CONCEPT(constructible<i64, member_size_t<T&>>)),
			auto
			operator(),
			(s, T&),
			(nrows, i64),
			(ncols, i64))
	const noexcept->view<meta::unptr_t<member_data_t<T&>>, kind::colmat> {
		VEG_DEBUG_ASSERT(nrows * ncols == s.size());
		return {s.data(), nrows, ncols, nrows};
	}
};
} // namespace nb
DDP_NIEBLOID(as_const);
DDP_NIEBLOID(as_mut);
DDP_NIEBLOID(split_at_row);
DDP_NIEBLOID(split_at_col);
DDP_NIEBLOID(split_at);
DDP_NIEBLOID(aliases);
DDP_NIEBLOID(slice_to_vec);
DDP_NIEBLOID(slice_to_mat);

namespace nb {
struct assign {
	DDP_TEMPLATE(
			(typename Out, typename In),
			requires(
					DDP_CONCEPT(eigen::matrix<uncvref_t<Out>>) &&
					VEG_CONCEPT(constructible<uncvref_t<Out>&, Out&>) &&
					DDP_CONCEPT(eigen::matrix<In>) && //
					VEG_CONCEPT(same<scalar_t<In>, scalar_t<uncvref_t<Out>>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<In>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<cols_t<uncvref_t<Out>>, cols_t<In>>)),
			void
			operator(),
			(out, Out&&),
			(in, In const&))
	const { as_mut{}(out) = as_const{}(in); }
};
struct add_to {
	DDP_TEMPLATE(
			(typename Out, typename Lhs, typename Rhs),
			requires(
					DDP_CONCEPT(eigen::matrix<uncvref_t<Out>>) &&
					VEG_CONCEPT(constructible<uncvref_t<Out>&, Out&>) &&
					DDP_CONCEPT(eigen::matrix<Lhs>) && //
					DDP_CONCEPT(eigen::matrix<Rhs>) &&
					VEG_CONCEPT(same<scalar_t<Lhs>, scalar_t<uncvref_t<Out>>>) &&
					VEG_CONCEPT(same<scalar_t<Rhs>, scalar_t<uncvref_t<Out>>>) &&

					DDP_CONCEPT(eigen::aux::maybe_same<rows_t<Lhs>, rows_t<Rhs>>) &&
					DDP_CONCEPT(eigen::aux::maybe_same<cols_t<Lhs>, cols_t<Rhs>>) &&

					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<Lhs>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<Lhs>>) &&

					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<Rhs>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<Rhs>>)),
			void
			operator(),
			(out, Out&&),
			(lhs, Lhs const&),
			(rhs, Rhs const&))
	const { as_mut{}(out) = as_const{}(lhs).operator+(as_const{}(rhs)); }
};
struct sub_to {
	DDP_TEMPLATE(
			(typename Out, typename Lhs, typename Rhs),
			requires(
					DDP_CONCEPT(eigen::matrix<uncvref_t<Out>>) &&
					VEG_CONCEPT(constructible<uncvref_t<Out>&, Out&>) &&
					DDP_CONCEPT(eigen::matrix<Lhs>) && //
					DDP_CONCEPT(eigen::matrix<Rhs>) &&
					VEG_CONCEPT(same<scalar_t<Lhs>, scalar_t<uncvref_t<Out>>>) &&
					VEG_CONCEPT(same<scalar_t<Rhs>, scalar_t<uncvref_t<Out>>>) &&

					DDP_CONCEPT(eigen::aux::maybe_same<rows_t<Lhs>, rows_t<Rhs>>) &&
					DDP_CONCEPT(eigen::aux::maybe_same<cols_t<Lhs>, rows_t<Rhs>>) &&

					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<Lhs>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<cols_t<uncvref_t<Out>>, rows_t<Lhs>>) &&

					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<Rhs>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<cols_t<uncvref_t<Out>>, rows_t<Rhs>>)),
			void
			operator(),
			(out, Out&&),
			(lhs, Lhs const&),
			(rhs, Rhs const&))
	const { as_mut{}(out) = as_const{}(lhs).operator-(as_const{}(rhs)); }
};
struct mul_scalar_to {
	DDP_TEMPLATE(
			(typename Out, typename In),
			requires(
					DDP_CONCEPT(eigen::matrix<uncvref_t<Out>>) &&
					VEG_CONCEPT(constructible<uncvref_t<Out>&, Out&>) &&
					DDP_CONCEPT(eigen::matrix<In>) &&
					VEG_CONCEPT(same<scalar_t<In>, scalar_t<uncvref_t<Out>>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<In>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<cols_t<uncvref_t<Out>>, rows_t<In>>)),
			void
			operator(),
			(out, Out&&),
			(in, In const&),
			(k = scalar_t<In>(1), scalar_t<In> const&))
	const { as_mut{}(out) = as_const{}(in).operator*(k); }
};
struct mul_scalar_add_to {
	DDP_TEMPLATE(
			(typename Out, typename In),
			requires(
					DDP_CONCEPT(eigen::matrix<uncvref_t<Out>>) &&
					VEG_CONCEPT(constructible<uncvref_t<Out>&, Out&>) &&
					DDP_CONCEPT(eigen::matrix<In>) &&
					VEG_CONCEPT(same<scalar_t<In>, scalar_t<uncvref_t<Out>>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<In>>) &&
					DDP_CONCEPT(
							eigen::aux::maybe_same<cols_t<uncvref_t<Out>>, rows_t<In>>)),
			void
			operator(),
			(out, Out&&),
			(in, In const&),
			(k = scalar_t<In>(1), scalar_t<In> const&))
	const { as_mut{}(out) += as_const{}(in).operator*(k); }
};

struct mul_add_to_noalias {
	DDP_TEMPLATE(
			(typename Out, typename Lhs, typename Rhs),
			requires(
					DDP_CONCEPT(eigen::matrix<uncvref_t<Out>>) &&
					VEG_CONCEPT(constructible<uncvref_t<Out>&, Out&>) &&
					DDP_CONCEPT(eigen::matrix<Lhs>) && //
					DDP_CONCEPT(eigen::matrix<Rhs>) &&
					VEG_CONCEPT(same<scalar_t<Lhs>, scalar_t<uncvref_t<Out>>>) &&
					VEG_CONCEPT(same<scalar_t<Rhs>, scalar_t<uncvref_t<Out>>>) &&

					DDP_CONCEPT(eigen::aux::maybe_same<cols_t<Lhs>, rows_t<Rhs>>) &&

					DDP_CONCEPT(
							eigen::aux::maybe_same<rows_t<uncvref_t<Out>>, rows_t<Lhs>>) &&

					DDP_CONCEPT(
							eigen::aux::maybe_same<cols_t<uncvref_t<Out>>, rows_t<Rhs>>)),
			void
			operator(),
			(out, Out&&),
			(lhs, Lhs const&),
			(rhs, Rhs const&),
			(k = scalar_t<Lhs>(1), scalar_t<Lhs> const&))
	const {
		as_mut{}(out).noalias() +=
				(as_const{}(lhs).operator*(as_const{}(rhs))).operator*(k);
	}
};

struct dot {
	DDP_TEMPLATE(
			(typename Lhs, typename Rhs),
			requires(
					DDP_CONCEPT(eigen::vector<Lhs>) && //
					DDP_CONCEPT(eigen::vector<Rhs>) &&
					VEG_CONCEPT(same<scalar_t<Lhs>, scalar_t<Rhs>>) &&
					DDP_CONCEPT(eigen::aux::maybe_same<rows_t<Lhs>, rows_t<Rhs>>)),
			auto
			operator(),
			(lhs, Lhs const&),
			(rhs, Rhs const&))
	const->scalar_t<Lhs> {
		return (as_const{}(lhs.transpose()).operator*(as_const{}(rhs)))[0];
	}
};
struct add_identity {
	DDP_TEMPLATE(
			typename T,
			requires(
					DDP_CONCEPT(eigen::matrix<uncvref_t<T>>) &&
					VEG_CONCEPT(constructible<uncvref_t<T>&, T&>)),
			void
			operator(),
			(mat, T&&),
			(factor = scalar_t<uncvref_t<T>>(1), scalar_t<uncvref_t<T>> const&))
	const {
		auto _mat = as_mut{}(mat);
		i64 const small_dim = _mat.rows() < _mat.cols() ? _mat.rows() : _mat.cols();
		for (i64 i = 0; i < small_dim; ++i) {
			_mat(i, i) += factor;
		}
	}
};
} // namespace nb
DDP_NIEBLOID(assign);
DDP_NIEBLOID(add_to);
DDP_NIEBLOID(sub_to);
DDP_NIEBLOID(mul_scalar_to);
DDP_NIEBLOID(mul_scalar_add_to);
DDP_NIEBLOID(mul_add_to_noalias);
DDP_NIEBLOID(dot);
DDP_NIEBLOID(add_identity);

template <typename T>
struct heap_matrix<T, kind::colmat> {
private:
	std::vector<T> data;
	i64 rows;
	i64 cols;

public:
	heap_matrix(i64 _rows, i64 _cols)
			: data(narrow<usize>(_rows * _cols)), rows(_rows), cols(_cols) {}

	auto get() const noexcept -> view<T const, kind::colmat> {
		return eigen::slice_to_mat(data, rows, cols);
	}
	auto mut() noexcept -> view<T, kind::colmat> {
		return eigen::slice_to_mat(data, rows, cols);
	}
};

template <typename T>
struct heap_matrix<T, kind::colvec> {
private:
	std::vector<T> data;

public:
	heap_matrix(i64 _rows) : data(narrow<usize>(_rows)) {}

	auto get() const noexcept -> view<T const, kind::colvec> {
		return eigen::slice_to_vec(data);
	}
	auto mut() noexcept -> view<T, kind::colvec> {
		return eigen::slice_to_vec(data);
	}
};

} // namespace eigen

namespace {
constexpr auto const& colmat =
		std::integral_constant<eigen::kind, eigen::kind::colmat>::value;
constexpr auto const& colvec =
		std::integral_constant<eigen::kind, eigen::kind::colvec>::value;
constexpr auto const& rowmat =
		std::integral_constant<eigen::kind, eigen::kind::rowmat>::value;
constexpr auto const& rowvec =
		std::integral_constant<eigen::kind, eigen::kind::rowvec>::value;
} // namespace
using eigen::matrix;
using eigen::view;

} // namespace ddp

template <typename T, typename CharT>
struct fmt::formatter<ddp::eigen::internal::with_formatter<T>, CharT> {
	auto parse(::fmt::basic_format_parse_context<CharT>& pc) {
		return pc.begin();
	}
	template <typename OutIt>
	auto format(
			ddp::eigen::internal::with_formatter<T> val,
			::fmt::basic_format_context<OutIt, CharT>& fc) {
		return val.fmt.format(val.val, fc);
	}
};

template <typename Matrix>
struct fmt::formatter<
		Matrix,
		char,
		veg::meta::enable_if_t<
				std::is_base_of<Eigen::MatrixBase<Matrix>, Matrix>::value>>
		: ddp::eigen::internal::fmt_formatter<typename Matrix::Scalar> {
	using scalar = veg::meta::uncvref_t<typename Matrix::Scalar>;

	template <typename OutIt>
	auto format(Matrix const& mat, ::fmt::basic_format_context<OutIt, char>& fc)
			-> decltype(fc.out()) {
		return ddp::eigen::internal::format_impl<scalar>(
				fc,
				mat.rows(),
				mat.cols(),
				{
						veg::as_ref,
						[&](veg::i64 i, veg::i64 j) noexcept { return mat(i, j); },
				},
				*this);
	}
};

#ifdef NDEBUG
#define __DDP_SET_UNINIT(T, Name) (void)0
#else
#define __DDP_SET_UNINIT(T, Name)                                              \
	(Name).setConstant(::std::numeric_limits<T>::quiet_NaN())
#endif

#define __DDP_TMP_IMPL_MAT(Stack_Func, Stack, Name, Type, Rows, Cols)          \
	auto Name##_storage##__LINE__ =                                              \
			(Stack).Stack_Func(::veg::Tag<Type>{}, (Rows) * (Cols)).unwrap();        \
	auto(Name) = ::ddp::eigen::slice_to_mat(Name##_storage##__LINE__, Rows, Cols)

#define __DDP_TMP_IMPL_VEC(Stack_Func, Stack, Name, Type, Rows)                \
	auto Name##_storage##__LINE__ =                                              \
			(Stack).Stack_Func(::veg::Tag<Type>{}, (Rows)).unwrap();                 \
	auto(Name) = ::ddp::eigen::slice_to_vec(Name##_storage##__LINE__)

#define DDP_TMP_MATRIX_UNINIT(Stack, Name, T, R, C)                            \
	__DDP_TMP_IMPL_MAT(make_new_for_overwrite, Stack, Name, T, R, C);            \
	__DDP_SET_UNINIT(T, Name)
#define DDP_TMP_MATRIX(Stack, Name, T, R, C)                                   \
	__DDP_TMP_IMPL_MAT(make_new, Stack, Name, T, R, C)

#define DDP_TMP_VECTOR_UNINIT(Stack, Name, T, R)                               \
	__DDP_TMP_IMPL_VEC(make_new_for_overwrite, Stack, Name, T, R);               \
	__DDP_SET_UNINIT(T, Name)
#define DDP_TMP_VECTOR(Stack, Name, T, R)                                      \
	__DDP_TMP_IMPL_VEC(make_new, Stack, Name, T, R)

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard DDP_PINOCCHIO_EIGEN_HPP_BDOKZRTGS */
