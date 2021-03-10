#ifndef DDP_PINOCCHIO_MATRIX_SEQ_HPP_SHW3X8TQS
#define DDP_PINOCCHIO_MATRIX_SEQ_HPP_SHW3X8TQS

#include "ddp/internal/eigen.hpp"
#include <memory>
#include <vector>

namespace ddp {
namespace idx {

template <eigen::kind>
struct dims;

template <>
struct dims<colvec> {
  i64 row_;
  auto rows() const -> i64 { return row_; }
  auto cols() const -> i64 { return (void)this, 1; }
};

template <>
struct dims<colmat> {
  i64 row_;
  i64 col_;
  auto rows() const -> i64 { return row_; }
  auto cols() const -> i64 { return col_; }
};
namespace internal {

struct layout {
  i64 begin;
  i64 end;
  i64 max_rows;
  i64 max_cols;
};

template <typename Idx, eigen::kind K>
struct idx_base {
  auto derived() const -> Idx const& { return static_cast<Idx const&>(*this); }
  auto dim_data() const -> veg::slice<dims<K> const> {
    return {derived().dim_data_impl(), index_end() - index_begin()};
  }
  auto offset_data() const -> veg::slice<i64 const> {
    return {derived().offset_data_impl(), index_end() - index_begin() + 1};
  }

  using layout = internal::layout;
  layout self;

  auto index_begin() const -> i64 { return self.begin; }
  auto index_end() const -> i64 { return self.end; }

  auto required_memory() const -> i64 { return offset(index_end()); }
  auto offset(i64 t) const -> i64 {
    VEG_ASSERT_ALL_OF( //
        (t >= index_begin()),
        (t <= index_end()));
    return offset_data()[t - index_begin()];
  }
  auto rows(i64 t) const -> i64 {
    VEG_ASSERT_ALL_OF( //
        (t >= index_begin()),
        (t < index_end()));
    return dim_data()[t - index_begin()].rows();
  }
  auto cols(i64 t) const -> i64 {
    VEG_ASSERT_ALL_OF( //
        (t >= index_begin()),
        (t < index_end()));
    return dim_data()[t - index_begin()].cols();
  }
};

} // namespace internal

template <eigen::kind K>
struct idx;

template <eigen::kind K>
struct idx_view : internal::idx_base<idx_view<K>, K> {
  using base = internal::idx_base<idx_view<K>, K>;
  struct layout {
    dims<K> const* dim_data;
    i64 const* offset_data;
  } self;

  auto dim_data_impl() const { return self.dim_data; }
  auto offset_data_impl() const { return self.offset_data; }

  idx_view(typename base::layout b, layout s) : base{b}, self{s} {}

  auto as_view() const -> idx_view { return *this; }

  using base::index_begin;
  using base::index_end;
  using base::required_memory;
  using base::offset;
  using base::rows;
  using base::cols;
};

template <eigen::kind K>
struct idx : internal::idx_base<idx<K>, K> {
  using base = internal::idx_base<idx<K>, K>;

  auto dim_data_impl() const { return self.dim_data.data(); }
  auto offset_data_impl() const { return self.offset_data.data(); }

  friend struct internal::idx_base<idx_view<K>, K>;
  struct layout {
    std::vector<dims<K>> dim_data;
    std::vector<i64> offset_data;
  } self;

  idx(typename base::layout b, layout s) : base{b}, self{VEG_FWD(s)} {}

  template <typename Fn>
  static auto make(i64 begin, i64 end, Fn dim_fn) -> idx {
    typename base::layout b{begin, end, 0, 0};
    layout self{{}, {}};

    constexpr auto to_usize = veg::narrow<usize>;
    self.offset_data.resize(to_usize(b.end - b.begin + 1));
    self.dim_data.resize(to_usize(b.end - b.begin));
    self.offset_data[0] = 0;
    for (i64 t = begin; t < b.end; ++t) {

      dims<K> rc = dim_fn(t);
      i64 r = rc.rows();
      i64 c = rc.cols();

      VEG_ASSERT_ALL_OF((r >= 0), (c >= 0));

      self.dim_data[to_usize(t - begin)] = rc;
      self.offset_data[to_usize(t - begin + 1)] =
          self.offset_data[to_usize(t - begin)] + r * c;

      b.max_rows = (r > b.max_rows) ? r : b.max_rows;
      b.max_cols = (c > b.max_cols) ? c : b.max_cols;
    }
    return {b, VEG_FWD(self)};
  }

public:
  VEG_TEMPLATE(
      typename Fn,
      requires __VEG_SAME_AS(
          dims<K>, (meta::detected_t<meta::invoke_result_t, Fn&, i64>)),
      idx,
      (begin, i64),
      (end, i64),
      (dims, Fn))
      : idx{make(begin, end, VEG_FWD(dims))} {}

  auto as_view() const -> idx_view<K> {
    return {base::self, {self.dim_data.data(), self.offset_data.data()}};
  }

  auto into_parts() && -> veg::tuple<typename base::layout, layout> {
    return {elems, base::self, self};
  }

  using base::index_begin;
  using base::index_end;
  using base::required_memory;
  using base::offset;
  using base::rows;
  using base::cols;
};

} // namespace idx
namespace internal {

template <typename T, eigen::kind K>
struct mat_seq {
  static_assert(!std::is_const<T>::value, "");
  using const_view = view<T const, K>;
  using mut_view = view<T, K>;

  struct layout {
    std::vector<T> data;
    idx::idx<K> idx;
  } self;

  explicit mat_seq(layout l) : self{VEG_FWD(l)} {}

public:
  explicit mat_seq(idx::idx<K> idx)
      : self{
            std::vector<T>(veg::narrow<usize>(idx.required_memory())),
            VEG_FWD(idx)} {}

  auto index_begin() const -> i64 { return self.idx.index_begin(); }
  auto index_end() const -> i64 { return self.idx.index_end(); }

  auto operator[](i64 t) const -> const_view {
    VEG_ASSERT_ALL_OF( //
        (t >= self.idx.index_begin()),
        (t < self.idx.index_end()));
    return eigen::dyn_cast<view, K>(view<T const, colmat>{
        self.data.data() + (self.idx.offset(t)),
        self.idx.rows(t),
        self.idx.cols(t),
        self.idx.rows(t),
    });
  }
  auto operator[](i64 t) -> mut_view {
    VEG_ASSERT_ALL_OF( //
        (t >= self.idx.index_begin()),
        (t < self.idx.index_end()));
    return eigen::dyn_cast<view, K>(view<T, colmat>{
        self.data.data() + self.idx.offset(t),
        self.idx.rows(t),
        self.idx.cols(t),
        self.idx.rows(t),
    });
  }
};

} // namespace internal
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_MATRIX_SEQ_HPP_SHW3X8TQS */
