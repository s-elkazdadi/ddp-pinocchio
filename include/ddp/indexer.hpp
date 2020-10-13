#ifndef INDEXER_HPP_6NQCBMG5
#define INDEXER_HPP_6NQCBMG5

#include "ddp/detail/utils.hpp"
#include <iterator>

namespace ddp {

enum struct access_e { bidirectional, random };

namespace indexing {

#define DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(Class_Name)                                                           \
  auto operator++(int) noexcept->Class_Name {                                                                          \
    Class_Name cur = *this;                                                                                            \
    ++(*this);                                                                                                         \
    return cur;                                                                                                        \
  }                                                                                                                    \
  auto operator--(int) noexcept->Class_Name {                                                                          \
    Class_Name cur = *this;                                                                                            \
    --(*this);                                                                                                         \
    return cur;                                                                                                        \
  }                                                                                                                    \
  friend auto operator!=(Class_Name a, Class_Name b)->bool { return not(a == b); }                                     \
  static_assert(true, "")
#define DDP_REDUNDANT_RANDOM_ACCESS_ITER_METHODS(Class_Name)                                                           \
  auto operator-=(std::ptrdiff_t n) noexcept->Class_Name& { return (*this) += -n; }                                    \
  auto operator[](std::ptrdiff_t n) const noexcept->reference { return *((*this) + n); }                               \
  friend auto operator+(Class_Name it, std::ptrdiff_t n) noexcept->Class_Name { return (it += n); }                    \
  friend auto operator+(std::ptrdiff_t n, Class_Name it) noexcept->Class_Name { return (it += n); }                    \
  friend auto operator-(Class_Name it, std::ptrdiff_t n) noexcept->Class_Name { return (it += -n); }                   \
  friend auto operator<(Class_Name a, Class_Name b) noexcept->bool { return (a - b) < 0; }                             \
  friend auto operator<=(Class_Name a, Class_Name b) noexcept->bool { return (a - b) <= 0; }                           \
  friend auto operator>(Class_Name a, Class_Name b) noexcept->bool { return (a - b) > 0; }                             \
  friend auto operator>=(Class_Name a, Class_Name b) noexcept->bool { return (a - b) >= 0; }                           \
  static_assert(true, "")

template <typename Indexer>
struct indexer_iter_t;

template <typename Indexer_Reference>
struct indexer_reference_traits {
  using indexer_t = typename Indexer_Reference::indexer_t;
};

template <typename Indexer>
struct indexer_reference_traits<Indexer const*> {
  using indexer_t = Indexer;
};

template <typename Indexer_Ref>
struct indexer_proxy_t {
  using indexer_t = typename indexer_reference_traits<Indexer_Ref>::indexer_t;

  Indexer_Ref m_indexer;
  index_t m_current_index;
  index_t m_memory_offset;

  auto rows() const noexcept -> typename indexer_t::row_kind { return m_indexer->rows(m_current_index); }
  auto cols() const noexcept -> typename indexer_t::col_kind { return m_indexer->cols(m_current_index); }

  auto max_rows() const noexcept -> typename indexer_t::max_row_kind { return m_indexer->max_rows(); }
  auto max_cols() const noexcept -> typename indexer_t::max_col_kind { return m_indexer->max_cols(); }

  auto current_index() const noexcept -> index_t { return m_current_index; }
  auto to_forward_iterator() const noexcept -> indexer_iter_t<Indexer_Ref>;
};

template <typename Indexer_Ref>
struct indexer_iter_t {
  using indexer_t = typename indexer_reference_traits<Indexer_Ref>::indexer_t;
  using proxy_t = indexer_proxy_t<Indexer_Ref>;

  using reference = proxy_t;
  static constexpr access_e iter_category = indexer_t::random_access ? access_e::random : access_e::bidirectional;
  static constexpr bool random_access = indexer_t::random_access;

  Indexer_Ref m_indexer;
  index_t m_current_index;
  index_t m_memory_offset;

  indexer_iter_t(Indexer_Ref indexer, index_t current_index, index_t memory_offset) noexcept
      : m_indexer{indexer}, m_current_index{current_index}, m_memory_offset{memory_offset} {
    DDP_ASSERT_MSG_ALL_OF(
        ("", m_current_index <= m_indexer->index_end()),
        ("", m_current_index >= m_indexer->index_begin()),
        ("", m_memory_offset >= 0));
  }

  auto operator++() noexcept -> indexer_iter_t& {
    DDP_ASSERT(m_current_index + 1 <= m_indexer->index_end());
    m_memory_offset += m_indexer->stride(m_current_index);
    ++m_current_index;
    return *this;
  }
  auto operator--() noexcept -> indexer_iter_t& {
    DDP_ASSERT(m_current_index - 1 >= m_indexer->index_begin());
    --m_current_index;
    m_memory_offset -= m_indexer->stride(m_current_index);
    return *this;
  }
  auto operator+=(std::ptrdiff_t n) noexcept -> indexer_iter_t& {
    DDP_ASSERT_MSG_ALL_OF(
        ("", m_current_index + n <= m_indexer->index_end()),
        ("", m_current_index + n >= m_indexer->index_begin()));
    m_current_index += n;
    if (n >= 0) {
      m_memory_offset += m_indexer->stride_n(m_current_index, n);
    } else {
      m_memory_offset -= m_indexer->stride_n(m_current_index, -n);
    }
    return *this;
  }
  friend auto operator==(indexer_iter_t a, indexer_iter_t b) noexcept -> bool {
    DDP_ASSERT((a.m_indexer == b.m_indexer));
    return a.m_current_index == b.m_current_index;
  }
  friend auto operator-(indexer_iter_t b, indexer_iter_t a) -> std::ptrdiff_t {
    DDP_ASSERT((a.m_indexer == b.m_indexer));
    return b.m_current_index - a.m_current_index;
  }
  auto operator*() const noexcept -> proxy_t {
    DDP_ASSERT(m_current_index != m_indexer->index_end());
    return {m_indexer, m_current_index, m_memory_offset};
  }

  DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(indexer_iter_t);
  DDP_REDUNDANT_RANDOM_ACCESS_ITER_METHODS(indexer_iter_t);
};

namespace _ {
template <typename Idx>
struct begin_end_impl {
  using iterator = indexer_iter_t<Idx const*>;
  using proxy_t = indexer_proxy_t<Idx const*>;

  static auto begin(Idx const& idx) noexcept -> iterator { return {&idx, idx.index_begin(), 0}; }
  static auto end(Idx const& idx) noexcept -> iterator { return {&idx, idx.index_end(), idx.required_memory()}; }
};
} // namespace _

// clang-format off
template <typename Idx, typename Impl = _::begin_end_impl<Idx>>
auto begin (Idx const& idx) noexcept -> typename Impl::iterator         { return Impl::begin(idx); }
template <typename Idx, typename Impl = _::begin_end_impl<Idx>>
auto end   (Idx const& idx) noexcept -> typename Impl::iterator         { return Impl::end(idx); }
// clang-format on

template <typename Indexer_Ref>
auto indexer_proxy_t<Indexer_Ref>::to_forward_iterator() const noexcept -> indexer_iter_t<Indexer_Ref> {
  return {m_indexer, m_current_index, m_memory_offset};
}

template <typename Idx_L, typename Idx_R>
struct row_concat_indexer_t {
  static_assert(std::is_same<typename Idx_L::col_kind, typename Idx_R::col_kind>::value, "");
  static_assert(std::is_same<typename Idx_L::max_col_kind, typename Idx_R::max_col_kind>::value, "");

  using row_kind = detail::add_t<typename Idx_L::row_kind, typename Idx_R::row_kind>;
  using col_kind = typename Idx_L::col_kind;

  using max_row_kind = detail::add_t<typename Idx_L::max_row_kind, typename Idx_R::max_row_kind>;
  using max_col_kind = typename Idx_L::max_col_kind;

  static constexpr bool random_access = Idx_L::random_access and Idx_R::random_access;

  Idx_L m_idx_l;
  Idx_R m_idx_r;

  row_concat_indexer_t(Idx_L l, Idx_R r) : m_idx_l{DDP_MOVE(l)}, m_idx_r{DDP_MOVE(r)} {
    DDP_ASSERT_MSG_ALL_OF(
        ("", m_idx_l.index_begin() == m_idx_r.index_begin()),
        ("", m_idx_l.index_end() == m_idx_r.index_end()));
    for (index_t t = this->index_begin(); t < this->index_end(); ++t) {
      detail::assert_eq(m_idx_l.cols(t), m_idx_r.cols(t));
    }
  }

  auto clone() const noexcept(noexcept(m_idx_l.clone()) and noexcept(m_idx_r.clone())) -> row_concat_indexer_t {
    return {m_idx_l.clone(), m_idx_r.clone()};
  }

  auto index_begin() const noexcept -> index_t { return m_idx_l.index_begin(); }
  auto index_end() const noexcept -> index_t { return m_idx_l.index_end(); }

  auto required_memory() const noexcept -> index_t { return m_idx_l.required_memory() + m_idx_r.required_memory(); };
  auto stride(index_t t) const noexcept -> index_t { return m_idx_l.stride(t) + m_idx_r.stride(t); }
  auto stride_n(index_t t) const noexcept -> index_t { return m_idx_l.stride_n(t) + m_idx_r.stride_n(t); }

  auto rows(index_t t) const noexcept -> row_kind { return m_idx_l.rows(t) + m_idx_r.rows(t); }
  auto cols(index_t t) const noexcept -> col_kind { return m_idx_l.cols(t); }

  auto max_rows() const noexcept -> max_row_kind { return m_idx_l.max_rows() + m_idx_r.max_rows(); }
  auto max_cols() const noexcept -> max_col_kind { return m_idx_l.max_cols(); }

  using iterator = typename _::begin_end_impl<row_concat_indexer_t>::iterator;
  using proxy_t = typename _::begin_end_impl<row_concat_indexer_t>::proxy_t;
};

template <typename Idx_L, typename Idx_R>
struct outer_product_indexer_t {
  static_assert(detail::check_eq<typename Idx_L::col_kind, fix_index<1>>::value == detail::yes, "");
  static_assert(detail::check_eq<typename Idx_R::col_kind, fix_index<1>>::value == detail::yes, "");

  using row_kind = typename Idx_L::row_kind;
  using col_kind = typename Idx_R::row_kind;

  using max_row_kind = typename Idx_L::max_row_kind;
  using max_col_kind = typename Idx_R::max_row_kind;

  static constexpr bool random_access = false;

  Idx_L m_idx_l;
  Idx_R m_idx_r;
  index_t m_required_memory;

private:
  outer_product_indexer_t(Idx_L l, Idx_R r, index_t required_memory) noexcept
      : m_idx_l{DDP_MOVE(l)}, m_idx_r{DDP_MOVE(r)}, m_required_memory{required_memory} {}

public:
  outer_product_indexer_t(Idx_L l, Idx_R r) noexcept
      : m_idx_l{DDP_MOVE(l)}, m_idx_r{DDP_MOVE(r)}, m_required_memory{0} {
    DDP_ASSERT_MSG_ALL_OF(
        ("", m_idx_l.index_begin() == m_idx_r.index_begin()),
        ("", m_idx_l.index_end() == m_idx_r.index_end()));

    for (index_t t = this->index_begin(); t < this->index_end(); ++t) {
      m_required_memory += this->stride(t);
    }
  }

  auto clone() const noexcept(noexcept(m_idx_l.clone()) and noexcept(m_idx_r.clone())) -> outer_product_indexer_t {
    return {m_idx_l.clone(), m_idx_r.clone(), m_required_memory};
  }

  auto index_begin() const noexcept -> index_t { return m_idx_l.index_begin(); }
  auto index_end() const noexcept -> index_t { return m_idx_l.index_end(); }

  auto required_memory() const noexcept -> index_t { return m_required_memory; };
  auto stride(index_t t) const noexcept -> index_t { return (this->rows(t) * this->cols(t)).value(); }

  auto rows(index_t t) const noexcept -> row_kind { return m_idx_l.rows(t); }
  auto cols(index_t t) const noexcept -> col_kind { return m_idx_r.rows(t); }

  auto max_rows() const noexcept -> max_row_kind { return m_idx_l.max_rows(); }
  auto max_cols() const noexcept -> max_col_kind { return m_idx_r.max_rows(); }

  using iterator = typename _::begin_end_impl<outer_product_indexer_t>::iterator;
  using proxy_t = typename _::begin_end_impl<outer_product_indexer_t>::proxy_t;
};

template <typename Rows, typename Cols = fix_index<1>>
struct regular_indexer_t {
  static_assert(DDP_INDEX_CONCEPT(Rows), "");
  static_assert(DDP_INDEX_CONCEPT(Cols), "");

  using row_kind = Rows;
  using col_kind = Cols;

  using max_row_kind = Rows;
  using max_col_kind = Cols;

  static constexpr bool random_access = true;

  row_kind m_rows;
  col_kind m_cols;
  index_t m_begin;
  index_t m_end;

  regular_indexer_t(index_t begin, index_t end, row_kind rows, col_kind cols) noexcept
      : m_rows{rows}, m_cols{cols}, m_begin{begin}, m_end{end} {
    DDP_ASSERT(begin < end);
  }

  auto clone() const noexcept -> regular_indexer_t { return *this; }

  auto index_begin() const noexcept -> index_t { return m_begin; }
  auto index_end() const noexcept -> index_t { return m_end; }

  auto required_memory() const noexcept -> index_t { return (m_rows * m_cols).value() * (m_end - m_begin); };
  auto stride(index_t) const noexcept -> index_t { return (m_rows * m_cols).value(); };
  auto stride_n(index_t, index_t n) const noexcept -> index_t { return n * (m_rows * m_cols).value(); };

  auto rows(index_t) const noexcept -> row_kind { return m_rows; };
  auto cols(index_t) const noexcept -> col_kind { return m_cols; };

  auto max_rows() const noexcept -> max_row_kind { return m_rows; };
  auto max_cols() const noexcept -> max_col_kind { return m_cols; };

  using iterator = typename _::begin_end_impl<regular_indexer_t>::iterator;
  using proxy_t = typename _::begin_end_impl<regular_indexer_t>::proxy_t;
};

template <typename Idx>
struct shift_time_idx_t {
  using row_kind = typename Idx::row_kind;
  using col_kind = typename Idx::col_kind;
  using max_row_kind = typename Idx::max_row_kind;
  using max_col_kind = typename Idx::max_col_kind;

  static constexpr bool random_access = Idx::random_access;
  Idx m_idx;
  index_t m_dt;

  auto clone() const noexcept(noexcept(m_idx.clone())) -> shift_time_idx_t { return {m_idx.clone(), m_dt}; }
  auto index_begin() const noexcept -> index_t { return m_idx.index_begin() - m_dt; }
  auto index_end() const noexcept -> index_t { return m_idx.index_end() - m_dt; }

  auto required_memory() const noexcept -> index_t { return m_idx.required_memory(); }
  auto stride(index_t t) const noexcept -> index_t { return m_idx.stride(t + m_dt); }
  auto stride_n(index_t t, index_t n) const noexcept -> index_t { return m_idx.stride_n(t + m_dt, n); }

  auto rows(index_t t) const noexcept -> row_kind { return m_idx.rows(t + m_dt); }
  auto cols(index_t t) const noexcept -> col_kind { return m_idx.cols(t + m_dt); }

  auto max_rows() const noexcept -> max_row_kind { return m_idx.max_rows(); }
  auto max_cols() const noexcept -> max_col_kind { return m_idx.max_cols(); }

  using iterator = typename _::begin_end_impl<shift_time_idx_t>::iterator;
  using proxy_t = typename _::begin_end_impl<shift_time_idx_t>::proxy_t;
};

inline auto clamp(index_t x, index_t const low, index_t const high) -> index_t {
  return (x < low) //
             ? low
             : ((x > high) //
                    ? high
                    : x);
}

template <typename Indexer>
struct range_row_filter_t {
  static_assert(detail::check_eq<typename Indexer::col_kind, fix_index<1>>::value == detail::yes, "");
  static_assert(detail::check_eq<typename Indexer::max_col_kind, fix_index<1>>::value == detail::yes, "");

  using row_kind = dyn_index;
  using col_kind = fix_index<1>;

  using max_row_kind = typename Indexer::max_row_kind;
  using max_col_kind = fix_index<1>;

  static constexpr bool random_access = Indexer::random_access;

  Indexer m_idx;
  index_t m_range_begin;
  index_t m_range_end;
  index_t m_required_memory;

private:
  range_row_filter_t(Indexer idx, index_t range_begin, index_t range_end, index_t required_memory) noexcept
      : m_idx{DDP_MOVE(idx)}, m_range_begin{range_begin}, m_range_end{range_end}, m_required_memory{required_memory} {}

public:
  range_row_filter_t(Indexer idx, index_t range_begin, index_t range_end) noexcept
      : m_idx{DDP_MOVE(idx)}, m_range_begin{range_begin}, m_range_end{range_end}, m_required_memory{0} {
    for (index_t t = std::max(this->index_begin(), m_range_begin); t < std::min(this->index_end(), m_range_end); ++t) {
      m_required_memory += this->stride(t);
    }
  }

  auto clone() const noexcept(noexcept(m_idx.clone())) -> range_row_filter_t {
    return {m_idx.clone(), m_range_begin, m_range_end, m_required_memory};
  }

  auto index_begin() const noexcept -> index_t { return m_idx.index_begin(); }
  auto index_end() const noexcept -> index_t { return m_idx.index_end(); }

  auto stride(index_t t) const -> index_t { return (rows(t) * cols(t)).value(); }
  auto stride_n(index_t t, index_t n) const noexcept -> index_t {
    DDP_ASSERT(n >= 0);
    index_t begin = clamp(t, m_range_begin, m_range_end);
    index_t end = clamp(t + n, m_range_begin, m_range_end);
    auto result = 0;
    for (index_t i = 0; i < n; ++i) {
      result += stride(t + i);
    }
    DDP_ASSERT(result == m_idx.stride_n(begin, end - begin));
    return m_idx.stride_n(begin, end - begin);
  };

  auto required_memory() const noexcept -> index_t { return m_required_memory; }
  auto unfiltered_rows(index_t t) const noexcept -> typename Indexer::row_kind { return m_idx.rows(t); }

  auto rows(index_t t) const noexcept -> row_kind {
    return (t >= m_range_begin and t < m_range_end) ? dyn_index{unfiltered_rows(t)} : dyn_index{0};
  }
  auto cols(index_t) const noexcept -> col_kind { return {}; }

  auto max_rows() const noexcept -> max_row_kind { return m_idx.max_rows(); };
  auto max_cols() const noexcept -> max_col_kind { return fix_index<1>{}; };

  using iterator = typename _::begin_end_impl<range_row_filter_t>::iterator;
  using proxy_t = typename _::begin_end_impl<range_row_filter_t>::proxy_t;
};

template <typename Indexer>
struct periodic_row_filter_t {
  Indexer m_idx;
  index_t m_period;
  index_t m_first_offset;
  index_t m_required_memory;

  static_assert(detail::check_eq<typename Indexer::col_kind, fix_index<1>>::value == detail::yes, "");
  static_assert(detail::check_eq<typename Indexer::max_col_kind, fix_index<1>>::value == detail::yes, "");

  using row_kind = dyn_index;
  using col_kind = fix_index<1>;

  using max_row_kind = typename Indexer::max_row_kind;
  using max_col_kind = fix_index<1>;

  static constexpr bool random_access = false;

private:
  periodic_row_filter_t(Indexer idx, index_t period, index_t first_offset, index_t required_memory) noexcept
      : m_idx{DDP_MOVE(idx)}, m_period{period}, m_first_offset{first_offset}, m_required_memory{required_memory} {}

public:
  periodic_row_filter_t(Indexer idx, index_t period, index_t first_offset) noexcept
      : m_idx{DDP_MOVE(idx)}, m_period{period}, m_first_offset{first_offset}, m_required_memory{0} {
    DDP_ASSERT_MSG_ALL_OF( //
        ("", period > 0),
        ("", first_offset < period));
    for (index_t t = this->index_begin(); t < this->index_end(); ++t) {
      m_required_memory += this->stride(t);
    }
  }

  auto clone() const noexcept(noexcept(m_idx.clone())) -> periodic_row_filter_t {
    return {m_idx.clone(), m_period, m_first_offset, m_required_memory};
  }

  auto index_begin() const noexcept -> index_t { return m_idx.index_begin(); }
  auto index_end() const noexcept -> index_t { return m_idx.index_end(); }

  auto required_memory() const noexcept -> index_t { return m_required_memory; }
  auto stride(index_t t) const -> index_t { return (rows(t) * cols(t)).value(); }

  auto unfiltered_rows(index_t t) const noexcept -> typename Indexer::row_kind { return m_idx.rows(t); }
  auto rows(index_t t) const noexcept -> row_kind {
    return ((t - m_idx.index_begin()) % m_period == m_first_offset) ? dyn_index{unfiltered_rows(t)} : dyn_index{0};
  }
  auto cols(index_t) const noexcept -> col_kind { return {}; }

  auto max_rows() const noexcept -> max_row_kind { return m_idx.max_rows(); };
  auto max_cols() const noexcept -> max_col_kind { return fix_index<1>{}; };

  using iterator = typename _::begin_end_impl<periodic_row_filter_t>::iterator;
  using proxy_t = typename _::begin_end_impl<periodic_row_filter_t>::proxy_t;
};

template <typename Idx_L, typename Idx_R>
struct outer_prod_result {
  using type = outer_product_indexer_t<Idx_L, Idx_R>;
  static auto construct(Idx_L idx_l, Idx_R idx_r) noexcept -> type { return {DDP_MOVE(idx_l), DDP_MOVE(idx_r)}; }
};

template <typename Rows_L, typename Rows_R>
struct outer_prod_result<regular_indexer_t<Rows_L, fix_index<1>>, regular_indexer_t<Rows_R, fix_index<1>>> {

  using idx_left_t = regular_indexer_t<Rows_L, fix_index<1>>;
  using idx_right_t = regular_indexer_t<Rows_R, fix_index<1>>;
  using type = regular_indexer_t<Rows_L, Rows_R>;

  static auto construct(idx_left_t idx_l, idx_right_t idx_r) noexcept -> type {
    DDP_ASSERT_MSG_ALL_OF(
        ("", idx_l.index_begin() == idx_r.index_begin()),
        ("", idx_l.index_end() == idx_r.index_end()));
    return {idx_l.index_begin(), idx_l.index_end(), idx_l.m_rows, idx_r.m_rows};
  }
};

template <typename Idx_L, typename Idx_R>
auto outer_prod(Idx_L idx_l, Idx_R idx_r) noexcept -> typename outer_prod_result<Idx_L, Idx_R>::type {
  return outer_prod_result<Idx_L, Idx_R>::construct(DDP_MOVE(idx_l), DDP_MOVE(idx_r));
}

template <typename Idx_L, typename Idx_R>
auto row_concat(Idx_L idx_l, Idx_R idx_r) noexcept -> row_concat_indexer_t<Idx_L, Idx_R> {
  return {DDP_MOVE(idx_l), DDP_MOVE(idx_r)};
}

template <typename Idx>
auto periodic_row_filter(Idx idx, index_t period, index_t first_offset) noexcept -> periodic_row_filter_t<Idx> {
  return {DDP_MOVE(idx), period, first_offset};
}

template <typename Rows, typename Cols>
auto mat_regular_indexer(index_t begin, index_t end, Rows rows, Cols cols) noexcept -> regular_indexer_t<Rows, Cols> {
  return {begin, end, rows, cols};
}

template <typename Rows>
auto vec_regular_indexer(index_t begin, index_t end, Rows rows) noexcept -> regular_indexer_t<Rows, fix_index<1>> {
  return {begin, end, rows, fix_index<1>{}};
}

} // namespace indexing
} // namespace ddp

#endif /* end of include guard INDEXER_HPP_6NQCBMG5 */
