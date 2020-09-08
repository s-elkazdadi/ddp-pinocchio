#ifndef MAT_SEQ_HPP_Y36FNCPW
#define MAT_SEQ_HPP_Y36FNCPW

#include "ddp/indexer.hpp"

namespace ddp {
namespace detail {
namespace matrix_seq {

template <typename Scalar, typename Indexer>
struct mat_seq_t {
  using storage_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using scalar_t = Scalar;
  using indexer_t = Indexer;

  using row_kind = typename indexer_t::row_kind;
  using col_kind = typename indexer_t::col_kind;

  using max_row_kind = typename indexer_t::max_row_kind;
  using max_col_kind = typename indexer_t::max_col_kind;

  indexer_t m_idx;
  storage_t m_data;

  using matrix_t = eigen::matrix_from_idx_t<scalar_t, indexer_t>;
  using const_view_t = eigen::view_t<matrix_t const>;
  using mut_view_t = eigen::view_t<matrix_t>;

  ~mat_seq_t() = default;
  mat_seq_t(mat_seq_t const&) = delete;
  mat_seq_t(mat_seq_t&&) noexcept = default;
  auto operator=(mat_seq_t const&) -> mat_seq_t& = delete;
  auto operator=(mat_seq_t&&) noexcept -> mat_seq_t& = default;
  explicit mat_seq_t(indexer_t idx) noexcept : m_idx{DDP_MOVE(idx)}, m_data{m_idx.required_memory()} {
    static_assert(std::numeric_limits<scalar_t>::is_specialized, "");
    m_data.setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
  }

private:
  mat_seq_t(indexer_t idx, storage_t&& data) noexcept : m_idx{DDP_MOVE(idx)}, m_data{DDP_MOVE(data)} {}

public:
  auto data() const noexcept -> scalar_t const* { return m_data.data(); }
  auto data() noexcept -> scalar_t* { return m_data.data(); }
  auto clone() const noexcept(false) -> mat_seq_t { return {indexer_t{m_idx}, storage_t{m_data}}; }

  template <bool Is_Const>
  struct proxy_t {
    using inner_proxy_t = typename indexer_t::proxy_t;
    using data_ptr_t = DDP_CONDITIONAL(Is_Const, scalar_t const*, scalar_t*);

    using value_type = DDP_CONDITIONAL(Is_Const, const_view_t, mut_view_t);

    inner_proxy_t m_inner_proxy;
    data_ptr_t m_data;

    auto rows() const noexcept -> row_kind { return m_inner_proxy.rows(); }
    auto cols() const noexcept -> col_kind { return m_inner_proxy.cols(); }
    auto offset() const noexcept -> index_t { return m_inner_proxy.m_memory_offset; }

    auto get() const noexcept -> value_type {
      Eigen::Map<matrix_t const> m_{
          m_data + this->offset(),
          this->rows().value(),
          this->cols().value(),
      };
      return {
          m_data + this->offset(),
          m_.rows(),
          m_.cols(),
          m_.outerStride(),
      };
    }

    auto operator*() const noexcept -> value_type { return get(); }
    explicit operator value_type() const { return get(); }

    auto current_index() const noexcept -> index_t { return m_inner_proxy.current_index(); }
    auto as_const() const noexcept -> proxy_t<true> { return {m_inner_proxy, m_data}; }
  };

  template <bool Is_Const>
  struct iterator_impl_t {
    using indexer_iterator_t = typename indexer_t::iterator;
    using data_ptr_t = DDP_CONDITIONAL(Is_Const, scalar_t const*, scalar_t*);
    using proxy_t = mat_seq_t::proxy_t<Is_Const>;

    using difference_type = std::ptrdiff_t;
    using value_type = DDP_CONDITIONAL(Is_Const, const_view_t, mut_view_t);
    using pointer = void;
    using reference = proxy_t;
    using iterator_category = std::input_iterator_tag;
    static constexpr access_e iter_category = indexer_iterator_t::iter_category;

    indexer_iterator_t m_iter;
    data_ptr_t m_data;

    auto operator++() noexcept -> iterator_impl_t& { return (++m_iter, *this); }
    auto operator--() noexcept -> iterator_impl_t& { return (--m_iter, *this); }
    auto operator+=(std::ptrdiff_t n) noexcept -> iterator_impl_t& {
      static_assert(iter_category == access_e::random, "");
      return (m_iter += n, *this);
    }
    friend auto operator==(iterator_impl_t a, iterator_impl_t b) noexcept -> bool { return a.m_iter == b.m_iter; }
    auto operator*() const -> proxy_t { return {*m_iter, m_data}; }
    friend auto operator-(iterator_impl_t a, iterator_impl_t b) -> std::ptrdiff_t { return a.m_iter - b.m_iter; }

    DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(iterator_impl_t);
    DDP_REDUNDANT_RANDOM_ACCESS_ITER_METHODS(iterator_impl_t);
  };

  using iterator = iterator_impl_t<false>;
  using const_iterator = iterator_impl_t<true>;

  friend auto begin(mat_seq_t& m) noexcept -> iterator { return {begin(m.m_idx), m.m_data.data()}; }
  friend auto begin(mat_seq_t const& m) noexcept -> const_iterator { return {begin(m.m_idx), m.m_data.data()}; }
  friend auto end(mat_seq_t& m) noexcept -> iterator { return {end(m.m_idx), m.m_data.data()}; }
  friend auto end(mat_seq_t const& m) noexcept -> const_iterator { return {end(m.m_idx), m.m_data.data()}; }

  auto operator[](index_t n) noexcept -> typename iterator::value_type {
    static_assert(iterator::iter_category == access_e::random, "");
    return begin(*this)[n];
  }
  auto operator[](index_t n) const noexcept -> typename const_iterator::value_type {
    static_assert(iterator::iter_category == access_e::random, "");
    return begin(*this)[n];
  }
};

template <typename Scalar, typename Indexer>
auto mat_seq(Indexer m_indexer) -> matrix_seq::mat_seq_t<Scalar, Indexer> {
  return matrix_seq::mat_seq_t<Scalar, Indexer>{DDP_MOVE(m_indexer)};
}

} // namespace matrix_seq
} // namespace detail
} // namespace ddp

#endif /* end of include guard MAT_SEQ_HPP_Y36FNCPW */
