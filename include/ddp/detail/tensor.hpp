#ifndef TENSOR_HPP_PTZWQRYY
#define TENSOR_HPP_PTZWQRYY

#include "ddp/indexer.hpp"
#include <functional>
#include <memory>

namespace ddp {
namespace indexing {

template <typename Idx_O, typename Idx_L, typename Idx_R>
struct tensor_indexer_t {
  using outdim_kind = typename Idx_O::row_kind;
  using indiml_kind = typename Idx_L::row_kind;
  using indimr_kind = typename Idx_R::row_kind;

  using max_outdim_kind = typename Idx_O::max_row_kind;
  using max_indiml_kind = typename Idx_L::max_row_kind;
  using max_indimr_kind = typename Idx_R::max_row_kind;

  static constexpr bool random_access = false;

  Idx_O m_idx_o;
  Idx_L m_idx_l;
  Idx_R m_idx_r;
  index_t m_required_memory;

private:
  tensor_indexer_t(Idx_O idx_o, Idx_L idx_l, Idx_R idx_r, index_t required_memory) noexcept
      : m_idx_o{DDP_MOVE(idx_o)},
        m_idx_l{DDP_MOVE(idx_l)},
        m_idx_r{DDP_MOVE(idx_r)},
        m_required_memory{required_memory} {}

public:
  tensor_indexer_t(Idx_O idx_o, Idx_L idx_l, Idx_R idx_r) noexcept
      : m_idx_o{DDP_MOVE(idx_o)}, m_idx_l{DDP_MOVE(idx_l)}, m_idx_r{DDP_MOVE(idx_r)}, m_required_memory{0} {

    DDP_ASSERT_MSG_ALL_OF(
        ("", m_idx_o.index_begin() == m_idx_l.index_begin()),
        ("", m_idx_o.index_begin() == m_idx_r.index_begin()),
        ("", m_idx_o.index_end() == m_idx_l.index_end()),
        ("", m_idx_o.index_end() == m_idx_r.index_end()));

    bool debug = false;
    if (m_idx_l.max_rows().value() == 44 and m_idx_r.max_rows().value() == 44 and m_idx_o.max_rows().value() == 44) {
      debug = true;
    }

    for (index_t t = this->index_begin(); t < this->index_end(); ++t) {
      m_required_memory += this->stride(t);
    }
  }

  auto clone() const noexcept(noexcept(m_idx_l.clone()) and noexcept(m_idx_r.clone())) -> tensor_indexer_t {
    return {m_idx_o.clone(), m_idx_l.clone(), m_idx_r.clone(), m_required_memory};
  }

  auto index_begin() const noexcept -> index_t { return m_idx_o.index_begin(); }
  auto index_end() const noexcept -> index_t { return m_idx_o.index_end(); }

  auto required_memory() const noexcept -> index_t { return m_required_memory; };
  auto stride(index_t t) const noexcept -> index_t {
    return (this->outdim(t) * this->indiml(t) * this->indimr(t)).value();
  }

  auto outdim(index_t t) const noexcept -> outdim_kind { return m_idx_o.rows(t); }
  auto indiml(index_t t) const noexcept -> indiml_kind { return m_idx_l.rows(t); }
  auto indimr(index_t t) const noexcept -> indimr_kind { return m_idx_r.rows(t); }

  using iterator = typename _::begin_end_impl<tensor_indexer_t>::iterator;
  using proxy_t = typename _::begin_end_impl<tensor_indexer_t>::proxy_t;
};

template <typename Idx_O, typename Idx_L, typename Idx_R>
auto tensor_indexer(Idx_O idx_o, Idx_L idx_l, Idx_R idx_r) -> tensor_indexer_t<Idx_O, Idx_L, Idx_R> {
  return {
      DDP_MOVE(idx_o),
      DDP_MOVE(idx_l),
      DDP_MOVE(idx_r),
  };
}

template <typename Idx_O, typename Idx_L, typename Idx_R>
struct indexer_proxy_t<tensor_indexer_t<Idx_O, Idx_L, Idx_R> const*> {
  using indexer_t = indexing::tensor_indexer_t<Idx_O, Idx_L, Idx_R>;

  indexer_t const* m_indexer;
  index_t m_current_index;
  index_t m_memory_offset;

  auto outdim() const noexcept -> typename indexer_t::outdim_kind { return m_indexer->outdim(m_current_index); }
  auto indiml() const noexcept -> typename indexer_t::indiml_kind { return m_indexer->indiml(m_current_index); }
  auto indimr() const noexcept -> typename indexer_t::indimr_kind { return m_indexer->indimr(m_current_index); }

  auto current_index() const noexcept -> index_t { return m_current_index; }
  auto to_forward_iterator() const noexcept -> indexer_iter_t<indexer_t const*>;
};

template <typename Idx_O, typename Idx_L, typename Idx_R>
auto indexer_proxy_t<tensor_indexer_t<Idx_O, Idx_L, Idx_R> const*>::to_forward_iterator() const noexcept
    -> indexer_iter_t<indexer_t const*> {
  return {m_indexer, m_current_index, m_memory_offset};
}

} // namespace indexing

namespace tensor {
template <typename L, typename R>
using mul_t = decltype(DDP_DECLVAL(L) * DDP_DECLVAL(R));

// template <typename T, index_t Rows, index_t Cols>
// using matrix_view_t = eigen::view_t<Eigen::Matrix<T, Rows, Cols>>;

template <typename T, typename Rows, typename Cols>
using matrix_mut_view_t = eigen::view_t<Eigen::Matrix<T, Rows::value_at_compile_time, Cols::value_at_compile_time>>;
template <typename T, typename Rows, typename Cols>
using matrix_const_view_t =
    eigen::view_t<Eigen::Matrix<T, Rows::value_at_compile_time, Cols::value_at_compile_time> const>;

template <typename T, typename Rows, typename Cols>
using contiguous_matrix_mut_view_t =
    Eigen::Map<Eigen::Matrix<T, Rows::value_at_compile_time, Cols::value_at_compile_time>>;
template <typename T, typename Rows, typename Cols>
using contiguous_matrix_const_view_t =
    Eigen::Map<Eigen::Matrix<T, Rows::value_at_compile_time, Cols::value_at_compile_time> const>;

template <typename T, typename Out_Dim, typename In_DimL, typename In_DimR>
struct tensor_view_t {
  T* m_data;
  Out_Dim m_outdim;
  In_DimL m_indiml;
  In_DimR m_indimr;

  using value_type = typename std::remove_const<T>::type;

  auto data() const noexcept -> T* { return m_data; }
  auto outdim() const noexcept -> Out_Dim { return m_outdim; }
  auto indiml() const noexcept -> In_DimL { return m_indiml; }
  auto indimr() const noexcept -> In_DimR { return m_indimr; }
  auto operator()(index_t i, index_t j, index_t k) const noexcept -> T& {
    DDP_ASSERT_MSG_ALL_OF( //
        ("", i < m_outdim.value()),
        ("", j < m_indiml.value()),
        ("", k < m_indimr.value()));
    return m_data[i + j * m_outdim.value() + k * (m_outdim * m_indiml).value()];
  }

  void assign(tensor_view_t<value_type const, Out_Dim, In_DimL, In_DimR> other) {
    static_assert(not std::is_const<T>::value, "");
    DDP_ASSERT_MSG_ALL_OF(
        ("", m_outdim == other.m_outdim),
        ("", m_indiml == other.m_indiml),
        ("", m_indimr == other.m_indimr));

    index_t size = (m_outdim * m_indiml * m_indimr).value();
    if (std::is_trivially_copyable<value_type>::value) {
      std::memcpy(m_data, other.m_data, static_cast<index_t>(sizeof(value_type)) * size);
    } else {
      if (std::less<value_type const*>{}(m_data, other.m_data)) {
        for (index_t i = 0; i < size; ++i) {
          m_data[i] = other.m_data[i];
        }
      } else if (std::less<value_type const*>{}(other.m_data, m_data)) {
        for (index_t i = size - 1; i >= 0; --i) {
          m_data[i] = other.m_data[i];
        }
      }
    }
  }
  void set_constant(value_type const& constant) {
    static_assert(not std::is_const<T>::value, "");

    for (index_t i = 0; i < (m_outdim * m_indiml * m_indimr).value(); ++i) {
      m_data[i] = constant;
    }
  }

  void noalias_contract_add_outdim(
      matrix_mut_view_t<value_type, In_DimL, In_DimR> out,
      matrix_const_view_t<value_type, Out_Dim, fix_index<1>> v) const noexcept {

    DDP_ASSERT_MSG_ALL_OF(
        ("", v.rows() == m_outdim.value()),
        ("", out.rows() == m_indiml.value()),
        ("", out.cols() == m_indimr.value()));
    DDP_ASSERT_MSG_ANY_OF( //
        ("non contiguous matrix", out.rows() == 1),
        ("", out.outerStride() == out.rows()));

    contiguous_matrix_const_view_t<value_type, Out_Dim, mul_t<In_DimL, In_DimR>> //
        in_{m_data, m_outdim.value(), (m_indiml * m_indimr).value()};

    contiguous_matrix_mut_view_t<value_type, fix_index<1>, mul_t<In_DimL, In_DimR>> //
        out_{out.data(), 1, (m_indiml * m_indimr).value()};

    out_.noalias() += v.transpose() * in_;
  }

  void noalias_contract_add_indimr(
      matrix_mut_view_t<value_type, Out_Dim, In_DimL> out,
      matrix_const_view_t<value_type, In_DimR, fix_index<1>> v) const noexcept {

    DDP_ASSERT_MSG_ALL_OF(
        ("", v.rows() == m_indimr.value()),
        ("", out.rows() == m_outdim.value()),
        ("", out.cols() == m_indiml.value()),
        ("non contiguous matrix", out.outerStride() == out.rows()));

    matrix_const_view_t<value_type, mul_t<Out_Dim, In_DimL>, In_DimR> //
        in{m_data, (m_outdim * m_indiml).value(), m_indimr.value()};

    matrix_mut_view_t<value_type, mul_t<Out_Dim, In_DimL>, fix_index<1>> //
        out_{out.data(), (m_outdim * m_indiml).value()};

    out_.noalias() += in * v;
  }

  void noalias_contract_add_indiml(
      matrix_mut_view_t<value_type, Out_Dim, In_DimR> out,
      matrix_const_view_t<value_type, In_DimL, fix_index<1>> v) const noexcept {

    DDP_ASSERT_MSG_ALL_OF( //
        ("", v.rows() == m_indiml.value()),
        ("", out.rows() == m_outdim.value()),
        ("", out.cols() == m_indimr.value()));

    for (index_t i = 0; i < m_indimr.value(); ++i) {

      matrix_const_view_t<value_type, Out_Dim, In_DimL> //
          in{m_data + i * (m_outdim * m_indiml).value(), m_outdim.value(), m_indiml.value()};

      out.col(i).noalias() += in * v;
    }
  }

  auto has_nan() const noexcept -> bool {
    using std::isnan;
    for (index_t i = 0; i < (m_outdim * m_indiml * m_indimr).value(); ++i) {
      if (isnan(m_data[i])) {
        return true;
      }
    }
    return false;
  }

  void print() const {
    for (index_t k = 0; k < m_outdim.value(); ++k) {
      for (index_t i = 0; i < m_indiml.value(); ++i) {
        for (index_t j = 0; j < m_indimr.value(); ++j) {
          fmt::print("{:>15}  ", (*this)(k, i, j));
        }
        fmt::print("\n");
      }
      fmt::print("\n");
    }
  }

  auto as_const_view() const noexcept -> tensor_view_t<value_type const, Out_Dim, In_DimL, In_DimR> {
    return {m_data, m_outdim, m_indiml, m_indimr};
  }
  auto as_mut_view() noexcept -> tensor_view_t<value_type, Out_Dim, In_DimL, In_DimR> {
    static_assert(not std::is_const<T>::value, "");
    return {m_data, m_outdim, m_indiml, m_indimr};
  }
  auto as_dynamic() noexcept -> tensor_view_t<T, dyn_index, dyn_index, dyn_index> {
    return {
        m_data,
        dyn_index{m_outdim},
        dyn_index{m_indiml},
        dyn_index{m_indimr},
    };
  }
};

template <
    typename T,
    typename Out_Dim,
    typename In_DimL,
    typename In_DimR,
    bool Stack_Allocated = mul_t<Out_Dim, mul_t<In_DimL, In_DimR>>::known_at_compile_time>
struct tensor_t {
  static_assert(Stack_Allocated, "");
  using size_kind = mul_t<Out_Dim, mul_t<In_DimL, In_DimR>>;

  T m_data[size_kind::value_at_compile_time];
  Out_Dim m_outdim;
  In_DimL m_indiml;
  In_DimR m_indimr;

  using value_type = typename std::remove_const<T>::type;

  auto data() noexcept -> T* { return m_data; }
  auto data() const noexcept -> T const* { return m_data; }
  auto outdim() const noexcept -> Out_Dim { return m_outdim; }
  auto indiml() const noexcept -> In_DimL { return m_indiml; }
  auto indimr() const noexcept -> In_DimR { return m_indimr; }

  auto operator()(index_t i, index_t j, index_t k) noexcept -> T& { return this->as_mut_view()(i, j, k); }
  auto operator()(index_t i, index_t j, index_t k) const noexcept -> T const& { return this->as_const_view()(i, j, k); }

  auto as_const_view() const noexcept -> tensor_view_t<T const, Out_Dim, In_DimL, In_DimR> {
    return {m_data, m_outdim, m_indiml, m_indimr};
  }
  auto as_mut_view() noexcept -> tensor_view_t<T, Out_Dim, In_DimL, In_DimR> {
    return {m_data, m_outdim, m_indiml, m_indimr};
  }
};

template <typename T, typename Out_Dim, typename In_DimL, typename In_DimR>
struct tensor_t<T, Out_Dim, In_DimL, In_DimR, false> {
  using size_kind = mul_t<Out_Dim, mul_t<In_DimL, In_DimR>>;
  static_assert(not size_kind::known_at_compile_time, "");

  Out_Dim m_outdim;
  In_DimL m_indiml;
  In_DimR m_indimr;
  std::unique_ptr<T[]> m_data =
      std::make_unique<T[]>(static_cast<std::size_t>((m_outdim * m_indiml * m_indimr).value()));

  using value_type = typename std::remove_const<T>::type;

  tensor_t(Out_Dim outdim, In_DimL indiml, In_DimR indimr) : m_outdim{outdim}, m_indiml{indiml}, m_indimr{indimr} {}

  auto data() noexcept -> T* { return m_data.get(); }
  auto data() const noexcept -> T const* { return m_data.get(); }
  auto outdim() const noexcept -> Out_Dim { return m_outdim; }
  auto indiml() const noexcept -> In_DimL { return m_indiml; }
  auto indimr() const noexcept -> In_DimR { return m_indimr; }

  auto operator()(index_t i, index_t j, index_t k) noexcept -> T& { return this->as_mut_view()(i, j, k); }
  auto operator()(index_t i, index_t j, index_t k) const noexcept -> T const& { return this->as_const_view()(i, j, k); }

  auto as_const_view() const noexcept -> tensor_view_t<T const, Out_Dim, In_DimL, In_DimR> {
    return {data(), m_outdim, m_indiml, m_indimr};
  }
  auto as_mut_view() noexcept -> tensor_view_t<T, Out_Dim, In_DimL, In_DimR> {
    return {data(), m_outdim, m_indiml, m_indimr};
  }
};
} // namespace tensor

namespace detail {
namespace matrix_seq {

template <typename Scalar, typename Indexer>
struct tensor_seq_t {
  using storage_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using scalar_t = Scalar;
  using indexer_t = Indexer;

  using outdim_kind = typename Indexer::outdim_kind;
  using indiml_kind = typename Indexer::indiml_kind;
  using indimr_kind = typename Indexer::indimr_kind;

  using max_outdim_kind = typename Indexer::max_outdim_kind;
  using max_indiml_kind = typename Indexer::max_indiml_kind;
  using max_indimr_kind = typename Indexer::max_indimr_kind;

  indexer_t m_idx;
  storage_t m_data;

  using const_view_t = tensor::tensor_view_t<scalar_t const, outdim_kind, indiml_kind, indimr_kind>;
  using mut_view_t = tensor::tensor_view_t<scalar_t, outdim_kind, indiml_kind, indimr_kind>;

  ~tensor_seq_t() = default;
  tensor_seq_t(tensor_seq_t const&) = delete;
  tensor_seq_t(tensor_seq_t&&) noexcept = default;
  auto operator=(tensor_seq_t const&) -> tensor_seq_t& = delete;
  auto operator=(tensor_seq_t&&) noexcept -> tensor_seq_t& = default;
  explicit tensor_seq_t(indexer_t idx) noexcept : m_idx{DDP_MOVE(idx)}, m_data{m_idx.required_memory()} {}

private:
  tensor_seq_t(indexer_t idx, storage_t&& data) noexcept : m_idx{DDP_MOVE(idx)}, m_data{DDP_MOVE(data)} {}

public:
  auto clone() noexcept(false) -> tensor_seq_t { return {m_idx, storage_t{m_data}}; }

  template <bool Is_Const>
  struct proxy_t {
    using inner_proxy_t = typename indexer_t::proxy_t;
    using data_ptr_t = DDP_CONDITIONAL(Is_Const, scalar_t const*, scalar_t*);

    using value_type = DDP_CONDITIONAL(Is_Const, const_view_t, mut_view_t);

    inner_proxy_t m_inner_proxy;
    data_ptr_t m_data;

    auto outdim() const noexcept -> outdim_kind { return m_inner_proxy.outdim(); }
    auto indiml() const noexcept -> indiml_kind { return m_inner_proxy.indiml(); }
    auto indimr() const noexcept -> indimr_kind { return m_inner_proxy.indimr(); }
    auto offset() const noexcept -> index_t { return m_inner_proxy.m_memory_offset; }

    auto get() const noexcept -> value_type {
      return {
          m_data + this->offset(),
          this->outdim(),
          this->indiml(),
          this->indimr(),
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
    using proxy_t = tensor_seq_t::proxy_t<Is_Const>;

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
    auto operator+=(std::ptrdiff_t n) noexcept -> iterator_impl_t& { return (m_iter += n, *this); }
    friend auto operator==(iterator_impl_t a, iterator_impl_t b) noexcept -> bool { return a.m_iter == b.m_iter; }
    auto operator*() const -> proxy_t { return {*m_iter, m_data}; }

    DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(iterator_impl_t);
    DDP_REDUNDANT_RANDOM_ACCESS_ITER_METHODS(iterator_impl_t);
  };

  using iterator = iterator_impl_t<false>;
  using const_iterator = iterator_impl_t<true>;

  friend auto begin(tensor_seq_t& s) noexcept -> iterator { return {begin(s.m_idx), s.m_data.data()}; }
  friend auto begin(tensor_seq_t const& s) noexcept -> const_iterator { return {begin(s.m_idx), s.m_data.data()}; }
  friend auto end(tensor_seq_t& s) noexcept -> iterator { return {end(s.m_idx), s.m_data.data()}; }
  friend auto end(tensor_seq_t const& s) noexcept -> const_iterator { return {end(s.m_idx), s.m_data.data()}; }

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
auto tensor_seq(Indexer idx) -> tensor_seq_t<Scalar, Indexer> {
  return tensor_seq_t<Scalar, Indexer>{DDP_MOVE(idx)};
}

} // namespace matrix_seq
} // namespace detail
} // namespace ddp

#endif /* end of include guard TENSOR_HPP_PTZWQRYY */
