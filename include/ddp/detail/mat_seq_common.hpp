#ifndef MAT_SEQ_HELPER_HPP_JASRI0AL
#define MAT_SEQ_HELPER_HPP_JASRI0AL

#include "ddp/detail/mat_seq.hpp"
#include "ddp/detail/tuple.hpp"
#include "ddp/zero.hpp"

namespace ddp {
namespace detail {
namespace matrix_seq {

template <typename Problem, typename Out_Idx>
struct affine_vector_function_seq_t {
  using problem_t = Problem;
  using scalar_t = typename problem_t::scalar_t;

  using x_indexer_t = typename problem_t::state_indexer_t;
  using dx_indexer_t = typename problem_t::dstate_indexer_t;

  using val_indexer_t = Out_Idx;
  using jac_indexer_t = typename indexing::outer_prod_result<Out_Idx, dx_indexer_t>::type;

  using x_mat_seq_t = mat_seq_t<scalar_t, x_indexer_t>;
  using val_mat_seq_t = mat_seq_t<scalar_t, val_indexer_t>;
  using jac_mat_seq_t = mat_seq_t<scalar_t, jac_indexer_t>;

  x_mat_seq_t m_origin;
  val_mat_seq_t m_val_data;
  jac_mat_seq_t m_jac_data;
  problem_t const* m_prob;

private:
  affine_vector_function_seq_t(
      x_mat_seq_t x_data, val_mat_seq_t val_data, jac_mat_seq_t jac_data, problem_t const& prob) noexcept
      : m_origin{DDP_MOVE(x_data)}, m_val_data{DDP_MOVE(val_data)}, m_jac_data{DDP_MOVE(jac_data)}, m_prob{&prob} {};

public:
  affine_vector_function_seq_t(Out_Idx out_idx, problem_t const& prob)
      : m_origin{prob.state_indexer(out_idx.index_begin(), out_idx.index_end())},
        m_val_data{out_idx.clone()},
        m_jac_data{
            indexing::outer_prod( //
                DDP_MOVE(out_idx),
                prob.dstate_indexer( //
                    m_val_data.m_idx.index_begin(),
                    m_val_data.m_idx.index_end() //
                    )                            //
                ),
        },
        m_prob{&prob} {}

  auto clone() const noexcept(false) -> affine_vector_function_seq_t {
    return {m_val_data.clone(), m_jac_data.clone(), m_prob};
  }

  struct x_new_truncated_t {
    x_mat_seq_t const& m_parent;
    friend auto begin(x_new_truncated_t x) -> typename x_mat_seq_t::const_iterator { return begin(x.m_parent); }
    friend auto end(x_new_truncated_t x) -> typename x_mat_seq_t::const_iterator { return --end(x.m_parent); }
  };

  void update_origin(x_mat_seq_t const& new_trajectory) noexcept {
    auto tmp = eigen::make_matrix<scalar_t>(m_prob->dstate_dim());

    auto tmp_jac_ = eigen::make_matrix<scalar_t>(m_prob->dstate_dim(), m_prob->dstate_dim());
    auto tmp_jac = eigen::as_mut_view(tmp_jac_);

    auto tmp_jac2 = eigen::make_matrix<scalar_t>(m_jac_data.m_idx.max_rows(), m_prob->dstate_dim());

    for (auto zipped : ranges::zip(m_origin, m_val_data, m_jac_data, x_new_truncated_t{new_trajectory})) {
      DDP_BIND(auto&&, (origin, val, jac, x_new), zipped);

      auto state_c = eigen::as_const_view(origin.get());
      auto state_new_c = eigen::as_const_view(x_new.get());

      auto tmp_storage = tmp_jac2.template topRows<jac_indexer_t::row_kind::value_at_compile_time>(jac.get().rows());

      // tmp = x_new - origin
      m_prob->difference(eigen::as_mut_view(tmp), state_c, state_new_c);
      m_prob->d_difference_dfinish(eigen::as_mut_view(tmp_jac), state_c, state_new_c);

      // clang-format off
      val.get().noalias()   += jac.get() * tmp;
      tmp_storage.noalias()  = jac.get() * tmp_jac;
      jac.get()              = tmp_storage;
      origin.get()           = x_new.get();
      // clang-format on
    }
  }

  template <bool Is_Const>
  struct proxy_t {
    typename x_mat_seq_t::template proxy_t<Is_Const> m_x_proxy;
    typename val_mat_seq_t::template proxy_t<Is_Const> m_val_proxy;
    typename jac_mat_seq_t::template proxy_t<Is_Const> m_jac_proxy;
    problem_t const* m_prob;

    using val_t = typename val_mat_seq_t::matrix_t;
    using jac_t = typename jac_mat_seq_t::matrix_t;

    auto origin() const -> typename x_mat_seq_t::template proxy_t<Is_Const>::value_type { return m_x_proxy.get(); }
    auto val() const -> typename val_mat_seq_t::template proxy_t<Is_Const>::value_type { return m_val_proxy.get(); }
    auto jac() const -> typename jac_mat_seq_t::template proxy_t<Is_Const>::value_type { return m_jac_proxy.get(); }

    template <typename In>
    void operator()(eigen::view_t<val_t> out, eigen::view_t<In const> input) const noexcept {

      // TODO: dynamic allocation?
      auto tmp1 = eigen::make_matrix<scalar_t>(m_prob->dstate_dim());

      m_prob->difference(eigen::as_mut_view(tmp1), eigen::as_const_view(origin()), eigen::as_const_view(input));

      out = val();
      out.noalias() += jac() * tmp1;
    }
  };

  template <bool Is_Const>
  struct iterator_impl_t {
    typename x_mat_seq_t::template iterator_impl_t<Is_Const> m_x_iter;
    typename val_mat_seq_t::template iterator_impl_t<Is_Const> m_val_iter;
    typename jac_mat_seq_t::template iterator_impl_t<Is_Const> m_jac_iter;
    problem_t const* m_prob;

    using proxy_t = affine_vector_function_seq_t::proxy_t<Is_Const>;

    using value_type = void;
    using reference = proxy_t;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;
    static constexpr access_e iter_category =
        ((val_mat_seq_t::template iterator_impl_t<Is_Const>::iter_category) == access_e::random and
         (jac_mat_seq_t::template iterator_impl_t<Is_Const>::iter_category) == access_e::random)
            ? access_e::random
            : access_e::bidirectional;

    auto operator++() noexcept -> iterator_impl_t& { return (++m_x_iter, ++m_val_iter, ++m_jac_iter, *this); }
    auto operator--() noexcept -> iterator_impl_t& { return (--m_x_iter, --m_val_iter, --m_jac_iter, *this); }
    auto operator+=(std::ptrdiff_t n) noexcept -> iterator_impl_t& {
      return (m_x_iter += n, m_val_iter += n, m_jac_iter += n, *this);
    }
    friend auto operator==(iterator_impl_t a, iterator_impl_t b) noexcept -> bool {
      assert((a.m_val_iter == b.m_val_iter) == (a.m_jac_iter == b.m_jac_iter));
      assert((a.m_val_iter == b.m_val_iter) == (a.m_x_iter == b.m_x_iter));
      return a.m_val_iter == b.m_val_iter;
    }
    auto operator*() const -> proxy_t { return {*m_x_iter, *m_val_iter, *m_jac_iter, m_prob}; }
    DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(iterator_impl_t);
    DDP_REDUNDANT_RANDOM_ACCESS_ITER_METHODS(iterator_impl_t);
  };

  using iterator = iterator_impl_t<false>;
  using const_iterator = iterator_impl_t<true>;

  friend auto begin(affine_vector_function_seq_t& s) -> iterator {
    return {begin(s.m_origin), begin(s.m_val_data), begin(s.m_jac_data), s.m_prob};
  }
  friend auto begin(affine_vector_function_seq_t const& s) -> const_iterator {
    return {begin(s.m_origin), begin(s.m_val_data), begin(s.m_jac_data), s.m_prob};
  }
  friend auto end(affine_vector_function_seq_t& s) -> iterator {
    return {end(s.m_origin), end(s.m_val_data), end(s.m_jac_data), s.m_prob};
  }
  friend auto end(affine_vector_function_seq_t const& s) -> const_iterator {
    return {end(s.m_origin), end(s.m_val_data), end(s.m_jac_data), s.m_prob};
  }

  auto operator[](index_t n) noexcept -> typename iterator::value_type {
    static_assert(iterator::iter_category == access_e::random, "");
    return begin(*this)[n];
  }
  auto operator[](index_t n) const noexcept -> typename const_iterator::value_type {
    static_assert(iterator::iter_category == access_e::random, "");
    return begin(*this)[n];
  }
};

template <typename Problem, typename Out_Idx>
struct constant_vector_function_seq_t {
  using problem_t = Problem;
  using scalar_t = typename problem_t::scalar_t;

  using val_indexer_t = Out_Idx;
  using val_mat_seq_t = mat_seq_t<scalar_t, val_indexer_t>;

  val_mat_seq_t m_val_data;
  problem_t const* m_model;

private:
  constant_vector_function_seq_t(val_mat_seq_t val_data, problem_t const& model) noexcept
      : m_val_data{DDP_MOVE(val_data)}, m_model{&model} {};

public:
  constant_vector_function_seq_t(Out_Idx out_idx, problem_t const& model)
      : m_val_data{DDP_MOVE(out_idx)}, m_model{&model} {}

  auto clone() const noexcept(false) -> constant_vector_function_seq_t { return {m_val_data.clone()}; }

  template <typename T>
  void update_origin(T const&) noexcept {}

  template <bool Is_Const>
  struct proxy_t {
    typename val_mat_seq_t::template proxy_t<Is_Const> m_val_proxy;
    problem_t const* m_model;

    using val_t = typename val_mat_seq_t::matrix_t;

    auto val() const -> typename val_mat_seq_t::template proxy_t<Is_Const>::value_type { return m_val_proxy.get(); }
    auto jac() const -> zero::zero_t { return {}; }

    template <typename In>
    void
    operator()(eigen::view_t<val_t> out, eigen::view_t<In const> input, eigen::view_t<In const> origin) const noexcept {
      (void)input;
      (void)origin;
      out = val();
    }
  };

  template <bool Is_Const>
  struct iterator_impl_t {
    typename val_mat_seq_t::template iterator_impl_t<Is_Const> m_val_iter;
    problem_t const* m_model;

    using proxy_t = constant_vector_function_seq_t::proxy_t<Is_Const>;

    using value_type = void;
    using reference = proxy_t;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;
    static constexpr access_e iter_category = val_mat_seq_t::template iter_category<Is_Const>::iter_category;

    auto operator++() noexcept -> iterator_impl_t& { return (++m_val_iter, *this); }
    auto operator--() noexcept -> iterator_impl_t& { return (--m_val_iter, *this); }
    auto operator+=(std::ptrdiff_t n) noexcept -> iterator_impl_t& { return (m_val_iter += n, *this); }
    friend auto operator==(iterator_impl_t a, iterator_impl_t b) noexcept -> bool {
      return a.m_val_iter == b.m_val_iter;
    }
    auto operator*() const -> proxy_t { return {*m_val_iter, m_model}; }
    DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(iterator_impl_t);
    DDP_REDUNDANT_RANDOM_ACCESS_ITER_METHODS(iterator_impl_t);
  };

  using iterator = iterator_impl_t<false>;
  using const_iterator = iterator_impl_t<true>;

  friend auto begin(constant_vector_function_seq_t& s) -> iterator { return {begin(s.m_val_data), s.m_model}; }
  friend auto begin(constant_vector_function_seq_t const& s) -> const_iterator {
    return {begin(s.m_val_data), s.m_model};
  }
  friend auto end(constant_vector_function_seq_t& s) -> iterator { return {end(s.m_val_data), s.m_model}; }
  friend auto end(constant_vector_function_seq_t const& s) -> const_iterator { return {end(s.m_val_data), s.m_model}; }

  auto operator[](index_t n) noexcept -> typename iterator::value_type {
    static_assert(iterator::iter_category == access_e::random, "");
    return begin(*this)[n];
  }
  auto operator[](index_t n) const noexcept -> typename const_iterator::value_type {
    static_assert(iterator::iter_category == access_e::random, "");
    return begin(*this)[n];
  }
};

} // namespace matrix_seq
} // namespace detail
} // namespace ddp

#endif /* end of include guard MAT_SEQ_HELPER_HPP_JASRI0AL */
