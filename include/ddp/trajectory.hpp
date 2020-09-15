#ifndef TRAJECTORY_HPP_OHPWQNFG
#define TRAJECTORY_HPP_OHPWQNFG

#include "ddp/detail/mat_seq.hpp"

namespace ddp {
namespace trajectory {

template <typename Scalar, typename State_Idx, typename Control_Idx>
struct trajectory_t {
  using scalar_t = Scalar;

  using x_indexer_t = State_Idx;
  using u_indexer_t = Control_Idx;

  using x_mat_seq_t = detail::matrix_seq::mat_seq_t<scalar_t, x_indexer_t>;
  using u_mat_seq_t = detail::matrix_seq::mat_seq_t<scalar_t, u_indexer_t>;

  using x_vec_t = typename x_mat_seq_t::matrix_t;
  using u_vec_t = typename u_mat_seq_t::matrix_t;

  x_mat_seq_t m_state_data;
  u_mat_seq_t m_control_data;

private:
  trajectory_t(x_mat_seq_t x_data, u_mat_seq_t u_data)
      : m_state_data(DDP_MOVE(x_data)), m_control_data(DDP_MOVE(u_data)) {
    assert(m_state_data.m_idx.index_begin() == m_control_data.m_idx.index_begin());
    assert(m_state_data.m_idx.index_end() == m_control_data.m_idx.index_end() + 1);
  }

public:
  trajectory_t(x_indexer_t x_idx, u_indexer_t u_idx)
      : trajectory_t(x_mat_seq_t{DDP_MOVE(x_idx)}, u_mat_seq_t{DDP_MOVE(u_idx)}) {}

  // --------------------------------------------------------------------------------

  auto clone() const -> trajectory_t { return trajectory_t{m_state_data.clone(), m_control_data.clone()}; }
  auto horizon() const -> index_t { return m_state_data.idx.n_steps - 1; }

  template <bool IsConst>
  struct proxy_t {
    using x_proxy_t = typename x_mat_seq_t::template proxy_t<IsConst>;
    using u_proxy_t = typename u_mat_seq_t::template proxy_t<IsConst>;
    x_proxy_t m_state_proxy;
    u_proxy_t m_control_proxy;

    auto x() const noexcept -> typename x_proxy_t::value_type { return m_state_proxy.get(); }
    auto u() const noexcept -> typename u_proxy_t::value_type { return m_control_proxy.get(); }

    auto x_next() -> typename x_proxy_t::value_type {
      auto next_idx_proxy = *(++m_state_proxy.m_inner_proxy.to_forward_iterator());
      auto next_matseq_proxy = x_proxy_t{next_idx_proxy, m_state_proxy.m_data};
      return next_matseq_proxy.get();
    }

    auto current_index() const -> index_t { return m_state_proxy.current_index(); }
    auto as_const() const -> proxy_t<true> { return {m_state_proxy.as_const(), m_control_proxy.as_const()}; }
  };

  template <bool Is_Const>
  struct iterator_impl_t {
    using x_iter_t = typename x_mat_seq_t::template iterator_impl_t<Is_Const>;
    using u_iter_t = typename u_mat_seq_t::template iterator_impl_t<Is_Const>;
    x_iter_t x_iter;
    u_iter_t u_iter;

    using proxy_t = trajectory_t::proxy_t<Is_Const>;

    using difference_type = std::ptrdiff_t;
    using value_type = void;
    using pointer = proxy_t*;
    using reference = proxy_t;
    using iterator_category = std::input_iterator_tag;
    static constexpr access_e iter_category =
        ((x_mat_seq_t::template iterator_impl_t<Is_Const>::iter_category) == access_e::random and
         (u_mat_seq_t::template iterator_impl_t<Is_Const>::iter_category) == access_e::random)
            ? access_e::random
            : access_e::bidirectional;

    auto operator==(iterator_impl_t other) const -> bool { return (x_iter == other.x_iter and u_iter == other.u_iter); }
    auto operator++() -> iterator_impl_t& { return (++x_iter, ++u_iter, *this); }
    auto operator--() -> iterator_impl_t& { return (--x_iter, --u_iter, *this); }
    auto operator+=(difference_type n) -> iterator_impl_t& { return (x_iter += n, u_iter += n, *this); }
    auto operator*() const -> proxy_t { return {*x_iter, *u_iter}; }
    DDP_REDUNDANT_BIDIRECTIONAL_ITER_METHODS(iterator_impl_t);
    DDP_REDUNDANT_RANDOM_ACCESS_ITER_METHODS(iterator_impl_t);
  };

  using iterator = iterator_impl_t<false>;
  using const_iterator = iterator_impl_t<true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  friend auto begin(trajectory_t& t) -> iterator { return {begin(t.m_state_data), begin(t.m_control_data)}; }
  friend auto begin(trajectory_t const& t) -> const_iterator {
    return {begin(t.m_state_data), begin(t.m_control_data)};
  }
  friend auto end(trajectory_t& t) -> iterator { return {--end(t.m_state_data), end(t.m_control_data)}; }
  friend auto end(trajectory_t const& t) -> const_iterator { return {--end(t.m_state_data), end(t.m_control_data)}; }

  auto x_0() -> typename proxy_t<false>::x_proxy_t::value_type { return (*begin(*this)).x(); }
  auto x_0() const -> typename proxy_t<true>::x_proxy_t::value_type { return (*begin(*this)).x(); }

  auto x_f() -> typename proxy_t<false>::x_proxy_t::value_type { return (*--end(m_state_data)).get(); }
  auto x_f() const -> typename proxy_t<true>::x_proxy_t::value_type { return (*--end(m_state_data)).get(); }
};

} // namespace trajectory
} // namespace ddp

#endif /* end of include guard TRAJECTORY_HPP_OHPWQNFG */
