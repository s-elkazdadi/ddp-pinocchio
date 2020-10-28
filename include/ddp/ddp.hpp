#ifndef DDP_HPP_FBGSDEO3
#define DDP_HPP_FBGSDEO3

#include "ddp/trajectory.hpp"
#include "ddp/detail/mat_seq_common.hpp"
#include "ddp/detail/tensor.hpp"
#include "ddp/detail/tuple.hpp"

namespace ddp {

enum struct mult_update_attempt_result_e {
  no_update,
  update_success,
  update_failure,
  optimum_attained,
};

enum struct method {
  primal,
  primal_dual_constant_multipliers,
  primal_dual_affine_multipliers,
};

template <typename Scalar>
struct solution_candidate_t {
  eigen::dyn_vec_t<Scalar> primal;
  eigen::dyn_vec_t<Scalar> dual;
  Scalar optimality_score;
};

template <typename Scalar>
struct iterate_t {
  index_t idx;
  solution_candidate_t<Scalar> sol;
  Scalar reg_after_bkw;
  Scalar reg_after_fwd;
  Scalar step;
  Scalar primal_optimality;
  Scalar dual_optimality;
};

template <typename Scalar>
struct solver_parameters_t {
  index_t max_iterations;
  Scalar optimality_stopping_threshold;
  Scalar mu_init;
};

template <typename Scalar>
struct regularization_t {

  void increase_reg() {
    m_factor = std::max(Scalar{1}, m_factor) * m_factor_update;

    if (m_reg == 0) {
      m_reg = m_min_value;
    } else {
      m_reg *= m_factor;
    }
  }

  void decrease_reg() {
    m_factor = std::min(Scalar{1}, m_factor) / m_factor_update;
    m_reg *= m_factor;

    if (m_reg <= m_min_value) {
      m_reg = 0;
    }
  }

  auto operator*() const -> Scalar { return m_reg; }

  Scalar m_reg;
  Scalar m_factor;
  Scalar const m_factor_update;
  Scalar const m_min_value;
};

template <typename Scalar>
struct optimization_state_t {
  regularization_t<Scalar> reg;
  Scalar n;
  Scalar w;
  Scalar initial_opt_obj;
  Scalar initial_opt_constr;
};

template <typename Scalar, typename Control_Idx, typename Eq_Idx, typename State_Idx, typename D_State_Idx>
struct derivative_storage_t {
  using scalar_t = Scalar;
  using control_indexer_t = Control_Idx;
  using eq_indexer_t = Eq_Idx;
  using state_indexer_t = State_Idx;
  using dstate_indexer_t = D_State_Idx;
  using lfx_t = eigen::matrix_t<
      scalar_t,
      fix_index<1>,
      typename dstate_indexer_t::row_kind,
      fix_index<1>,
      typename dstate_indexer_t::max_row_kind,
      Eigen::RowMajor>;

  using lfxx_t = eigen::matrix_t<
      scalar_t,
      typename dstate_indexer_t::row_kind,
      typename dstate_indexer_t::row_kind,
      typename dstate_indexer_t::row_kind,
      typename dstate_indexer_t::max_row_kind,
      Eigen::ColMajor>;

  template <typename L, typename R>
  using prod_t = typename indexing::outer_prod_result<L, R>::type;
  using one_idx_t = indexing::regular_indexer_t<fix_index<1>>;
  template <typename Idx>
  using mat_seq_t = detail::matrix_seq::mat_seq_t<scalar_t, Idx>;
  template <typename O, typename L, typename R>
  using tensor_seq_t = detail::matrix_seq::tensor_seq_t<scalar_t, indexing::tensor_indexer_t<O, L, R>>;

  // clang-format off
    lfx_t                                                                 lfx;
    lfxx_t                                                                lfxx;

    mat_seq_t<prod_t<one_idx_t, dstate_indexer_t>>                        lx;
    mat_seq_t<prod_t<one_idx_t, control_indexer_t>>                       lu;
    mat_seq_t<prod_t<dstate_indexer_t, dstate_indexer_t>>                 lxx;
    mat_seq_t<prod_t<control_indexer_t, dstate_indexer_t>>                lux;
    mat_seq_t<prod_t<control_indexer_t, control_indexer_t>>               luu;

    mat_seq_t       <state_indexer_t>                                     f_val;
    mat_seq_t<prod_t<dstate_indexer_t, dstate_indexer_t>>                 fx;
    mat_seq_t<prod_t<dstate_indexer_t, control_indexer_t>>                fu;
    tensor_seq_t<dstate_indexer_t, dstate_indexer_t, dstate_indexer_t>    fxx;
    tensor_seq_t<dstate_indexer_t, control_indexer_t, dstate_indexer_t>   fux;
    tensor_seq_t<dstate_indexer_t, control_indexer_t, control_indexer_t>  fuu;

    mat_seq_t       <eq_indexer_t>                                        eq_val;
    mat_seq_t<prod_t<eq_indexer_t, dstate_indexer_t>>                     eq_x;
    mat_seq_t<prod_t<eq_indexer_t, control_indexer_t>>                    eq_u;
    tensor_seq_t<eq_indexer_t, dstate_indexer_t, dstate_indexer_t>        eq_xx;
    tensor_seq_t<eq_indexer_t, control_indexer_t, dstate_indexer_t>       eq_ux;
    tensor_seq_t<eq_indexer_t, control_indexer_t, control_indexer_t>      eq_uu;
  // clang-format on

  template <typename T>
  using const_view_t = typename T::const_view_t;
  template <typename T>
  using mut_view_t = typename T::mut_view_t;

#define DDP_HEAD(First, ...) First
#define DDP_TAIL(First, ...) __VA_ARGS__

#define DDP_PROXY_MEMBER(r, _, Elem) const_view_t<DDP_TAIL Elem> DDP_HEAD Elem;
#define DDP_ITERATOR_MEMBER(r, _, Elem) typename DDP_TAIL Elem ::const_iterator DDP_HEAD Elem;

#define DDP_AND_EQUALITY(r, _, Elem) and DDP_HEAD Elem == other.DDP_HEAD Elem
#define DDP_PREFIX(r, Prefix, Elem) Prefix DDP_HEAD Elem;
#define DDP_COMMA_PREFIX(r, Prefix, Elem) , Prefix DDP_HEAD Elem
#define DDP_APPLY_BEGIN_END(r, Prefix_BeginEnd, Elem)                                                                  \
  DDP_TAIL Prefix_BeginEnd(                                                                                            \
      BOOST_PP_CAT(BOOST_PP_IDENTITY(DDP_HEAD Prefix_BeginEnd)(), BOOST_PP_IDENTITY(DDP_HEAD Elem)()))

#define DDP_COMMA_APPLY_BEGIN_END(r, Prefix_BeginEnd, Elem)                                                            \
  , DDP_TAIL Prefix_BeginEnd(BOOST_PP_CAT(DDP_HEAD Prefix_BeginEnd, DDP_HEAD Elem))
#define DDP_IDENTITY(X) X

  // clang-format off
#define DDP_GET_ITER(Class_Name, Name, Ret_Type, Parent_Access_Prefix, Members_Seq)                                    \
    friend auto Name(Class_Name const& s) -> Ret_Type {                                                                \
      return {                                                                                                         \
          DDP_APPLY_BEGIN_END(_,                                                                                       \
              (s.parent.Parent_Access_Prefix, Name),                                                                   \
              BOOST_PP_IDENTITY(BOOST_PP_SEQ_HEAD(Members_Seq))()                                                      \
            )                                                                                                          \
          BOOST_PP_SEQ_FOR_EACH(                                                                                       \
              DDP_COMMA_APPLY_BEGIN_END,                                                                               \
              (s.parent.Parent_Access_Prefix, Name),                                                                   \
              BOOST_PP_SEQ_TAIL(Members_Seq)                                                                           \
            )                                                                                                          \
      };                                                                                                               \
    }                                                                                                                  \
    static_assert(true, "")

#define DDP_DEFINE_ACCESSOR(Accessor_Name, Parent_Access_Prefix, Members_Seq)                                          \
  DDP_DEFINE_ACCESSOR_IMPL(Accessor_Name, Parent_Access_Prefix, BOOST_PP_VARIADIC_SEQ_TO_SEQ(Members_Seq))

#define DDP_DEFINE_ACCESSOR_IMPL(Accessor_Name, Parent_Access_Prefix, Members_Seq)                                     \
    struct Accessor_Name {                                                                                             \
      derivative_storage_t const& parent;                                                                              \
      struct proxy {                                                                                                   \
        BOOST_PP_SEQ_FOR_EACH(DDP_PROXY_MEMBER, _, Members_Seq)                                                        \
      };                                                                                                               \
      struct iterator {                                                                                                \
        using value_type = proxy;                                                                                      \
        using reference = proxy;                                                                                       \
        using pointer = void;                                                                                          \
        using difference_type = std::ptrdiff_t;                                                                        \
        using iterator_category = std::input_iterator_tag;                                                             \
        static constexpr access_e iter_category = access_e::bidirectional;                                             \
        BOOST_PP_SEQ_FOR_EACH(DDP_ITERATOR_MEMBER, _, Members_Seq)                                                     \
        auto operator==(iterator const& other) const -> bool {                                                         \
          return true BOOST_PP_SEQ_FOR_EACH(DDP_AND_EQUALITY, _, Members_Seq);                                         \
        }                                                                                                              \
        auto operator!=(iterator const& other) const -> bool { return not(*this == other); }                           \
                                                                                                                       \
        auto operator++() -> iterator& {                                                                               \
          BOOST_PP_SEQ_FOR_EACH(DDP_PREFIX, ++, Members_Seq)                                                           \
          return *this;                                                                                                \
        }                                                                                                              \
        auto operator--() -> iterator& {                                                                               \
          BOOST_PP_SEQ_FOR_EACH(DDP_PREFIX, --, Members_Seq)                                                           \
          return *this;                                                                                                \
        }                                                                                                              \
        auto operator++(int) -> iterator {                                                                             \
          iterator cur = *this;                                                                                        \
          ++(*this);                                                                                                   \
          return cur;                                                                                                  \
        }                                                                                                              \
        auto operator--(int) -> iterator {                                                                             \
          iterator cur = *this;                                                                                        \
          --(*this);                                                                                                   \
          return cur;                                                                                                  \
        }                                                                                                              \
        auto operator*() const -> proxy {                                                                              \
          return {                                                                                                     \
            **BOOST_PP_IDENTITY(DDP_HEAD BOOST_PP_SEQ_HEAD(Members_Seq))()                                             \
            BOOST_PP_SEQ_FOR_EACH(DDP_COMMA_PREFIX, **, BOOST_PP_SEQ_TAIL(Members_Seq))};                              \
        }                                                                                                              \
      };                                                                                                               \
      using const_iterator = iterator;                                                                                 \
      using reverse_iterator = std::reverse_iterator<iterator>;                                                        \
      DDP_GET_ITER(Accessor_Name, begin, iterator, Parent_Access_Prefix, Members_Seq);                                 \
      DDP_GET_ITER(Accessor_Name, end, iterator, Parent_Access_Prefix, Members_Seq);                                   \
    };
  // clang-format on
  DDP_DEFINE_ACCESSOR(
      costs_t,
      l,
      (x, mat_seq_t<prod_t<one_idx_t, dstate_indexer_t>>)           //
      (u, mat_seq_t<prod_t<one_idx_t, control_indexer_t>>)          //
      (xx, mat_seq_t<prod_t<dstate_indexer_t, dstate_indexer_t>>)   //
      (ux, mat_seq_t<prod_t<control_indexer_t, dstate_indexer_t>>)  //
      (uu, mat_seq_t<prod_t<control_indexer_t, control_indexer_t>>) //
  )
  DDP_DEFINE_ACCESSOR(
      transition_t,
      f,
      (x, mat_seq_t<prod_t<dstate_indexer_t, dstate_indexer_t>>)                 //
      (u, mat_seq_t<prod_t<dstate_indexer_t, control_indexer_t>>)                //
      (xx, tensor_seq_t<dstate_indexer_t, dstate_indexer_t, dstate_indexer_t>)   //
      (ux, tensor_seq_t<dstate_indexer_t, control_indexer_t, dstate_indexer_t>)  //
      (uu, tensor_seq_t<dstate_indexer_t, control_indexer_t, control_indexer_t>) //
  )
  DDP_DEFINE_ACCESSOR(
      eq_t,
      eq_,
      (val, mat_seq_t<eq_indexer_t>)                                         //
      (x, mat_seq_t<prod_t<eq_indexer_t, dstate_indexer_t>>)                 //
      (u, mat_seq_t<prod_t<eq_indexer_t, control_indexer_t>>)                //
      (xx, tensor_seq_t<eq_indexer_t, dstate_indexer_t, dstate_indexer_t>)   //
      (ux, tensor_seq_t<eq_indexer_t, control_indexer_t, dstate_indexer_t>)  //
      (uu, tensor_seq_t<eq_indexer_t, control_indexer_t, control_indexer_t>) //
  )

#undef DDP_DEFINE_ACCESSOR_IMPL
#undef DDP_DEFINE_ACCESSOR
#undef DDP_GET_ITER
#undef DDP_IDENTITY
#undef DDP_COMMA_APPLY_BEGIN_END
#undef DDP_APPLY_BEGIN_END
#undef DDP_COMMA_PREFIX
#undef DDP_PREFIX
#undef DDP_AND_EQUALITY
#undef DDP_ITERATOR_MEMBER
#undef DDP_PROXY_MEMBER
#undef DDP_TAIL
#undef DDP_HEAD

  auto l() const -> costs_t { return {*this}; }
  auto f() const -> transition_t { return {*this}; }
  auto eq() const -> eq_t { return {*this}; }
};

namespace detail {
template <typename... Ts>
struct incomplete;
struct no_multiplier_feedback_t {
  index_t m_begin;
  index_t m_end;

  struct proxy_t {
    static auto val() -> zero::zero_t { return {}; }
    static auto jac() -> zero::zero_t { return {}; }
  };

  struct iterator {
    index_t m_index;
    index_t m_begin;
    index_t m_end;

    using difference_type = std::ptrdiff_t;
    using value_type = void;
    using pointer = proxy_t*;
    using reference = proxy_t;
    static constexpr access_e iter_category = access_e::random;

    auto operator++() -> iterator& {
      DDP_ASSERT(m_index + 1 < m_end);
      ++m_index;
      return *this;
    }
    auto operator--() -> iterator& {
      DDP_ASSERT(m_index - 1 >= m_begin);
      return *this;
    }
    auto operator+=(difference_type n) -> iterator& {
      DDP_ASSERT_MSG_ALL_OF( //
          ("", m_index + n < m_end),
          ("", m_index + n >= m_begin));
      return *this;
    };
    friend auto operator==(iterator a, iterator b) -> bool {
      DDP_ASSERT_MSG_ALL_OF( //
          ("", a.m_end == b.m_end),
          ("", a.m_begin == b.m_begin));
      return a.m_index == b.m_index;
    }
  };

  using const_iterator = iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<iterator>;

  friend auto begin(no_multiplier_feedback_t const& s) -> iterator { return {s.m_begin, s.m_begin, s.m_end}; }
  friend auto end(no_multiplier_feedback_t const& s) -> iterator { return {s.m_begin, s.m_begin, s.m_end}; }
};
} // namespace detail

template <typename Problem>
struct ddp_solver_t {
  using problem_t = Problem;
  using scalar_t = typename Problem::scalar_t;
  using state_indexer_t = typename Problem::state_indexer_t;
  using dstate_indexer_t = typename Problem::dstate_indexer_t;
  using control_indexer_t = typename Problem::control_indexer_t;
  using eq_indexer_t = typename Problem::eq_indexer_t;

  using trajectory_t = trajectory::trajectory_t<scalar_t, state_indexer_t, control_indexer_t>;

  enum struct fn_kind { zero, constant, affine };

  template <fn_kind K, typename = void>
  struct multiplier_sequence_impl;
  template <typename Dummy>
  struct multiplier_sequence_impl<fn_kind::zero, Dummy> {
    using type = detail::no_multiplier_feedback_t;
  };
  template <typename Dummy>
  struct multiplier_sequence_impl<fn_kind::constant, Dummy> {
    using eq_type = detail::matrix_seq::constant_vector_function_seq_t<problem_t, eq_indexer_t>;
    struct type {
      eq_type eq;
    };
    static auto zero(eq_indexer_t const& eq_idx, problem_t const& prob, trajectory_t const& traj) -> type {
      (void)prob;
      (void)traj;
      auto multipliers = type{eq_type{eq_idx.clone(), prob}};
      for (auto e : multipliers.eq) {
        e.val().setZero();
      }
      return multipliers;
    }
  };
  template <typename Dummy>
  struct multiplier_sequence_impl<fn_kind::affine, Dummy> {
    using eq_type = detail::matrix_seq::affine_vector_function_seq_t<problem_t, eq_indexer_t>;
    struct type {
      eq_type eq;
    };
    static auto zero(eq_indexer_t const& eq_idx, problem_t const& prob, trajectory_t const& traj) -> type {
      auto multipliers = type{eq_type{eq_idx.clone(), prob}};
      for (auto zipped : ranges::zip(traj, multipliers.eq)) {
        DDP_BIND(auto&&, (xu, e), zipped);

        prob.neutral_configuration(eigen::as_mut_view(e.origin()));
        e.val().setZero();
        e.jac().setZero();
        e.origin() = xu.x();
      }
      return multipliers;
    }
  };

  template <method M>
  struct multiplier_sequence {
    static constexpr fn_kind K = (M == method::primal or                         //
                                  M == method::primal_dual_constant_multipliers) //
                                     ? fn_kind::constant                         //
                                     : fn_kind::affine;                          //
    using type = typename multiplier_sequence_impl<K>::type;
    static auto zero(eq_indexer_t const& eq_idx, problem_t const& prob, trajectory_t const& traj) -> type {
      return multiplier_sequence_impl<K>::zero(eq_idx, prob, traj);
    }
  };

  template <method M>
  struct multiplier_feedback_sequence {
    static constexpr fn_kind K = M == method::primal ? fn_kind::zero : fn_kind::affine;
    using type = typename multiplier_sequence_impl<K>::type;
    static auto zero(eq_indexer_t const& eq_idx, problem_t const& prob) -> type {
      return multiplier_sequence_impl<K>::zero(eq_idx, prob);
    }
  };

  using control_feedback_t = detail::matrix_seq::affine_vector_function_seq_t<problem_t, control_indexer_t>;

  template <method M>
  auto zero_multipliers(trajectory_t const& traj) const -> typename multiplier_sequence<M>::type {
    return multiplier_sequence<M>::zero(eq_idx, prob, traj);
  }

  template <method M>
  auto zero_feedback_multipliers() const -> typename multiplier_feedback_sequence<M>::type {
    return multiplier_sequence<M>::zero(eq_idx, prob);
  }

  template <typename Control_Gen>
  auto make_trajectory(Control_Gen it_u) const -> trajectory_t {

    trajectory_t traj{prob.state_indexer(this->index_begin(), this->index_end() + 1), u_idx.clone()};
    traj.x_0() = x_init;
    for (auto xu : traj) {
      update_at_pos(xu, it_u());
      it_u.next(eigen::as_const_view(xu.x_next()));
    }
    return traj;
  }

  // clang-format off
  void update_at_pos(
      typename trajectory_t::template proxy_t<false>    xu,
      eigen::view_t<typename trajectory_t::u_vec_t const>  u_new
  ) const {
    // clang-format on
    index_t t = xu.current_index();

    auto xu_c = xu.as_const();
    xu.u() = u_new;
    prob.eval_f_to(xu.x_next(), t, xu_c.x(), xu_c.u());
  }

  template <typename Idx>
  using mat_seq_t = detail::matrix_seq::mat_seq_t<scalar_t, Idx>;

  using one_idx_t = indexing::regular_indexer_t<fix_index<1>>;
  template <typename L, typename R>
  using prod_t = typename indexing::outer_prod_result<L, R>::type;

  template <typename O, typename L, typename R>
  using tensor_seq_t = detail::matrix_seq::tensor_seq_t<scalar_t, indexing::tensor_indexer_t<O, L, R>>;

  using derivative_storage_t =
      ddp::derivative_storage_t<scalar_t, control_indexer_t, eq_indexer_t, state_indexer_t, dstate_indexer_t>;

  auto uninit_derivative_storage() const -> derivative_storage_t {

    index_t b = this->index_begin();
    index_t e = this->index_end();
    auto one_idx = indexing::vec_regular_indexer(b, e, fix_index<1>{});

    index_t dim_xf = prob.dstate_dim().value();

    typename derivative_storage_t::lfx_t lfx(1, dim_xf);
    typename derivative_storage_t::lfxx_t lfxx(dim_xf, dim_xf);

    lfx.setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
    lfxx.setConstant(std::numeric_limits<scalar_t>::quiet_NaN());

    auto x_idx = prob.dstate_indexer(b, e);

    namespace m = detail::matrix_seq;

    auto storage = derivative_storage_t{
        // final cost
        lfx,
        lfxx,
        // cost
        m::mat_seq<scalar_t>(indexing::outer_prod(one_idx, x_idx)),
        m::mat_seq<scalar_t>(indexing::outer_prod(one_idx, u_idx)),
        m::mat_seq<scalar_t>(indexing::outer_prod(x_idx, x_idx)),
        m::mat_seq<scalar_t>(indexing::outer_prod(u_idx, x_idx)),
        m::mat_seq<scalar_t>(indexing::outer_prod(u_idx, u_idx)),
        // transition function
        m::mat_seq<scalar_t>(x_idx),
        m::mat_seq<scalar_t>(indexing::outer_prod(x_idx, x_idx)),
        m::mat_seq<scalar_t>(indexing::outer_prod(x_idx, u_idx)),
        m::tensor_seq<scalar_t>(indexing::tensor_indexer(x_idx, x_idx, x_idx)),
        m::tensor_seq<scalar_t>(indexing::tensor_indexer(x_idx, u_idx, x_idx)),
        m::tensor_seq<scalar_t>(indexing::tensor_indexer(x_idx, u_idx, u_idx)),
        // equality constraints
        m::mat_seq<scalar_t>(eq_idx.clone()),
        m::mat_seq<scalar_t>(indexing::outer_prod(eq_idx.clone(), x_idx)),
        m::mat_seq<scalar_t>(indexing::outer_prod(eq_idx.clone(), u_idx)),
        m::tensor_seq<scalar_t>(indexing::tensor_indexer(eq_idx.clone(), x_idx, x_idx)),
        m::tensor_seq<scalar_t>(indexing::tensor_indexer(eq_idx.clone(), u_idx, x_idx)),
        m::tensor_seq<scalar_t>(indexing::tensor_indexer(eq_idx.clone(), u_idx, u_idx)),
    };

    // clang-format off
    for (auto zipped : ddp::ranges::zip(
             storage.lx, storage.lu, storage.lxx, storage.lux, storage.luu,
             storage.fx, storage.fu, storage.fxx, storage.fux, storage.fuu,
             storage.eq_val, storage.eq_x, storage.eq_u,
             storage.eq_xx, storage.eq_ux, storage.eq_uu
             )) {
      DDP_BIND(auto&&, (
             lx, lu, lxx, lux, luu,
             fx, fu, fxx, fux, fuu,
             eq_val, eq_x, eq_u, eq_xx, eq_ux, eq_uu
           ),
          zipped);
      // clang-format on

      scalar_t zero = 0;
      (*lx).setZero();
      (*lu).setZero();
      (*lxx).setZero();
      (*lux).setZero();
      (*luu).setZero();

      (*fx).setZero();
      (*fu).setZero();

      (*eq_val).setZero();
      (*eq_x).setZero();
      (*eq_u).setZero();

      (*fxx).set_constant(zero);
      (*fux).set_constant(zero);
      (*fuu).set_constant(zero);

      (*eq_xx).set_constant(zero);
      (*eq_ux).set_constant(zero);
      (*eq_uu).set_constant(zero);
    }

    return storage;
  }

  auto optimality_constr(derivative_storage_t const& derivs) const -> scalar_t {
    using std::max;
    scalar_t retval(0);
    for (auto eq : derivs.eq()) {
      retval = max(retval, eq.val.stableNorm());
    }
    return retval;
  }

  template <typename Mult_Seq>
  auto optimality_lag(trajectory_t const& traj, Mult_Seq const& mults, derivative_storage_t const& derivs) const
      -> scalar_t {

    namespace rng = ddp::ranges;
    using std::max;

    scalar_t retval = 0;
    auto adj = derivs.lfx;

    eigen::matrix_from_idx_t<scalar_t, eq_indexer_t> pe_storage(eq_idx.max_rows().value());
    eigen::matrix_from_idx_t<scalar_t, control_indexer_t> lu_storage(u_idx.max_rows().value());

    for (auto zipped : //
         rng::reverse( //
             rng::zip( //
                 traj,
                 mults.eq,
                 derivs.l(),
                 derivs.f(),
                 derivs.eq()))) {
      DDP_BIND(auto const&, (xu, p_eq, l, f, eq), zipped);

      index_t t = xu.current_index();

      auto pe = eigen::as_mut_view(pe_storage.                                       //
                                   template topRows<                                 //
                                       eq_indexer_t::row_kind::value_at_compile_time //
                                       >(eq_idx.rows(t).value()));

      auto lu = eigen::as_mut_view(lu_storage.                                            //
                                   template topRows<                                      //
                                       control_indexer_t::row_kind::value_at_compile_time //
                                       >(u_idx.rows(t).value()));

      p_eq(pe, xu.x());

      lu = l.u;
      lu.noalias() += pe.transpose() * eq.u;
      lu.noalias() += adj * f.u;

      retval = max(retval, lu.stableNorm());

      adj = l.x + (adj * f.x).eval();
      adj.noalias() += pe.transpose() * eq.x;
      adj.noalias() += eq.val.transpose() * p_eq.jac();
    }
    return retval;
  }

  template <typename Mult_Seq>
  auto optimality_obj(
      trajectory_t const& traj, Mult_Seq const& mults, scalar_t const& mu, derivative_storage_t const& derivs) const
      -> scalar_t {

    namespace rng = ddp::ranges;
    using std::max;

    scalar_t retval = 0;
    auto adj = derivs.lfx;

    eigen::matrix_from_idx_t<scalar_t, eq_indexer_t> pe_storage(eq_idx.max_rows().value());
    eigen::matrix_from_idx_t<scalar_t, control_indexer_t> lu_storage(u_idx.max_rows().value());

    for (auto zipped : //
         rng::reverse( //
             rng::zip( //
                 traj,
                 mults.eq,
                 derivs.l(),
                 derivs.f(),
                 derivs.eq()))) {
      DDP_BIND(auto const&, (xu, p_eq, l, f, eq), zipped);
      index_t t = xu.current_index();

      auto pe = eigen::as_mut_view(pe_storage.                                       //
                                   template topRows<                                 //
                                       eq_indexer_t::row_kind::value_at_compile_time //
                                       >(eq_idx.rows(t).value()));

      auto lu = eigen::as_mut_view(lu_storage.                                            //
                                   template topRows<                                      //
                                       control_indexer_t::row_kind::value_at_compile_time //
                                       >(u_idx.rows(t).value()));

      p_eq(pe, xu.x());

      lu = l.u;
      lu.noalias() += pe.transpose() * eq.u;
      lu.noalias() += mu * eq.val.transpose() * eq.u;
      lu.noalias() += adj * f.u;

      retval = max(retval, lu.stableNorm());

      adj = adj * f.x;
      adj += l.x;
      adj += mu * eq.val.transpose() * eq.x;
      adj += pe.transpose() * eq.x;
      adj += eq.val.transpose() * p_eq.jac();
    }
    return retval;
  }

  auto index_begin() const -> index_t {
    index_t b = u_idx.index_begin();
    DDP_ASSERT(b == eq_idx.index_begin());
    return b;
  }

  auto index_end() const -> index_t {
    index_t e = u_idx.index_end();
    DDP_ASSERT(e == eq_idx.index_end());
    return e;
  }

  template <method M>
  auto update_derivatives(
      derivative_storage_t& derivatives,
      control_feedback_t& fb_seq,
      typename multiplier_sequence<M>::type& mults,
      trajectory_t const& traj,
      scalar_t const& mu,
      scalar_t const& w,
      scalar_t const& n,
      scalar_t const& stopping_threshold) const -> detail::tuple<mult_update_attempt_result_e, scalar_t, scalar_t> {
    std::string name = detail::to_owned(prob.name());
    log_file_t primal_log{((M == method::primal_dual_affine_multipliers //
                                ? "/tmp/affine_mults_"
                                : "/tmp/") +
                           name + "_primal.dat")
                              .c_str()};
    log_file_t dual_log{("/tmp/" + DDP_MOVE(name) + "_dual.dat").c_str()};

    prob.compute_derivatives(derivatives, traj);

    mults.eq.update_origin(traj.m_state_data);
    fb_seq.update_origin(traj.m_state_data);

    auto opt_obj = optimality_obj(traj, mults, mu, derivatives);
    auto opt_constr = optimality_constr(derivatives);

    fmt::print(primal_log.ptr, "{}\n", opt_constr);
    fmt::print(dual_log.ptr, "{}\n", opt_obj);

    fmt::print(
        stdout,
        "opt obj: {}\n"
        "opt constr: {}\n",
        opt_obj,
        opt_constr);

    if (opt_constr < stopping_threshold and opt_obj < stopping_threshold) {
      return {
          mult_update_attempt_result_e::optimum_attained,
          opt_obj,
          opt_constr,
      };
    }

    if (opt_obj < w) {
      if (opt_constr < n) {

        for (auto zipped : ranges::zip( //
                 mults.eq,              //
                 derivatives.eq(),      //
                 fb_seq)) {
          DDP_BIND(auto&&, (p_eq, eq, fb), zipped);

          p_eq.val() += mu * (eq.val + eq.u * fb.val());
          p_eq.jac() += mu * (eq.x + eq.u * fb.jac());
        }
        return {
            mult_update_attempt_result_e::update_success,
            opt_obj,
            opt_constr,
        };
      } else {
        return {
            mult_update_attempt_result_e::update_failure,
            opt_obj,
            opt_constr,
        };
      }
    } else {
      return {
          mult_update_attempt_result_e::no_update,
          opt_obj,
          opt_constr,
      };
    }
  }

  using dyn_vec_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
  template <typename Mults>
  void cost_seq_aug(                      //
      eigen::view_t<dyn_vec_t> out_costs, //
      trajectory_t const& traj,           //
      Mults const& mults,                 //
      scalar_t const& mu                  //
  ) const {

    eigen::matrix_from_idx_t<scalar_t, eq_indexer_t> ce_storage(eq_idx.max_rows().value());
    eigen::matrix_from_idx_t<scalar_t, eq_indexer_t> pe_storage(eq_idx.max_rows().value());

    for (auto zipped : ranges::zip(traj, mults.eq)) {
      DDP_BIND(auto const&, (xu, p_eq), zipped);
      index_t t = xu.current_index();

      auto const x = xu.x();
      auto const u = xu.u();

      auto ce = eigen::as_mut_view(ce_storage.                                       //
                                   template topRows<                                 //
                                       eq_indexer_t::row_kind::value_at_compile_time //
                                       >(eq_idx.rows(t).value()));

      auto pe = eigen::as_mut_view(pe_storage.                                       //
                                   template topRows<                                 //
                                       eq_indexer_t::row_kind::value_at_compile_time //
                                       >(eq_idx.rows(t).value()));

      auto const l = prob.l(t, x, u);
      prob.eval_eq_to(ce, t, x, u); // ce = eq(t,x,u)
      p_eq(pe, x);                  // pe = p_eq(t, x, u)

      out_costs(t - this->index_begin()) = l + pe.dot(ce) + (mu / 2) * ce.squaredNorm();
    }
    auto const x = traj.x_f();
    out_costs(this->index_end() - this->index_begin()) = prob.lf(x);
  }

  ~ddp_solver_t() = default;
  ddp_solver_t(ddp_solver_t const&) = delete;
  ddp_solver_t(ddp_solver_t&&) = delete;
  auto operator=(ddp_solver_t const&) -> ddp_solver_t& = delete;
  auto operator=(ddp_solver_t &&) -> ddp_solver_t& = delete;

  // clang-format off
  template <method M>
  auto solve(
      solver_parameters_t<scalar_t>  solver_parameters,
      trajectory_t                   initial_trajectory
  ) const -> detail::tuple<trajectory_t, control_feedback_t> {
    // clang-format on
    std::string name = detail::to_owned(prob.name());
    log_file_t traj_log{((M == method::primal_dual_affine_multipliers //
                              ? "/tmp/affine_mults_"
                              : "/tmp/") +
                         name + "_traj.dat")
                            .c_str()};

    auto derivatives = uninit_derivative_storage();
    auto& traj = initial_trajectory;
    auto new_traj = traj.clone();

    regularization_t<scalar_t> reg{0, 1, 2, 1e-5};

    auto mults = zero_multipliers<M>(traj);

    auto ctrl_fb = control_feedback_t{u_idx, prob};

    scalar_t mu = solver_parameters.mu_init;

    auto previous_opt_constr = optimality_constr(derivatives);

    scalar_t w = 1 / mu;
    scalar_t n = 1 / pow(mu, static_cast<scalar_t>(0.1L));

    traj.println_to_file(traj_log.ptr);

    for (index_t iter = 0; iter < solver_parameters.max_iterations; ++iter) {

      if (iter == 0) {
        prob.compute_derivatives(derivatives, traj);
      } else {
        DDP_BIND(
            auto,
            (mult_update_rv, opt_obj, opt_constr),
            (chronometer_t{"computing derivatives"},
             update_derivatives<M>( //
                 derivatives,
                 ctrl_fb,
                 mults,
                 traj,
                 mu,
                 w,
                 n,
                 solver_parameters.optimality_stopping_threshold)));
        (void)opt_obj;

        scalar_t const beta = 0.5;
        chronometer_t c{"updating multipliers"};
        switch (mult_update_rv) {
        case mult_update_attempt_result_e::no_update: {
          break;
        }
        case mult_update_attempt_result_e::update_failure: {
          using std::pow;
          fmt::print("desired new mu {}\n", pow(mu / (previous_opt_constr / opt_constr), 1 / (1 - beta)));
          mu = 10 * std::max(     //
                        std::min( //
                            pow(mu / (previous_opt_constr / opt_constr), 1.0 / (1 - beta)),
                            mu * scalar_t{1e5}),
                        mu);
          break;
        }
        case mult_update_attempt_result_e::update_success: {
          using std::pow;
          n = opt_constr / pow(mu, beta / 2);
          w /= pow(mu, scalar_t{1});
          previous_opt_constr = opt_constr;
          break;
        }
        case mult_update_attempt_result_e::optimum_attained:
          return {DDP_MOVE(traj), DDP_MOVE(ctrl_fb)};
        }
      }

      {
        chronometer_t c{"backward pass"};
        backward_pass<M>(ctrl_fb, reg, mu, traj, mults, derivatives);
      }
      fmt::print(
          stdout,
          "====================================================================================================\n"
          "iter: {:5}   mu: {:13}   reg: {:13}   w: {:13}   n: {:13}\n",
          iter,
          mu,
          *reg,
          w,
          n);

      {
        scalar_t step = (chronometer_t{"forward_pass"}, forward_pass<M>(new_traj, traj, mults, ctrl_fb, mu, true));
        if (step >= 0.5) {
          reg.decrease_reg();
        }

        swap(traj, new_traj);

        traj.println_to_file(traj_log.ptr);
        fmt::print(stdout, "step: {}\n", step);
        fmt::print(stdout, "eq: ");
      }

      fmt::string_view sep = "";
      for (auto eq : derivatives.eq()) {
        if (eq.val.size() > 0) {
          fmt::print("{}{}", sep, eq.val.norm());
          sep = ", ";
        }
      }
      fmt::print("\n");
    }

    return {DDP_MOVE(traj), DDP_MOVE(ctrl_fb)};
  }

  // clang-format off
  template <method M>
  void backward_pass(
      control_feedback_t&                             ctrl_fb,
      regularization_t<scalar_t>&                     regularization,
      scalar_t&                                       mu,
      trajectory_t const&                             current_traj,
      typename multiplier_sequence<M>::type const&    mults,
      derivative_storage_t const&                     derivatives
  ) const;

  template <method M>
  auto forward_pass(
      trajectory_t&                                   new_traj_storage,
      trajectory_t const&                             reference_traj,
      typename multiplier_sequence<M>::type const&    old_mults,
      control_feedback_t const&                       feedback,
      scalar_t                                        mu,
      bool                                            do_linesearch = true
  ) const -> scalar_t;

  Problem const&                          prob;
  control_indexer_t                       u_idx;
  eq_indexer_t                            eq_idx;
  typename trajectory_t::x_vec_t const&   x_init;
  // clang-format on
}; // namespace ddp

} // namespace ddp

#endif /* end of include guard DDP_HPP_FBGSDEO3 */
