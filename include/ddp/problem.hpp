#ifndef PROBLEM_HPP_SQUCCNMX
#define PROBLEM_HPP_SQUCCNMX

#include "ddp/trajectory.hpp"
#include "ddp/ddp.hpp"
#include "ddp/constraint.hpp"

namespace ddp {

template <typename Out, typename XX, typename UX, typename UU, typename DX, typename DU>
void add_second_order_term(Out&& out, XX const& xx, UX const& ux, UU const& uu, DX const& dx, DU const& du) {
  DDP_ASSERT_MSG_ALL_OF(
      ("", out.cols() == 1),
      ("", out.rows() == xx.outdim().value()),
      ("", out.rows() == ux.outdim().value()),
      ("", out.rows() == uu.outdim().value()),
      ("", dx.rows() == xx.indiml().value()),
      ("", dx.rows() == xx.indimr().value()),
      ("", dx.rows() == ux.indimr().value()),
      ("", du.rows() == uu.indiml().value()),
      ("", du.rows() == uu.indimr().value()),
      ("", du.rows() == ux.indiml().value()));

  for (index_t j = 0; j < dx.rows(); ++j) {
    for (index_t i = 0; i < dx.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * dx(i) * xx(k, i, j) * dx(j);
      }
    }
  }

  for (index_t j = 0; j < du.rows(); ++j) {
    for (index_t i = 0; i < du.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * du(i) * uu(k, i, j) * du(j);
      }
    }
  }

  for (index_t j = 0; j < dx.rows(); ++j) {
    for (index_t i = 0; i < du.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += du(i) * ux(k, i, j) * dx(j);
      }
    }
  }
}

namespace detail {

template <typename Iter>
void split_into_segments(
    Iter* const iter_arr, index_t const n_split, Iter const begin, Iter const end, index_t const diff) {

  index_t const r = diff % n_split;
  Iter it = begin;

  for (index_t i = 0; i < n_split; ++i) {
    index_t const n_iterations = diff / n_split + ((i < r) ? 1 : 0);

    iter_arr[i] = it;
    for (index_t j = 0; j < n_iterations; ++j) {
      ++it;
    }
  }
  iter_arr[n_split] = it;
  DDP_ASSERT((it == end));
}

} // namespace detail

template <typename Problem>
void compute_derivatives(
    Problem const& prob, typename Problem::derivative_storage_t& derivs, typename Problem::trajectory_t const& traj) {
  derivs.lfx.setZero();
  derivs.lfxx.setZero();

  // clang-format off
    auto rng = ranges::zip(
          derivs.lx, derivs.lu, derivs.lxx, derivs.lux, derivs.luu,
          derivs.f_val, derivs.fx, derivs.fu, derivs.fxx, derivs.fux, derivs.fuu,
          derivs.eq_val, derivs.eq_x, derivs.eq_u, derivs.eq_xx, derivs.eq_ux, derivs.eq_uu,
          traj);
  // clang-format on

  using iter = typename decltype(rng)::iterator;
  constexpr index_t max_threads =
#ifdef NDEBUG
      16
#else
      1
#endif
      ;

  index_t const n_threads = std::min(max_threads, index_t{omp_get_num_procs()});
  iter _iter_arr[max_threads + 1];

  {
    index_t const horizon = 1 + (*(--end(traj))).current_index() - (*begin(traj)).current_index();
    detail::split_into_segments(_iter_arr, n_threads, begin(rng), end(rng), horizon);
  }

  iter const* iter_arr = _iter_arr;

  prob.lf_second_order_deriv( //
      eigen::as_mut_view(derivs.lfxx),
      eigen::as_mut_view(derivs.lfx),
      traj.x_f());

#ifdef NDEBUG
#pragma omp parallel default(none) shared(n_threads) shared(prob) shared(iter_arr) num_threads(n_threads)
#endif
  {
    index_t thread_id = omp_get_thread_num();
    DDP_ASSERT(thread_id >= 0);
    DDP_ASSERT(thread_id < n_threads);

    auto k = prob.dynamics().acquire_workspace();

    iter const it_end = iter_arr[thread_id + 1];
    for (iter it = iter_arr[thread_id]; it != it_end; ++it) {
      auto zipped = *it;
      // clang-format off
      DDP_BIND(auto&&, (
          lx, lu, lxx, lux, luu,
          f_v, fx, fu, fxx, fux, fuu,
          eq_v, eq_x, eq_u, eq_xx, eq_ux, eq_uu,
          xu), zipped);
      // clang-format on

      DDP_ASSERT(not xu.x().hasNaN());
      DDP_ASSERT(not xu.x_next().hasNaN());
      DDP_ASSERT(not xu.u().hasNaN());

      auto t = xu.current_index();
      auto x = xu.x();
      auto u = xu.u();

      prob.l_second_order_deriv(lxx.get(), lux.get(), luu.get(), lx.get(), lu.get(), t, x, u);

      {
        auto msg = fmt::format("  computing f  derivatives from thread {}", thread_id);
        ddp::chronometer_t timer(msg.c_str());
        k = prob.dynamics().second_order_deriv(
            fxx.get(),
            fux.get(),
            fuu.get(),
            fx.get(),
            fu.get(),
            f_v.get(),
            t,
            x,
            u,
            DDP_MOVE(k));
      }
      {
        auto msg = fmt::format("  computing eq derivatives from thread {}", thread_id);
        ddp::chronometer_t timer(msg.c_str());
        k = prob.constraint().second_order_deriv(
            eq_xx.get(),
            eq_ux.get(),
            eq_uu.get(),
            eq_x.get(),
            eq_u.get(),
            eq_v.get(),
            t,
            x,
            u,
            DDP_MOVE(k));
      }

      {
        using scalar_t = typename Problem::scalar_t;
        auto msg = fmt::format("  checking all derivatives from thread {}", thread_id);
        ddp::chronometer_t timer(msg.c_str());
        DDP_ASSERT(not eq_v.get().hasNaN());
        DDP_ASSERT(not eq_x.get().hasNaN());
        DDP_ASSERT(not eq_u.get().hasNaN());
        DDP_ASSERT(not eq_xx.get().has_nan());
        DDP_ASSERT(not eq_ux.get().has_nan());
        DDP_ASSERT(not eq_uu.get().has_nan());

        DDP_ASSERT(not fx.get().hasNaN());
        DDP_ASSERT(not fu.get().hasNaN());
        DDP_ASSERT(not fxx.get().has_nan());
        DDP_ASSERT(not fux.get().has_nan());
        DDP_ASSERT(not fuu.get().has_nan());

        auto ne = eq_v.rows();
        auto ndx = fu.rows();
        auto ndu = fu.cols();

        scalar_t l;
        scalar_t l_;
        auto f = f_v.get().eval();
        auto f_ = f_v.get().eval();
        auto eq = eq_v.get().eval();
        auto eq_ = eq_v.get().eval();

        scalar_t _dl1;
        scalar_t _ddl1;
        scalar_t _dddl1;
        scalar_t _dl2;
        scalar_t _ddl2;
        scalar_t _dddl2;

        auto _df1 = eigen::make_matrix<scalar_t>(ndx);
        auto _ddf1 = eigen::make_matrix<scalar_t>(ndx);
        auto _dddf1 = eigen::make_matrix<scalar_t>(ndx);
        auto _df2 = eigen::make_matrix<scalar_t>(ndx);
        auto _ddf2 = eigen::make_matrix<scalar_t>(ndx);
        auto _dddf2 = eigen::make_matrix<scalar_t>(ndx);

        auto _deq1 = eigen::make_matrix<scalar_t>(ne);
        auto _ddeq1 = eigen::make_matrix<scalar_t>(ne);
        auto _dddeq1 = eigen::make_matrix<scalar_t>(ne);
        auto _deq2 = eigen::make_matrix<scalar_t>(ne);
        auto _ddeq2 = eigen::make_matrix<scalar_t>(ne);
        auto _dddeq2 = eigen::make_matrix<scalar_t>(ne);

        auto* dl = &_dl1;
        auto* df = &_df1;
        auto* deq = &_deq1;

        auto* ddl = &_ddl1;
        auto* ddf = &_ddf1;
        auto* ddeq = &_ddeq1;

        auto* dddl = &_dddl1;
        auto* dddf = &_dddf1;
        auto* dddeq = &_dddeq1;

        auto dx_ = eigen::make_matrix<scalar_t>(ndx);
        dx_.setRandom();

        auto du_ = eigen::make_matrix<scalar_t>(ndu);
        du_.setRandom();

        auto finite_diff_err = [&](scalar_t eps_x, scalar_t eps_u) {
          auto dx = (dx_.operator*(eps_x)).eval();
          auto du = (du_.operator*(eps_u)).eval();

          auto x_ = x.eval();
          auto u_ = u.eval();

          prob.dynamics().integrate_x(eigen::as_mut_view(x_), eigen::as_const_view(x), eigen::as_const_view(dx));
          prob.dynamics().integrate_u(
              eigen::as_mut_view(u_),
              t,
              eigen::as_const_view(u),
              eigen::into_view(eigen::as_const_view(du)));

          l = prob.l(t, x, u);
          k = prob.eval_f_to(eigen::as_mut_view(f), t, x, u, DDP_MOVE(k));
          k = prob.eval_eq_to(eigen::as_mut_view(eq), t, x, u, DDP_MOVE(k));

          l_ = prob.l(t, eigen::as_const_view(x_), eigen::as_const_view(u_));
          k = prob.eval_f_to(
              eigen::as_mut_view(f_),
              t,
              eigen::as_const_view(x_),
              eigen::as_const_view(u_),
              DDP_MOVE(k));
          k = prob.eval_eq_to(
              eigen::as_mut_view(eq_),
              t,
              eigen::as_const_view(x_),
              eigen::as_const_view(u_),
              DDP_MOVE(k));

          *dl = l_ - l;
          prob.dynamics().difference_out(eigen::as_mut_view(*df), eigen::as_const_view(f), eigen::as_const_view(f_));
          prob.constraint().difference_out(
              eigen::as_mut_view(*deq),
              eigen::as_const_view(eq),
              eigen::as_const_view(eq_));

          *ddl = *dl - (lx.get() * dx + lu.get() * du).value();
          *ddf = *df - (fx.get() * dx + fu.get() * du);
          *ddeq = *deq - (eq_x.get() * dx + eq_u.get() * du);

          *dddl = *ddl - (0.5 * dx.transpose() * lxx.get() * dx   //
                          + 0.5 * du.transpose() * luu.get() * du //
                          + du.transpose() * lux.get() * dx)
                             .value();
          *dddf = -*ddf;
          *dddeq = -*ddeq;

          ddp::add_second_order_term(*dddf, fxx.get(), fux.get(), fuu.get(), dx, du);
          ddp::add_second_order_term(*dddeq, eq_xx.get(), eq_ux.get(), eq_uu.get(), dx, du);

          *dddf = -*dddf;
          *dddeq = -*dddeq;
        };

        scalar_t eps_x = 1e-30;
        scalar_t eps_u = 1e-30;

        scalar_t eps_factor = 0.1;

        finite_diff_err(eps_x, eps_u);

        dl = &_dl2;
        df = &_df2;
        deq = &_deq2;

        ddl = &_ddl2;
        ddf = &_ddf2;
        ddeq = &_ddeq2;

        dddl = &_dddl2;
        dddf = &_dddf2;
        dddeq = &_dddeq2;

        finite_diff_err(eps_x * eps_factor, eps_u * eps_factor);

        auto dl1 = _dl1;
        auto dl2 = _dl2;
        auto ddl1 = _ddl1;
        auto ddl2 = _ddl2;
        auto dddl1 = _dddl1;
        auto dddl2 = _dddl2;

        auto df1 = _df1.array();
        auto df2 = _df2.array();
        auto ddf1 = _ddf1.array();
        auto ddf2 = _ddf2.array();
        auto dddf1 = _dddf1.array();
        auto dddf2 = _dddf2.array();

        auto deq1 = _deq1.array();
        auto deq2 = _deq2.array();
        auto ddeq1 = _ddeq1.array();
        auto ddeq2 = _ddeq2.array();
        auto dddeq1 = _dddeq1.array();
        auto dddeq2 = _dddeq2.array();

        using std::fabs;
        using std::pow;

        auto eps = pow(std::numeric_limits<scalar_t>::epsilon(), 0.9);

        DDP_EXPECT_MSG_ALL_OF(
            (fmt::format("at t = {}\n{}\n{}", t, dl1, dl2),
             (fabs(dl1) <= eps or fabs(dl2) <= 2 * eps_factor * fabs(dl1))),
            (fmt::format("at t = {}\n{}\n{}", t, ddl1, ddl2),
             (fabs(ddl1) <= eps or fabs(ddl2) <= 2 * pow(eps_factor, 2) * fabs(ddl1))),
            (fmt::format("at t = {}\n{}\n{}", t, dddl1, dddl2),
             (fabs(dddl1) <= eps or fabs(dddl2) <= 2 * pow(eps_factor, 3) * fabs(dddl1))));

        for (index_t i = 0; i < df1.size(); ++i) {
          DDP_EXPECT_MSG_ALL_OF(
              (fmt::format("at t = {}, i = {}\n{}\n{}", t, i, df1[i], df2[i]),
               (fabs(df1[i]) <= eps or fabs(df2[i]) <= 2 * eps_factor * fabs(df1[i]))),
              (fmt::format("at t = {}, i = {}\n{}\n{}", t, i, ddf1[i], ddf2[i]),
               (fabs(ddf1[i]) <= eps or fabs(ddf2[i]) <= 2 * pow(eps_factor, 2) * fabs(ddf1[i]))),
              (fmt::format("at t = {}, i = {}\n{}\n{}", t, i, dddf1[i], dddf2[i]),
               (fabs(dddf1[i]) <= eps or fabs(dddf2[i]) <= 2 * pow(eps_factor, 3) * fabs(dddf1[i]))));
        }

        for (index_t i = 0; i < deq1.size(); ++i) {
          DDP_EXPECT_MSG_ALL_OF(
              (fmt::format("at t = {}\n{}\n{}", t, deq1[i], deq2[i]),
               (fabs(deq1[i]) <= eps or fabs(deq2[i]) <= 2 * eps_factor * fabs(deq1[i]))),
              (fmt::format("at t = {}\n{}\n{}", t, ddeq1[i], ddeq2[i]),
               (fabs(ddeq1[i]) <= eps or fabs(ddeq2[i]) <= 2 * pow(eps_factor, 2) * fabs(ddeq1[i]))),
              (fmt::format("at t = {}\n{}\n{}", t, dddeq1[i], dddeq2[i]),
               (fabs(dddeq1[i]) <= eps or fabs(dddeq2[i]) <= 2 * pow(eps_factor, 3) * fabs(dddeq1[i]))));
        }
      }
    }
  }
}

template <typename Dynamics, typename Eq>
struct problem_t {
  using dynamics_t = Dynamics;
  using scalar_t = typename dynamics_t::scalar_t;

  using constraint_t = Eq;

  using state_indexer_t = typename dynamics_t::state_indexer_t;
  using dstate_indexer_t = typename dynamics_t::dstate_indexer_t;
  using control_indexer_t = typename dynamics_t::control_indexer_t;
  using dcontrol_indexer_t = typename dynamics_t::dcontrol_indexer_t;
  using eq_indexer_t = typename constraint_t::constr_indexer_t;

  using f_dims = typename dynamics_t::dims;
  using eq_dims = typename constraint_t::dims;

  using key = typename dynamics_t::key;

  auto dynamics() const -> dynamics_t const& { return m_dynamics; }
  auto constraint() const -> constraint_t const& { return m_constraint; }

  auto state_dim() const -> typename f_dims::x_t { return m_dynamics.state_dim(); }
  auto dstate_dim() const -> typename f_dims::dx_t { return m_dynamics.dstate_dim(); }
  auto control_dim(index_t t) const -> typename f_dims::u_t { return m_dynamics.control_dim(t); }
  auto dcontrol_dim(index_t t) const -> typename f_dims::du_t { return m_dynamics.dcontrol_dim(t); }

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return m_dynamics.state_indexer(begin, end);
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return m_dynamics.dstate_indexer(begin, end);
  }

  void neutral_configuration(x_mut<f_dims> out) const { m_dynamics.neutral_configuration(out); }

  auto lf(x_const<f_dims> x) const -> scalar_t {
    (void)x;
    (void)this;
    return 0;
  }
  void lf_second_order_deriv(xx_mut<f_dims> lxx, xt_mut<f_dims> lx, x_const<f_dims> x) const {
    (void)x;
    lxx.setZero();
    lx.setZero();
  }

  auto l(index_t t, x_const<f_dims> x, u_const<f_dims> u) const -> scalar_t {
    (void)t;
    (void)x;
    (void)this;
    return 0.5 * c * u.squaredNorm();
  }
  void l_second_order_deriv(
      xx_mut<f_dims> lxx,
      ux_mut<f_dims> lux,
      uu_mut<f_dims> luu,
      xt_mut<f_dims> lx,
      ut_mut<f_dims> lu,
      index_t t,
      x_const<f_dims> x,
      u_const<f_dims> u) const {
    (void)t;
    (void)x;
    lxx.setZero();
    lux.setZero();
    luu.setIdentity();
    luu *= c;
    lx.setZero();
    lu = c * u.transpose();
  }

  void difference(dout_mut<f_dims> out, out_const<f_dims> start, out_const<f_dims> finish) const {
    m_dynamics.difference_out(out, start, finish);
  }
  void d_difference_dfinish(out_x_mut<f_dims> out, out_const<f_dims> start, out_const<f_dims> finish) const {
    m_dynamics.d_difference_out_dfinish(out, start, finish);
  }

  auto eval_f_to(out_mut<f_dims> x_out, index_t t, x_const<f_dims> x, u_const<f_dims> u, key k = {}) const -> key {
    if (!k) {
      k = m_dynamics.acquire_workspace();
    }
    return m_dynamics.eval_to(x_out, t, x, u, DDP_MOVE(k));
  }
  auto eval_eq_to(out_mut<eq_dims> eq_out, index_t t, x_const<f_dims> x, u_const<f_dims> u, key k = {}) const -> key {
    if (!k) {
      k = m_dynamics.acquire_workspace();
    }
    return m_constraint.eval_to(eq_out, t, x, u, DDP_MOVE(k));
  }

  using derivative_storage_t =
      ddp::derivative_storage_t<scalar_t, control_indexer_t, eq_indexer_t, state_indexer_t, dstate_indexer_t>;
  using trajectory_t = trajectory::trajectory_t<scalar_t, state_indexer_t, control_indexer_t>;

  void compute_derivatives(derivative_storage_t& derivs, trajectory_t const& traj) const {
    ddp::compute_derivatives(*this, derivs, traj);
  }

  auto name() const -> fmt::string_view { return m_dynamics.name(); }

  index_t m_begin{};
  index_t m_end{};
  scalar_t c = 1e2;
  dynamics_t m_dynamics;
  constraint_t m_constraint;
};

template <typename Dynamics, typename Eq_Constraint>
auto problem(index_t begin, index_t end, typename Dynamics::scalar_t cost_reg, Dynamics dyn, Eq_Constraint eq)
    -> problem_t<Dynamics, Eq_Constraint> {
  return {begin, end, cost_reg, DDP_MOVE(dyn), DDP_MOVE(eq)};
}

template <typename Problem, typename Slack_Idx>
struct multiple_shooting_t {
  using problem_t = Problem;
  using slack_indexer_t = Slack_Idx;
  using scalar_t = typename problem_t::scalar_t;

  using orig_dynamics_t = typename problem_t::dynamics_t;
  using orig_constraint_t = typename problem_t::constraint_t;

  using state_indexer_t = typename problem_t::state_indexer_t;
  using dstate_indexer_t = typename problem_t::dstate_indexer_t;
  using control_indexer_t = indexing::row_concat_indexer_t<typename problem_t::control_indexer_t, Slack_Idx>;
  using dcontrol_indexer_t = indexing::row_concat_indexer_t<typename problem_t::dcontrol_indexer_t, Slack_Idx>;
  using eq_indexer_t = indexing::row_concat_indexer_t<typename problem_t::eq_indexer_t, Slack_Idx>;

  using f_dims = dimensions_from_idx_t<
      scalar_t,
      state_indexer_t,
      dstate_indexer_t,
      control_indexer_t,
      dcontrol_indexer_t,
      state_indexer_t,
      dstate_indexer_t>;

  using eq_dims = dimensions_from_idx_t<
      scalar_t,
      state_indexer_t,
      dstate_indexer_t,
      control_indexer_t,
      dcontrol_indexer_t,
      eq_indexer_t,
      eq_indexer_t>;

  using key = typename problem_t::key;

  struct dynamics_t {
    multiple_shooting_t const& m_parent;

    using scalar_t = multiple_shooting_t::scalar_t;
    using dims = multiple_shooting_t::f_dims;

    using key = typename multiple_shooting_t::key;

    auto orig() const -> orig_dynamics_t { return m_parent.m_prob.dynamics(); }

    auto state_dim() const DDP_DECLTYPE_AUTO(m_parent.state_dim());
    auto dstate_dim() const DDP_DECLTYPE_AUTO(m_parent.dstate_dim());
    auto state_indexer(index_t begin, index_t end) const DDP_DECLTYPE_AUTO(m_parent.state_indexer(begin, end));
    auto dstate_indexer(index_t begin, index_t end) const DDP_DECLTYPE_AUTO(m_parent.dstate_indexer(begin, end));
    auto control_dim(index_t t) const DDP_DECLTYPE_AUTO(m_parent.control_dim(t));
    auto dcontrol_dim(index_t t) const DDP_DECLTYPE_AUTO(m_parent.dcontrol_dim(t));

    auto acquire_workspace() const -> key { return orig().acquire_workspace(); }

    void neutral_configuration(x_mut<dims> out) const { orig().neutral_configuration(out); }
    void difference_out(dx_mut<dims> out, x_const<dims> start, x_const<dims> finish) const {
      orig().difference_out(out, start, finish);
    }
    void d_difference_out_dfinish(out_x_mut<dims> out, x_const<dims> start, x_const<dims> finish) const {
      orig().d_difference_out_dfinish(out, start, finish);
    }

    auto eval_to(out_mut<dims> x_out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
      return m_parent.eval_f_to(x_out, t, x, u, DDP_MOVE(k));
    }

    auto first_order_deriv( //
        out_x_mut<dims> fx, //
        out_u_mut<dims> fu, //
        out_mut<dims> f,    //
        index_t t,          //
        x_const<dims> x,    //
        u_const<dims> u,    //
        key k               //
    ) const -> key {
      DDP_BIND(auto, (fu_orig, fu_slack), eigen::split_at_col_mut(fu, orig().dcontrol_dim(t)));
      DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, orig().control_dim(t)));

      if (u_slack.rows() == 0) {
        k = orig().first_order_deriv(fx, fu_orig, f, t, x, u_orig, DDP_MOVE(k));
        return k;
      }

      DDP_ASSERT(u_slack.rows() == dstate_dim().value());
      DDP_BIND(auto, (fu_slack_x, fu_slack_0), eigen::split_at_col_mut(fu_slack, dstate_dim()));
      DDP_BIND(auto, (u_slack_x, u_slack_0), eigen::split_at_row(u_slack, state_dim()));
      DDP_ASSERT(fu_slack_0.cols() == 0);
      DDP_ASSERT(u_slack_0.rows() == 0);

      auto _tmp0 = f.eval();
      auto _tmp1 = fx.eval();
      auto _tmp2 = fx.eval();
      auto tmp1 = eigen::as_mut_view(_tmp1);

      k = orig().first_order_deriv(fx, fu_orig, f, t, x, u_orig, DDP_MOVE(k));
      orig().d_integrate_x(tmp1, eigen::as_const_view(f), u_slack_x);
      orig().d_integrate_x_dx(fu_slack_x, eigen::as_const_view(f), u_slack_x);

      fx = tmp1 * fx;
      fu_orig = tmp1 * fu_orig;

      return eval_to(f, t, x, u, DDP_MOVE(k));
    }

    auto second_order_deriv(  //
        out_xx_mut<dims> fxx, //
        out_ux_mut<dims> fux, //
        out_uu_mut<dims> fuu, //
        out_x_mut<dims> fx,   //
        out_u_mut<dims> fu,   //
        out_mut<dims> f,      //
        index_t t,            //
        x_const<dims> x,      //
        u_const<dims> u,      //
        key k                 //
    ) const -> key {
      k = finite_diff_hessian_compute<dynamics_t>{*this, orig().second_order_finite_diff()}
              .second_order_deriv(fxx, fux, fuu, fx, fu, f, t, x, u, DDP_MOVE(k));
      return k;
    }
    auto second_order_finite_diff() const -> bool { return orig().second_order_finite_diff(); }

    void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const { orig().integrate_x(out, x, dx); }
    void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
      DDP_BIND(auto, (out_orig, out_slack), eigen::split_at_row_mut(out, orig().control_dim(t)));
      DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, orig().control_dim(t)));
      DDP_BIND(auto, (du_orig, du_slack), eigen::split_at_row(du, orig().dcontrol_dim(t)));
      orig().integrate_u(out_orig, t, u_orig, du_orig);
      if (u_slack.rows() > 0) {
        DDP_ASSERT(u_slack.rows() == dstate_dim().value());
        out_slack = u_slack + du_slack;
      }
    }

    void d_integrate_x(out_x_mut<dims>, x_const<dims> x, dx_const<dims> dx) const;
    void d_integrate_x_dx(out_x_mut<dims>, x_const<dims> x, dx_const<dims> dx) const;
  };

  struct constraint_t {
    multiple_shooting_t const& m_parent;

    using scalar_t = multiple_shooting_t::scalar_t;
    using dims = multiple_shooting_t::eq_dims;

    using key = typename multiple_shooting_t::key;

    auto orig() const -> orig_constraint_t { return m_parent.m_prob.constraint(); }

    auto eq_idx() const -> eq_indexer_t { return indexing::row_concat(orig().eq_idx(), m_parent.m_slack_idx.clone()); }
    auto eq_dim(index_t t) const -> typename eq_indexer_t::row_kind {
      return eq_idx().rows(t) + m_parent.m_slack_idx.rows(t);
    }

    void integrate_x(x_mut<dims> out, x_const<dims> x, dx_const<dims> dx) const {
      m_parent.dynamics().integrate_x(out, x, dx);
    }
    void integrate_u(u_mut<dims> out, index_t t, u_const<dims> u, u_const<dims> du) const {
      m_parent.dynamics().integrate_u(out, t, u, du);
    }
    template <typename Out, typename In>
    void difference_out(Out out, In start, In finish) const {
      DDP_DEBUG_ASSERT_MSG_ALL_OF(          //
          ("", out.rows() == start.rows()), //
          ("", out.rows() == finish.rows()));
      out = finish - start;
    }

    auto first_order_deriv(    //
        out_x_mut<dims> out_x, //
        out_u_mut<dims> out_u, //
        out_mut<dims> out,     //
        index_t t,             //
        x_const<dims> x,       //
        u_const<dims> u,       //
        key k                  //
    ) const -> key {
      auto ne = orig().eq_dim(t);
      auto nu = m_parent.dynamics().orig().dcontrol_dim(t);

      DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, nu));
      DDP_BIND(auto, (out_x_orig, out_x_slack), eigen::split_at_row_mut(out_x, ne));
      DDP_BIND(
          auto,
          (out_u_orig_orig, out_u_orig_slack, out_u_slack_orig, out_u_slack_slack),
          eigen::split_at_mut(out_u, ne, nu));

      DDP_BIND(auto, (out_orig, out_slack), eigen::split_at_row_mut(out, ne));

      if (out_slack.rows() > 0) {
        out_slack = u_slack;
      }

      out_u_orig_slack.setZero();
      out_u_slack_orig.setZero();
      out_u_slack_slack.setIdentity();
      out_u_slack_slack *= m_parent.m_factor;

      out_x_slack.setZero();
      k = orig().first_order_deriv(
          eigen::into_view(out_x_orig),
          eigen::into_view(out_u_orig_orig),
          eigen::into_view(out_orig),
          t,
          x,
          u_orig,
          DDP_MOVE(k));
      return k;
    }

    auto second_order_deriv(  //
        out_xx_mut<dims> fxx, //
        out_ux_mut<dims> fux, //
        out_uu_mut<dims> fuu, //
        out_x_mut<dims> fx,   //
        out_u_mut<dims> fu,   //
        out_mut<dims> f,      //
        index_t t,            //
        x_const<dims> x,      //
        u_const<dims> u,      //
        key k                 //
    ) const -> key {
      return finite_diff_hessian_compute<constraint_t>{*this, orig().dynamics().second_order_finite_diff()}
          .second_order_deriv(fxx, fux, fuu, fx, fu, f, t, x, u, DDP_MOVE(k));
    }

    auto eval_to(out_mut<dims> out, index_t t, x_const<dims> x, u_const<dims> u, key k) const -> key {
      return m_parent.eval_eq_to(out, t, x, u, DDP_MOVE(k));
    }
  };

  auto dynamics() const -> dynamics_t { return {*this}; }
  auto constraint() const -> constraint_t { return {*this}; }

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t { return m_prob.state_indexer(begin, end); }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return m_prob.dstate_indexer(begin, end);
  }

  auto state_dim() const -> typename f_dims::x_t { return m_prob.state_dim(); }
  auto dstate_dim() const -> typename f_dims::dx_t { return m_prob.dstate_dim(); }
  auto control_dim(index_t t) const -> typename f_dims::u_t { return m_prob.control_dim(t) + m_slack_idx.rows(t); }
  auto dcontrol_dim(index_t t) const -> typename f_dims::du_t { return m_prob.dcontrol_dim(t) + m_slack_idx.rows(t); }

  void neutral_configuration(x_mut<f_dims> out) const { m_prob.neutral_configuration(out); }

  auto lf(x_const<f_dims> x) const -> scalar_t { return m_prob.lf(x); }
  auto l(index_t t, x_const<f_dims> x, u_const<f_dims> u) const -> scalar_t {
    DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, m_prob.control_dim(t)));
    (void)u_slack;
    return m_prob.l(t, x, u_orig);
  }
  void lf_second_order_deriv(xx_mut<f_dims> lxx, xt_mut<f_dims> lx, x_const<f_dims> x) const {
    m_prob.lf_second_order_deriv(lxx, lx, x);
  }
  void l_second_order_deriv(
      xx_mut<f_dims> lxx,
      ux_mut<f_dims> lux,
      uu_mut<f_dims> luu,
      xt_mut<f_dims> lx,
      ut_mut<f_dims> lu,
      index_t t,
      x_const<f_dims> x,
      u_const<f_dims> u) const {
    DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, m_prob.control_dim(t)));
    (void)u_slack;
    DDP_BIND(auto, (lu_orig, lu_slack), eigen::split_at_col_mut(lu, m_prob.control_dim(t)));
    DDP_BIND(auto, (lux_orig, lux_slack), eigen::split_at_row_mut(lux, m_prob.control_dim(t)));
    DDP_BIND(
        auto,
        (luu_orig_orig, luu_orig_slack, luu_slack_orig, luu_slack_slack),
        eigen::split_at_mut(luu, m_prob.control_dim(t), m_prob.control_dim(t)));

    lu_slack.setZero();
    luu_slack_slack.setZero();
    luu_slack_orig.setZero();
    luu_orig_slack.setZero();
    lux_slack.setZero();
    m_prob.l_second_order_deriv(lxx, lux_orig, luu_orig_orig, lx, lu_orig, t, x, u_orig);
  }
  void difference(dout_mut<f_dims> out, out_const<f_dims> start, out_const<f_dims> finish) const {
    m_prob.difference(out, start, finish);
  }
  void d_difference_dfinish(out_x_mut<f_dims> out, out_const<f_dims> start, out_const<f_dims> finish) const {
    m_prob.d_difference_dfinish(out, start, finish);
  }

  auto eval_f_to(out_mut<f_dims> x_out, index_t t, x_const<f_dims> x, u_const<f_dims> u, key k = {}) const -> key {
    if (!k) {
      k = dynamics().acquire_workspace();
    }
    DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, m_prob.control_dim(t)));

    if (u_slack.rows() == 0) {
      k = m_prob.dynamics().eval_to(x_out, t, x, u_orig, DDP_MOVE(k));
    } else {
      DDP_ASSERT(u_slack.rows() == dstate_dim().value());

      DDP_BIND(auto, (u_slack_x, u_slack_0), eigen::split_at_row(u_slack, dstate_dim()));
      DDP_ASSERT(u_slack_0.rows() == 0);

      auto _tmp = x_out.eval();
      auto tmp = eigen::as_mut_view(_tmp);

      k = m_prob.dynamics().eval_to(tmp, t, x, u_orig, DDP_MOVE(k));
      dynamics().integrate_x(x_out, eigen::as_const_view(tmp), u_slack_x);
    }
    return k;
  }

  auto eval_eq_to(out_mut<eq_dims> eq_out, index_t t, x_const<f_dims> x, u_const<f_dims> u, key k = {}) const -> key {
    if (!k) {
      k = dynamics().acquire_workspace();
    }

    DDP_BIND(auto, (u_orig, u_slack), eigen::split_at_row(u, m_prob.control_dim(t)));
    DDP_BIND(auto, (out_orig, out_slack), eigen::split_at_row_mut(eq_out, m_prob.constraint().eq_dim(t)));

    k = m_prob.eval_eq_to(eigen::into_view(out_orig), t, x, u_orig, DDP_MOVE(k));
    if (u_slack.rows() > 0) {
      DDP_ASSERT(u_slack.rows() == dstate_dim().value());
      out_slack = u_slack.operator*(m_factor);
    }
    return k;
  }

  using derivative_storage_t =
      ddp::derivative_storage_t<scalar_t, dcontrol_indexer_t, eq_indexer_t, state_indexer_t, dstate_indexer_t>;
  using trajectory_t = trajectory::trajectory_t<scalar_t, state_indexer_t, control_indexer_t>;

  void compute_derivatives(derivative_storage_t& derivs, trajectory_t const& traj) const {
    ddp::compute_derivatives(*this, derivs, traj);
  }

  auto name() const -> std::string {
    auto n = m_prob.name();
    return "multi_shooting_" + std::string{n.begin(), n.end()};
  }
  problem_t m_prob;
  slack_indexer_t m_slack_idx;
  scalar_t m_factor = 1;
};

template <typename Problem, typename Slack_Idx>
auto multi_shooting(Problem p, Slack_Idx idx, typename Problem::scalar_t factor = 1)
    -> multiple_shooting_t<Problem, Slack_Idx> {
  return {DDP_MOVE(p), DDP_MOVE(idx), factor};
}

} // namespace ddp

#endif /* end of include guard PROBLEM_HPP_SQUCCNMX */
