#ifndef DDP_PINOCCHIO_DERIVATIVE_STORAGE_HPP_LLE0WSK3S
#define DDP_PINOCCHIO_DERIVATIVE_STORAGE_HPP_LLE0WSK3S

#include "ddp/internal/second_order_finite_diff.hpp"
#include "ddp/internal/tensor.hpp"
#include "ddp/cost.hpp"
#include "ddp/dynamics.hpp"
#include "ddp/constraint.hpp"
#include "ddp/trajectory.hpp"

namespace ddp {
namespace internal {

template <
    typename Out,
    typename XX,
    typename UX,
    typename UU,
    typename DX,
    typename DU>
void add_second_order_term(
    Out&& out,
    XX const& xx,
    UX const& ux,
    UU const& uu,
    DX const& dx,
    DU const& du) {
  VEG_DEBUG_ASSERT_ALL_OF(
      out.cols() == 1,
      out.rows() == xx.outdim(),
      out.rows() == ux.outdim(),
      out.rows() == uu.outdim(),
      dx.rows() == xx.indiml(),
      dx.rows() == xx.indimr(),
      dx.rows() == ux.indimr(),
      du.rows() == uu.indiml(),
      du.rows() == uu.indimr(),
      du.rows() == ux.indiml());

  for (i64 j = 0; j < dx.rows(); ++j) {
    for (i64 i = 0; i < dx.rows(); ++i) {
      for (i64 k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * dx(i) * xx(k, i, j) * dx(j);
      }
    }
  }

  for (i64 j = 0; j < du.rows(); ++j) {
    for (i64 i = 0; i < du.rows(); ++i) {
      for (i64 k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * du(i) * uu(k, i, j) * du(j);
      }
    }
  }

  for (i64 j = 0; j < dx.rows(); ++j) {
    for (i64 i = 0; i < du.rows(); ++i) {
      for (i64 k = 0; k < out.rows(); ++k) {
        out(k) += du(i) * ux(k, i, j) * dx(j);
      }
    }
  }
}

#define DDP_ACCESS(name)                                                       \
  auto name(i64 t)& { return self.name[t]; }                                   \
  auto name(i64 t) const& { return self.name[t]; }                             \
  auto name(i64 t)&& = delete

template <typename T>
struct first_order_derivatives {
  using scalar = T;

  struct layout {
    std::vector<T> lfx;

    internal::mat_seq<T, colvec> lx;
    internal::mat_seq<T, colvec> lu;

    internal::mat_seq<T, colvec> f;
    internal::mat_seq<T, colmat> fx;
    internal::mat_seq<T, colmat> fu;

    internal::mat_seq<T, colvec> eq;
    internal::mat_seq<T, colmat> eqx;
    internal::mat_seq<T, colmat> equ;
  } self;

  auto lfx() const& { return eigen::slice_to_vec(self.lfx); }
  auto lfx() & { return eigen::slice_to_vec(self.lfx); }

  DDP_ACCESS(lx);
  DDP_ACCESS(lu);

  DDP_ACCESS(f);
  DDP_ACCESS(fx);
  DDP_ACCESS(fu);

  DDP_ACCESS(eq);
  DDP_ACCESS(eqx);
  DDP_ACCESS(equ);
};

template <typename T>
struct second_order_derivatives : first_order_derivatives<T> {
  using scalar = T;

  struct layout {
    std::vector<T> lfxx;

    internal::mat_seq<T, colmat> lxx;
    internal::mat_seq<T, colmat> lux;
    internal::mat_seq<T, colmat> luu;

    internal::tensor_seq<T> fxx;
    internal::tensor_seq<T> fux;
    internal::tensor_seq<T> fuu;

    internal::tensor_seq<T> eqxx;
    internal::tensor_seq<T> equx;
    internal::tensor_seq<T> equu;
  } self;

  second_order_derivatives(first_order_derivatives<T> base, layout self_)
      : first_order_derivatives<T>(VEG_FWD(base)), self(VEG_FWD(self_)) {}

  auto lfxx() const& {
    return eigen::slice_to_mat(
        self.lfxx, this->lfx().rows(), this->lfx().rows());
  }
  auto lfxx() & {
    return eigen::slice_to_mat(
        self.lfxx, this->lfx().rows(), this->lfx().rows());
  }

  DDP_ACCESS(lxx);
  DDP_ACCESS(lux);
  DDP_ACCESS(luu);

  DDP_ACCESS(fxx);
  DDP_ACCESS(fux);
  DDP_ACCESS(fuu);

  DDP_ACCESS(eqxx);
  DDP_ACCESS(equx);
  DDP_ACCESS(equu);
};

#undef DDP_ACCESS

template <typename Cost, typename Dynamics, typename Constraint>
auto compute_first_derivatives_req(
    Cost const& cost, Dynamics const& dynamics, Constraint const& constraint)
    -> mem_req {
  return mem_req::max_of({
      cost.d_eval_to_req(),
      dynamics.d_eval_to_req(),
      constraint.d_eval_to_req(),
  });
}

// single threaded
template <
    typename T,
    typename Cost,
    typename Dynamics,
    typename Constraint,
    typename Traj>
void compute_first_derivatives(
    first_order_derivatives<T>& derivs,
    Cost const& cost,
    Dynamics const& dynamics,
    Constraint const& constraint,
    Traj const& traj,
    veg::dynamic_stack_view stack) {

  cost.d_eval_final_to(derivs.lfx(), traj.x_f(), stack);

  i64 begin = derivs.self.lx.index_begin();
  i64 end = derivs.self.lx.index_end();

  for (i64 t = begin; t < end; ++t) {
    auto k = dynamics.acquire_workspace();
    VEG_BIND(auto, (x, u), traj[t]);

    cost.d_eval_to(derivs.lx(t), derivs.lu(t), t, x, u, stack);

    k = dynamics.d_eval_to(
        derivs.fx(t), derivs.fu(t), derivs.f(t), t, x, u, VEG_FWD(k), stack);

    k = constraint.d_eval_to(
        derivs.eqx(t), derivs.equ(t), derivs.eq(t), t, x, u, VEG_FWD(k), stack);
  }
}

#ifdef DDP_OPENMP

static constexpr i64 max_threads = 16;
inline auto n_threads() -> i64 {
  return veg::meta::min2(max_threads, veg::narrow<i64>(omp_get_num_procs()));
}

#else

inline auto n_threads() -> i64 {
  return 1;
}

#endif

template <typename Cost, typename Dynamics, typename Constraint>
auto compute_second_derivatives_req(
    Cost const& cost, Dynamics const& dynamics, Constraint const& constraint)
    -> mem_req {
  mem_req single_thread = mem_req::max_of({
      cost.d_eval_to_req(),
      cost.dd_eval_to_req(),
      ddp::second_order_deriv_2_req(dynamics),
      ddp::second_order_deriv_2_req(constraint),
  });
  i64 n = internal::n_threads();
  return {single_thread.align, single_thread.size * n};
}

// multi threaded
template <
    typename T,
    typename Cost,
    typename Dynamics,
    typename Constraint,
    typename Traj>
void compute_second_derivatives(
    second_order_derivatives<T>& derivs,
    Cost const& cost,
    Dynamics const& dynamics,
    Constraint const& constraint,
    Traj const& traj,
    veg::dynamic_stack_view stack) {

  cost.d_eval_final_to(derivs.lfx(), traj.x_f(), stack);
  auto const n_threads = internal::n_threads();

#ifdef DDP_OPENMP
#pragma omp parallel default(none) shared(derivs) shared(cost)                 \
    shared(dynamics) shared(constraint) shared(traj) num_threads(n_threads)
#endif
  {

    first_order_derivatives<T>& _1 = derivs;

    i64 begin = _1.self.lx.index_begin();
    i64 end = _1.self.lx.index_end();
    auto k = dynamics.acquire_workspace();

    veg::dynamic_stack_view thread_stack = {{nullptr, 0}};

#ifdef DDP_OPENMP
#pragma omp critical
#endif
    {
      struct like_T {
        alignas(T) unsigned char buf[sizeof(T)];
      };
      auto buf = stack
                     .make_new(
                         veg::tag<like_T>,
                         stack.remaining_bytes() / veg::narrow<i64>(sizeof(T)) /
                             n_threads)
                     .unwrap();

      thread_stack = {veg::make::slice(buf)};
    }

#ifdef DDP_OPENMP
#pragma omp for
#endif
    for (usize i = 0; i < veg::narrow<usize>(end - begin); ++i) {
      i64 t = begin + veg::narrow<i64>(i);
      VEG_BIND(auto, (x, u), traj[t]);

      cost.d_eval_to(derivs.lx(t), derivs.lu(t), t, x, u, thread_stack);
      cost.dd_eval_to(
          derivs.lxx(t), derivs.lux(t), derivs.luu(t), t, x, u, thread_stack);

      k = ddp::second_order_deriv_2(
          dynamics,
          derivs.fxx(t),
          derivs.fux(t),
          derivs.fuu(t),
          derivs.fx(t),
          derivs.fu(t),
          derivs.f(t),
          t,
          x,
          u,
          VEG_FWD(k),
          thread_stack);

      k = ddp::second_order_deriv_2(
          constraint,
          derivs.eqxx(t),
          derivs.equx(t),
          derivs.equu(t),
          derivs.eqx(t),
          derivs.equ(t),
          derivs.eq(t),
          t,
          x,
          u,
          VEG_FWD(k),
          thread_stack);
      {
        auto msg = fmt::format(
            "  checking all derivatives from thread {}", omp_get_thread_num());
        VEG_ASSERT_ALL_OF( //
            !derivs.eq(t).hasNaN(),
            !derivs.eqx(t).hasNaN(),
            !derivs.equ(t).hasNaN(),
            !derivs.eqxx(t).has_nan(),
            !derivs.equx(t).has_nan(),
            !derivs.equu(t).has_nan(),
            !derivs.f(t).hasNaN(),
            !derivs.fx(t).hasNaN(),
            !derivs.fu(t).hasNaN(),
            !derivs.fxx(t).has_nan(),
            !derivs.fux(t).has_nan(),
            !derivs.fuu(t).has_nan());

        auto ne = derivs.eq(t).rows();
        auto ndx = derivs.fu(t).rows();
        auto ndu = derivs.fu(t).cols();

        T l{};
        T l_{};
        auto f = derivs.f(t).eval();
        auto f_ = derivs.f(t).eval();
        auto eq = derivs.eq(t).eval();
        auto eq_ = derivs.eq(t).eval();

        T _dl1{};
        T _ddl1{};
        T _dddl1{};
        T _dl2{};
        T _ddl2{};
        T _dddl2{};

        using vec = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
        auto _df1 = vec(ndx);
        auto _ddf1 = vec(ndx);
        auto _dddf1 = vec(ndx);
        auto _df2 = vec(ndx);
        auto _ddf2 = vec(ndx);
        auto _dddf2 = vec(ndx);

        auto _deq1 = vec(ne);
        auto _ddeq1 = vec(ne);
        auto _dddeq1 = vec(ne);
        auto _deq2 = vec(ne);
        auto _ddeq2 = vec(ne);
        auto _dddeq2 = vec(ne);

        auto* dl = &_dl1;
        auto* df = &_df1;
        auto* deq = &_deq1;

        auto* ddl = &_ddl1;
        auto* ddf = &_ddf1;
        auto* ddeq = &_ddeq1;

        auto* dddl = &_dddl1;
        auto* dddf = &_dddf1;
        auto* dddeq = &_dddeq1;

        auto dx_ = vec(ndx);
        dx_.setRandom();

        auto du_ = vec(ndu);
        du_.setRandom();

        auto finite_diff_err = [&](T eps_x, T eps_u) {
          auto dx = (dx_.operator*(eps_x)).eval();
          auto du = (du_.operator*(eps_u)).eval();

          auto x_ = x.eval();
          auto u_ = u.eval();

          dynamics.state_space().integrate(
              eigen::as_mut(x_),
              t,
              eigen::as_const(x),
              eigen::as_const(dx),
              thread_stack);
          dynamics.control_space().integrate(
              eigen::as_mut(u_),
              t,
              eigen::as_const(u),
              eigen::as_const(du),
              thread_stack);

          l = cost.eval(t, x, u, thread_stack);
          k = dynamics.eval_to(
              eigen::as_mut(f), t, x, u, VEG_FWD(k), thread_stack);
          k = constraint.eval_to(
              eigen::as_mut(eq), t, x, u, VEG_FWD(k), thread_stack);

          l_ = cost.eval(
              t, eigen::as_const(x_), eigen::as_const(u_), thread_stack);
          k = dynamics.eval_to(
              eigen::as_mut(f_),
              t,
              eigen::as_const(x_),
              eigen::as_const(u_),
              VEG_FWD(k),
              thread_stack);
          k = constraint.eval_to(
              eigen::as_mut(eq_),
              t,
              eigen::as_const(x_),
              eigen::as_const(u_),
              VEG_FWD(k),
              thread_stack);

          *dl = l_ - l;
          dynamics.output_space().difference(
              eigen::as_mut(*df),
              t + 1,
              eigen::as_const(f),
              eigen::as_const(f_),
              thread_stack);
          constraint.output_space().difference(
              eigen::as_mut(*deq),
              t,
              eigen::as_const(eq),
              eigen::as_const(eq_),
              thread_stack);

          *ddl = *dl -
                 (derivs.lx(t).transpose() * dx + derivs.lu(t).transpose() * du)
                     .value();
          *ddf = *df - (derivs.fx(t) * dx + derivs.fu(t) * du);
          *ddeq = *deq - (derivs.eqx(t) * dx + derivs.equ(t) * du);

          *dddl = *ddl - (0.5 * dx.transpose() * derivs.lxx(t) * dx   //
                          + 0.5 * du.transpose() * derivs.luu(t) * du //
                          + du.transpose() * derivs.lux(t) * dx)
                             .value();
          *dddf = -*ddf;
          *dddeq = -*ddeq;

          auto dddf2 = dddf->eval();
          dddf2.setZero();

          internal::add_second_order_term(
              dddf2, derivs.fxx(t), derivs.fux(t), derivs.fuu(t), dx, du);

          *dddf += dddf2;

          internal::add_second_order_term(
              *dddeq, derivs.eqxx(t), derivs.equx(t), derivs.equu(t), dx, du);

          *dddf = -*dddf;
          *dddeq = -*dddeq;
        };

        T eps_x = 1e-30;
        T eps_u = 1e-30;

        T eps_factor = 0.1;

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

        auto eps = pow(std::numeric_limits<T>::epsilon(), 0.5);

        VEG_EXPECT_ALL_OF_ELSE(
            (fmt::format("at t = {}\n{}\n{}", t, dl1, dl2),
             (fabs(dl1) <= eps || fabs(dl2) <= 2 * eps_factor * fabs(dl1))),
            (fmt::format("at t = {}\n{}\n{}", t, ddl1, ddl2),
             (fabs(ddl1) <= eps ||
              fabs(ddl2) <= 2 * pow(eps_factor, 2) * fabs(ddl1))),
            (fmt::format("at t = {}\n{}\n{}", t, dddl1, dddl2),
             (fabs(dddl1) <= eps ||
              fabs(dddl2) <= 2 * pow(eps_factor, 3) * fabs(dddl1))));

        for (i64 j = 0; j < df1.size(); ++j) {
          VEG_EXPECT_ALL_OF_ELSE(
              (fmt::format("at t = {}, j = {}\n{}\n{}", t, j, df1[j], df2[j]),
               (fabs(df1[j]) <= eps ||
                fabs(df2[j]) <= 2 * eps_factor * fabs(df1[j]))),
              (fmt::format("at t = {}, j = {}\n{}\n{}", t, j, ddf1[j], ddf2[j]),
               (fabs(ddf1[j]) <= eps ||
                fabs(ddf2[j]) <= 2 * pow(eps_factor, 2) * fabs(ddf1[j]))),
              (fmt::format(
                   "at t = {}, j = {}\n{}\n{}", t, j, dddf1[j], dddf2[j]),
               (fabs(dddf1[j]) <= eps ||
                fabs(dddf2[j]) <= 2 * pow(eps_factor, 3) * fabs(dddf1[j]))));
        }

        for (i64 j = 0; j < deq1.size(); ++j) {
          VEG_EXPECT_ALL_OF_ELSE(
              (fmt::format("at t = {}\n{}\n{}", t, deq1[j], deq2[j]),
               (fabs(deq1[j]) <= eps ||
                fabs(deq2[j]) <= 2 * eps_factor * fabs(deq1[j]))),
              (fmt::format("at t = {}\n{}\n{}", t, ddeq1[j], ddeq2[j]),
               (fabs(ddeq1[j]) <= eps ||
                fabs(ddeq2[j]) <= 2 * pow(eps_factor, 2) * fabs(ddeq1[j]))),
              (fmt::format("at t = {}\n{}\n{}", t, dddeq1[j], dddeq2[j]),
               (fabs(dddeq1[j]) <= eps ||
                fabs(dddeq2[j]) <= 2 * pow(eps_factor, 3) * fabs(dddeq1[j]))));
        }
      }
    }
  }
}

} // namespace internal
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_DERIVATIVE_STORAGE_HPP_LLE0WSK3S  \
        */
