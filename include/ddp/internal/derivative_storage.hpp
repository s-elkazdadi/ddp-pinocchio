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
  mem_req single_thread = mem_req::sum_of({
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
      auto buf = stack
                     .make_new(
                         veg::tag<double>,
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

      cost.dd_eval_to(
          derivs.lxx(t), derivs.lux(t), derivs.luu(t), t, x, u, thread_stack);

      k = second_order_deriv_2(
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

      k = second_order_deriv_2(
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
    }
  }
}

// FIXME
VEG_INSTANTIATE_CLASS(first_order_derivatives, double);
VEG_INSTANTIATE_CLASS(second_order_derivatives, double);

VEG_INSTANTIATE(
    compute_first_derivatives_req,
    quadratic_cost_fixed_size<double>,
    pinocchio_dynamics_free<double>,
    config_constraint<
        pinocchio_dynamics_free<double>,
        veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>>);

VEG_INSTANTIATE(
    compute_first_derivatives,
    first_order_derivatives<double>&,
    quadratic_cost_fixed_size<double>,
    pinocchio_dynamics_free<double>,
    config_constraint<
        pinocchio_dynamics_free<double>,
        veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>>,
    trajectory<double>,
    veg::dynamic_stack_view);

VEG_INSTANTIATE(
    compute_second_derivatives,
    second_order_derivatives<double>&,
    quadratic_cost_fixed_size<double>,
    pinocchio_dynamics_free<double>,
    config_constraint<
        pinocchio_dynamics_free<double>,
        veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>>,
    trajectory<double>,
    veg::dynamic_stack_view);

} // namespace internal
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_DERIVATIVE_STORAGE_HPP_LLE0WSK3S  \
        */
