#ifndef DDP_FWD_TCC_O5KLTLOB
#define DDP_FWD_TCC_O5KLTLOB

#include "ddp/ddp.hpp"

namespace ddp {

// clang-format off
template <typename Problem>
template <method M>
auto ddp_solver_t<Problem>::
  forward_pass(
      trajectory_t&                                   new_traj_storage,
      trajectory_t const&                             reference_traj,
      typename multiplier_sequence<M>::type const&    old_mults,
      control_feedback_t const&                       feedback,
      scalar_t                                        mu,
      bool                                            do_linesearch
  ) const -> scalar_t {
  // clang-format on

  dyn_vec_t costs_old_traj;
  dyn_vec_t costs_new_traj;
  if (do_linesearch) {
    costs_old_traj.resize(index_end() - index_begin() + 1);
    costs_new_traj.resize(index_end() - index_begin() + 1);
    cost_seq_aug(eigen::as_mut_view(costs_old_traj), reference_traj, old_mults, mu);
  }

  scalar_t step = 1;
  bool success = false;

  auto tmp = eigen::make_matrix<scalar_t>(prob.dstate_dim());

  while (not success) {
    if (step < 1e-10) {
      break;
    }

    for (auto zipped : ddp::ranges::zip(new_traj_storage, reference_traj, feedback)) {
      DDP_BIND(auto&&, (xu_new, xu_old, fb), zipped);
      index_t t = xu_new.current_index();

      auto xu_new_c = xu_new.as_const();

      prob.difference(eigen::as_mut_view(tmp), xu_old.x(), xu_new_c.x());

      xu_new.u() = xu_old.u()        //
                   + step * fb.val() //
                   + fb.jac() * tmp;
      prob.eval_f_to(xu_new.x_next(), t, xu_new_c.x(), xu_new_c.u());
    }

    if (do_linesearch) {
      cost_seq_aug(eigen::as_mut_view(costs_new_traj), new_traj_storage, old_mults, mu);

      if ((costs_new_traj - costs_old_traj).sum() <= 0) {
        success = true;
      } else {
        step *= 0.5;
      }
    } else {
      success = true;
    }
  }

  return step;
}

} // namespace ddp

#endif /* end of include guard DDP_FWD_TCC_O5KLTLOB */
