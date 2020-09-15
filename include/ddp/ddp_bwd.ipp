#ifndef DDP_IMPL_HPP_UBVAKU5V
#define DDP_IMPL_HPP_UBVAKU5V

#include "ddp/ddp.hpp"
#include <Eigen/Cholesky>

namespace ddp {

template <typename Problem>
template <method M>
auto ddp_solver_t<Problem>::
    // clang-format off
  backward_pass(
      trajectory_t const&                             current_traj,
      typename multiplier_sequence<M>::type const&    mults,
      scalar_t                                        regularization,
      scalar_t                                        mu,
      derivative_storage_t const&                     derivatives
  ) const -> backward_pass_result_t<M>
{
  // clang-format on
  bool success = false;
  auto ctrl_fb = control_feedback_t{u_idx, model};

  // TODO preallocate V_{x,xx}, Q_{x,u,xx,ux,uu}
  while (not success) {
    auto V_xx = derivatives.lfxx.eval();
    auto V_x = derivatives.lfx.transpose().eval();
    auto const v_x = eigen::as_const_view(V_x);

    for (auto zipped :    //
         ranges::reverse( //
             ranges::zip(
                 current_traj,     //
                 derivatives.l(),  //
                 derivatives.f(),  //
                 derivatives.eq(), //
                 mults.eq,         //
                 ctrl_fb))) {
      DDP_BIND(auto&&, (xu, l, f, eq, eq_mult, u_fb), zipped);
      index_t t = xu.current_index();

      auto const pe = eq_mult.val();
      auto const pe_x = eq_mult.jac();

      auto const tmp_ = (pe + mu * eq.val).eval();
      auto const tmp2_ = (pe_x + mu * eq.x).eval();

      auto const tmp = eigen::as_const_view(tmp_);
      auto const tmp2 = eigen::as_const_view(tmp2_);

      {
        using std::isfinite;
        assert(isfinite(mu));
        assert(not pe.hasNaN());
        assert(not pe_x.hasNaN());
      }

      // clang-format off
      auto Q_x         = l.x.transpose().eval();                  auto Q_x_v = eigen::as_mut_view(Q_x);
      Q_x_v.noalias() += f.x.transpose() * V_x;
      Q_x_v.noalias() += eq.x.transpose() * tmp;
      Q_x_v.noalias() += pe_x.transpose() * eq.val;

      auto Q_u         = l.u.transpose().eval();                  auto Q_u_v = eigen::as_mut_view(Q_u);
      Q_u_v.noalias() += f.u.transpose() * V_x;
      Q_u_v.noalias() += eq.u.transpose() * tmp;

      auto Q_xx         = l.xx.eval();                            auto Q_xx_v = eigen::as_mut_view(Q_xx);
      Q_xx_v.noalias() += f.x.transpose() * V_xx * f.x;
      Q_xx_v.noalias() += eq.x.transpose() * tmp2;
      Q_xx_v.noalias() += pe_x.transpose() * eq.x;
      eq.xx.noalias_contract_add_outdim(Q_xx_v, tmp);
      f .xx.noalias_contract_add_outdim(Q_xx_v, v_x);

      auto Q_uu         = l.uu.eval();                            auto Q_uu_v = eigen::as_mut_view(Q_uu);
      Q_uu_v.noalias() += f.u.transpose() * V_xx * f.u;
      Q_uu_v.noalias() += mu * (eq.u.transpose() * eq.u);
      eq.uu.noalias_contract_add_outdim(Q_uu_v, tmp);
      f .uu.noalias_contract_add_outdim(Q_uu_v, v_x);

      auto Q_ux         = l.ux.eval();                            auto Q_ux_v = eigen::as_mut_view(Q_ux);
      Q_ux_v.noalias() += f.u.transpose() * V_xx * f.x;
      Q_ux_v.noalias() +=  eq.u.transpose() * tmp2;
      eq.ux.noalias_contract_add_outdim(Q_ux_v, tmp);
      f .ux.noalias_contract_add_outdim(Q_ux_v, v_x);
      // clang-format on

      auto I_u = decltype(Q_uu)::Identity(Q_uu.rows(), Q_uu.rows());

      auto const fact = ((Q_uu + regularization * I_u).eval()).llt();
      if (fact.info() == Eigen::NumericalIssue) {
        if (regularization < mu) {
          regularization = mu;
        }
        mu *= 2;
        regularization *= 2;
        fmt::print("t  : {}\n", t);
        fmt::print("mu : {}\n", mu);
        fmt::print("reg: {}\n", regularization);
        fmt::print("Quu:\n{}\n\n", Q_uu);
        break;
      }

      u_fb.origin() = xu.x();
      u_fb.val() = fact.solve(-Q_u);
      u_fb.jac() = fact.solve(-Q_ux);

      auto const k = u_fb.val();
      auto const K = u_fb.jac();

      // clang-format off
      V_x            = Q_x;
      V_x.noalias() += Q_ux.transpose() * k;

      V_xx            = Q_xx;
      V_xx.noalias() += Q_ux.transpose() * K;
      // clang-format on

      if (t == 0) {
        success = true;
      }
    }
  }
  return {DDP_MOVE(ctrl_fb), DDP_MOVE(mu), DDP_MOVE(regularization)};
}

} // namespace ddp

#endif /* end of include guard DDP_IMPL_HPP_UBVAKU5V */
