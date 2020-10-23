#ifndef DDP_IMPL_HPP_UBVAKU5V
#define DDP_IMPL_HPP_UBVAKU5V

#include "ddp/ddp.hpp"
#include <Eigen/Cholesky>

namespace ddp {

template <typename Problem>
template <method M>
void ddp_solver_t<Problem>::
    // clang-format off
  backward_pass(
      control_feedback_t&                             ctrl_fb,
      regularization_t<scalar_t>&                     regularization,
      scalar_t&                                       mu,
      trajectory_t const&                             current_traj,
      typename multiplier_sequence<M>::type const&    mults,
      derivative_storage_t const&                     derivatives
  ) const
{
  // clang-format on
  bool success = false;

  // TODO preallocate V_{x,xx}, Q_{x,u,xx,ux,uu}
  while (not success) {
    auto V_xx = derivatives.lfxx.eval();
    auto V_x = derivatives.lfx.transpose().eval();
    auto const v_x = eigen::as_const_view(V_x);

    scalar_t expected_decrease = 0;

    for (auto zipped :             //
         ranges::reverse(          //
             ranges::zip(          //
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

      auto const tmp_ = (pe + eq.val.operator*(mu)).eval();
      auto const tmp2_ = (pe_x + eq.x.operator*(mu)).eval();

      auto const tmp = eigen::as_const_view(tmp_);
      auto const tmp2 = eigen::as_const_view(tmp2_);

      bool const has_eq = tmp.rows() > 0;

      {
        using std::isfinite;
        DDP_ASSERT_MSG_ALL_OF(("", isfinite(mu)), ("", not pe.hasNaN()), ("", not pe_x.hasNaN()));
      }

      // clang-format off
      auto Q_x         = l.x.transpose().eval();                  auto Q_x_v = eigen::as_mut_view(Q_x);
      Q_x_v.noalias() += f.x.transpose() * V_x;
      if (has_eq) {
        Q_x_v.noalias() += eq.x.transpose() * tmp;
        Q_x_v.noalias() += pe_x.transpose() * eq.val;
      }

      auto Q_u         = l.u.transpose().eval();                  auto Q_u_v = eigen::as_mut_view(Q_u);
      Q_u_v.noalias() += f.u.transpose() * V_x;
      if (has_eq) {
        Q_u_v.noalias() += eq.u.transpose() * tmp;
      }

      auto Q_xx         = l.xx.eval();                            auto Q_xx_v = eigen::as_mut_view(Q_xx);
      Q_xx_v.noalias() += f.x.transpose() * V_xx * f.x;
      if (has_eq) {
        Q_xx_v.noalias() += eq.x.transpose() * tmp2;
        Q_xx_v.noalias() += pe_x.transpose() * eq.x;
        eq.xx.noalias_contract_add_outdim(Q_xx_v, tmp);
      }
      f .xx.noalias_contract_add_outdim(Q_xx_v, v_x);

      auto Q_uu         = l.uu.eval();                            auto Q_uu_v = eigen::as_mut_view(Q_uu);
      Q_uu_v.noalias() += f.u.transpose() * V_xx * f.u;
      if (has_eq) {
        Q_uu_v.noalias() +=  (eq.u.transpose() * eq.u).operator*(mu); // *see below for reason
        eq.uu.noalias_contract_add_outdim(Q_uu_v, tmp);
      }
      f .uu.noalias_contract_add_outdim(Q_uu_v, v_x);

      auto Q_ux         = l.ux.eval();                            auto Q_ux_v = eigen::as_mut_view(Q_ux);
      Q_ux_v.noalias() += f.u.transpose() * V_xx * f.x;
      if (has_eq) {
        Q_ux_v.noalias() +=  eq.u.transpose() * tmp2;
        eq.ux.noalias_contract_add_outdim(Q_ux_v, tmp);
      }
      f .ux.noalias_contract_add_outdim(Q_ux_v, v_x);
      // clang-format on

      /*
       * when scalar_t is boost::multiprecision::big_float
       * and du has size 1 at compile time
       *
       * => eq.u.transpose() * eq.u() has shape [1, 1]
       * => eq.u.transpose() * eq.u() is convertible to scalar_t
       * => mu * (eq.u.transpose() * eq.u())
       *     --^--
       *  boost::multiprecision::operator* converts rhs to scalar_t
       * => results in scalar_t instead of matrix
       */

      auto I_u = decltype(Q_uu)::Identity(Q_uu.rows(), Q_uu.rows());

      auto const fact = ((Q_uu + *regularization * I_u).eval()).llt();
      if (fact.info() == Eigen::NumericalIssue) {
        regularization.increase_reg();
        break;
      }

      u_fb.origin() = xu.x();
      u_fb.val() = fact.solve(-Q_u);
      u_fb.jac() = fact.solve(-Q_ux);

      auto const k = u_fb.val();
      auto const K = u_fb.jac();

      expected_decrease += (0.5 * k.transpose() * Q_uu * k).value();

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
}

} // namespace ddp

#endif /* end of include guard DDP_IMPL_HPP_UBVAKU5V */
