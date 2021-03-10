#ifndef DDP_PINOCCHIO_DDP_HPP_CAAKIEJQS
#define DDP_PINOCCHIO_DDP_HPP_CAAKIEJQS

#include "ddp/internal/derivative_storage.hpp"
#include "ddp/internal/function_models.hpp"
#include "ddp/space.hpp"
#include "ddp/internal/idx_transforms.hpp"
#include <Eigen/Cholesky>

namespace ddp {

enum struct method {
  constant_multipliers,
  affine_multipliers,
};

template <typename Scalar>
struct regularization {

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

template <typename Dynamics, typename Cost, typename Constraint>
struct ddp {

  using scalar = typename Dynamics::scalar;
  using trajectory = ::ddp::trajectory<scalar>;
  using key = decltype(__VEG_DECLVAL(Dynamics const&).acquire_workspace());

  using state_space = decltype(__VEG_DECLVAL(Constraint const&).state_space());
  using control_space =
      decltype(__VEG_DECLVAL(Constraint const&).control_space());
  using constraint_output_space =
      decltype(__VEG_DECLVAL(Constraint const&).output_space());

  template <method K, typename = void>
  struct multiplier_sequence;
  template <typename Dummy>
  struct multiplier_sequence<method::constant_multipliers, Dummy> {
    using eq_type =
        internal::constant_function_seq<scalar, constraint_output_space>;

    struct type {
      eq_type eq;
    };

    static auto zero(
        state_space x_space,
        constraint_output_space eq_space,
        trajectory const& traj) -> type {

      (void)traj, (void)x_space;
      auto begin = traj.index_begin();
      auto end = traj.index_end();
      auto multipliers = type{eq_type{begin, end, eq_space}};
      for (i64 t = begin; t < end; ++t) {
        multipliers.eq.self.val[t].setZero();
      }
      return multipliers;
    }
  };
  template <typename Dummy>
  struct multiplier_sequence<method::affine_multipliers, Dummy> {
    using eq_type = internal::
        affine_function_seq<scalar, state_space, constraint_output_space>;

    struct type {
      eq_type eq;
    };

    static auto zero(
        state_space x_space,
        constraint_output_space eq_space,
        trajectory const& traj) -> type {

      auto begin = traj.index_begin();
      auto end = traj.index_end();
      auto multipliers = type{eq_type{begin, end, x_space, eq_space}};

      for (i64 t = begin; t < end; ++t) {
        auto& fn = multipliers.eq.self;
        VEG_ASSERT_ALL_OF(!fn.val.self.data.empty());
        fn.val[t].setZero();
        fn.jac[t].setZero();
        eigen::assign(fn.origin[t], traj.self.x[t]);
      }
      return multipliers;
    }
  };

  template <method M>
  auto zero_multipliers(trajectory const& traj) const ->
      typename multiplier_sequence<M>::type {
    return multiplier_sequence<M>::zero(
        self.dynamics.state_space(), self.constraint.output_space(), traj);
  }

  template <typename Control_Gen>
  auto make_trajectory(
      i64 begin,
      i64 end,
      view<scalar const, colvec> x_init,
      Control_Gen it_u) const -> trajectory {

    trajectory traj{
        ::ddp::space_to_idx(self.dynamics.state_space(), begin, end + 1),
        ::ddp::space_to_idx(self.dynamics.control_space(), begin, end),
    };

    auto stack_storage = std::vector<unsigned char>(veg::narrow<veg::usize>(
        self.dynamics.eval_to_req().size + self.dynamics.eval_to_req().align));

    auto stack = veg::dynamic_stack_view(
        {stack_storage.data(), veg::narrow<veg::i64>(stack_storage.size())});
    auto k = self.dynamics.acquire_workspace();
    traj[begin][0_c] = x_init;
    for (i64 t = begin; t < end; ++t) {
      VEG_BIND(auto, (x, u), traj[t]);
      auto x_next = traj.x(t + 1);

      eigen::assign(u, it_u(eigen::as_const(x)));
      k = self.dynamics.eval_to(
          x_next, t, eigen::as_const(x), eigen::as_const(u), VEG_FWD(k), stack);
    }
    return traj;
  }

  using control_feedback =
      internal::affine_function_seq<scalar, state_space, control_space>;
  using derivative_storage = internal::second_order_derivatives<scalar>;

  template <method M>
  void solve(trajectory traj) {

    auto mult = zero_multipliers<M>(traj);
    auto reg = regularization<scalar>{0, 1, 2, 1e-5};
    auto ctrl = control_feedback(
        traj.index_begin(),
        traj.index_end(),
        self.dynamics.state_space(),
        self.dynamics.control_space());

    i64 begin = traj.index_begin();
    i64 end = traj.index_end();

    using vecseq = internal::mat_seq<scalar, colvec>;
    using matseq = internal::mat_seq<scalar, colmat>;

    idx::idx<colvec> idxx =
        ::ddp::space_to_idx(self.constraint.state_space(), begin, end);
    idx::idx<colvec> idxu =
        ::ddp::space_to_idx(self.constraint.control_space(), begin, end);
    idx::idx<colvec> idxe =
        ::ddp::space_to_idx(self.constraint.output_space(), begin, end);

    auto x = idxx.as_view();
    auto u = idxu.as_view();
    auto e = idxe.as_view();

    auto prod2 = [&](auto l, auto r) { return matseq(idx::prod_idx(l, r)); };
    auto prod3 = [&](auto o, auto l, auto r) {
      return internal::tensor_seq<scalar>(
          idx::tensor_idx(begin, end, [&](i64 t) -> idx::tensor_dims {
            return {o.rows(t), l.rows(t), r.rows(t)};
          }));
    };

    auto derivs = internal::second_order_derivatives<scalar>{
        {
            std::vector<scalar>(veg::narrow<veg::usize>(traj.x_f().rows())),

            vecseq(idxx),
            vecseq(idxu),

            vecseq(idxx),
            prod2(x, x),
            prod2(x, u),

            vecseq(idxe),
            prod2(e, x),
            prod2(e, u),
        },

        {
            std::vector<scalar>(
                veg::narrow<veg::usize>(traj.x_f().rows() * traj.x_f().rows())),

            prod2(x, x),
            prod2(u, x),
            prod2(u, u),

            prod3(x, x, x),
            prod3(x, u, x),
            prod3(x, u, u),

            prod3(e, x, x),
            prod3(e, u, x),
            prod3(e, u, u),
        }};

    auto traj2 = traj;

    scalar mu = 1e8;

    mem_req req = internal::compute_second_derivatives_req(
        self.cost, self.dynamics, self.constraint);
    auto stack_storage = std::vector<unsigned char>(
        ((1U << 20U) + 4 * veg::narrow<usize>(req.size)) * sizeof(scalar) +
        veg::narrow<usize>(req.align));

    veg::dynamic_stack_view stack(veg::slice<void>{stack_storage});
    for (i64 i = 0; i < 3000; ++i) {

      internal::compute_second_derivatives(
          derivs, self.cost, self.dynamics, self.constraint, traj, stack);

      this->bwd_pass<M>(ctrl, reg, scalar(1), traj, mult, derivs, stack);
      auto res = this->fwd_pass<M>(
          traj2,
          traj,
          mult,
          ctrl,
          mu,
          self.dynamics.acquire_workspace(),
          stack,
          true);

      std::swap(traj, traj2);
    }
  }

  template <method M>
  auto bwd_pass(
      control_feedback& control_feedback,
      regularization<scalar>& reg,
      scalar mu,
      trajectory const& current_traj,
      typename multiplier_sequence<M>::type const& mults,
      derivative_storage const& derivatives,
      veg::dynamic_stack_view stack) const -> scalar {

    (void)stack;

    // clang-format on
    bool success = false;

    scalar expected_decrease = 0;
    // TODO preallocate V_{x,xx}, Q_{x,u,xx,ux,uu}
    while (!success) {
      auto V_xx = derivatives.lfxx().eval();
      auto V_x = derivatives.lfx().eval();
      auto const v_x = eigen::as_const(V_x);

      expected_decrease = 0;

      for (i64 t = current_traj.index_end() - 1;
           t >= current_traj.index_begin();
           --t) {
        auto xu = current_traj[t];

        auto lx = derivatives.lx(t);
        auto lu = derivatives.lu(t);
        auto lxx = derivatives.lxx(t);
        auto lux = derivatives.lux(t);
        auto luu = derivatives.luu(t);

        auto fx = derivatives.fx(t);
        auto fu = derivatives.fu(t);
        auto fxx = derivatives.fxx(t);
        auto fux = derivatives.fux(t);
        auto fuu = derivatives.fuu(t);

        auto eq_ = derivatives.eq(t);
        auto eqx = derivatives.eqx(t);
        auto equ = derivatives.equ(t);
        auto eqxx = derivatives.eqxx(t);
        auto equx = derivatives.equx(t);
        auto equu = derivatives.equu(t);

        auto pe = mults.eq.self.val[t];
        auto pe_x = mults.eq.self.jac[t];

        auto const tmp_ = (pe + eq_.operator*(mu)).eval();
        auto const tmp2_ = (pe_x + eqx.operator*(mu)).eval();

        auto const tmp = eigen::as_const(tmp_);
        auto const tmp2 = eigen::as_const(tmp2_);

        bool const has_eq = tmp.rows() > 0;

        {
          using std::isfinite;
          VEG_DEBUG_ASSERT_ALL_OF( //
              isfinite(mu),
              !pe.hasNaN(),
              !pe_x.hasNaN());
        }

        auto Q_x = lx.eval();
        Q_x.noalias() += fx.transpose() * v_x;
        if (has_eq) {
          Q_x.noalias() += eqx.transpose() * tmp;
          Q_x.noalias() += pe_x.transpose() * eq_;
        }

        auto Q_u = lu.eval();
        Q_u.noalias() += fu.transpose() * v_x;
        if (has_eq) {
          Q_u.noalias() += equ.transpose() * tmp;
        }

        auto Q_xx = lxx.eval();
        Q_xx.noalias() += fx.transpose() * V_xx * fx;
        if (has_eq) {
          Q_xx.noalias() += eqx.transpose() * tmp2;
          Q_xx.noalias() += pe_x.transpose() * eqx;
          eqxx.noalias_contract_add_outdim(eigen::as_mut(Q_xx), tmp);
        }
        fxx.noalias_contract_add_outdim(eigen::as_mut(Q_xx), v_x);

        auto Q_uu = luu.eval();
        Q_uu.noalias() += fu.transpose() * V_xx * fu;
        if (has_eq) {
          Q_uu.noalias() +=
              (equ.transpose() * equ).operator*(mu); // *see below for reason
          equu.noalias_contract_add_outdim(eigen::as_mut(Q_uu), tmp);
        }
        fuu.noalias_contract_add_outdim(eigen::as_mut(Q_uu), v_x);

        auto Q_ux = lux.eval();
        Q_ux.noalias() += fu.transpose() * V_xx * fx;
        if (has_eq) {
          Q_ux.noalias() += equ.transpose() * tmp2;
          equx.noalias_contract_add_outdim(eigen::as_mut(Q_ux), tmp);
        }
        fux.noalias_contract_add_outdim(eigen::as_mut(Q_ux), v_x);

        auto I_u = decltype(Q_uu)::Identity(Q_uu.rows(), Q_uu.rows());
        auto llt_res = (Q_uu + *reg * I_u).eval().llt();

        if (!(llt_res.info() == Eigen::ComputationInfo::Success)) {
          reg.increase_reg();
          break;
        }

        {
          control_feedback.self.origin[t] = xu[0_c];
          control_feedback.self.val[t] = llt_res.solve(Q_u);
          control_feedback.self.jac[t] = llt_res.solve(Q_ux);
          control_feedback.self.val[t] *= -1;
          control_feedback.self.jac[t] *= -1;
        }

        auto const k = control_feedback.self.val[t];
        auto const K = control_feedback.self.jac[t];

        expected_decrease += (0.5 * k.transpose() * Q_uu * k).value();

        // clang-format off
      V_x            = Q_x;
      V_x.noalias() += k.transpose() * Q_ux;

      V_xx            = Q_xx;
      V_xx.noalias() += Q_ux.transpose() * K;
        // clang-format on

        if (t == 0) {
          success = true;
        }
      }
    }
    return expected_decrease;
  }

  template <method M>
  auto fwd_pass(
      trajectory& new_traj_storage,
      trajectory const& reference_traj,
      typename multiplier_sequence<M>::type const& old_mults,
      control_feedback const& feedback,
      scalar mu,
      key k,
      veg::dynamic_stack_view stack,
      bool do_linesearch = true) const -> veg::tuple<key, scalar> {

    auto begin = reference_traj.index_begin();
    auto end = reference_traj.index_end();

    auto costs_old_storage =
        stack.make_new_for_overwrite(veg::tag<scalar>, end - begin + 1)
            .unwrap();
    auto costs_new_storage =
        stack.make_new_for_overwrite(veg::tag<scalar>, end - begin + 1)
            .unwrap();

    auto costs_old_traj = eigen::slice_to_vec(costs_old_storage);
    auto costs_new_traj = eigen::slice_to_vec(costs_new_storage);

    if (do_linesearch) {
      k = cost_seq_aug(
          eigen::slice_to_vec(costs_old_traj),
          reference_traj,
          old_mults,
          mu,
          VEG_FWD(k),
          stack);
    }

    scalar step = 1;
    bool success = false;

    auto tmp_storage = stack
                           .make_new_for_overwrite( //
                               veg::tag<scalar>,
                               self.constraint.state_space().max_ddim())
                           .unwrap();
    auto tmp = eigen::slice_to_vec(tmp_storage);

    while (!success) {
      if (step < 1e-10) {
        break;
      }

      for (i64 t = begin; t < end; ++t) {

        auto x_old = reference_traj[t][0_c];
        auto u_old = reference_traj[t][1_c];

        auto x_new = new_traj_storage[t][0_c];
        auto u_new = new_traj_storage[t][1_c];
        auto x_next_new = new_traj_storage.x(t + 1);

        self.dynamics.state_space().difference(
            eigen::as_mut(tmp),
            t,
            eigen::as_const(x_old),
            eigen::as_const(x_new),
            stack);

        u_new = u_old                         //
                + step * feedback.self.val[t] //
                + feedback.self.jac[t] * tmp;
        k = self.dynamics.eval_to(
            x_next_new,
            t,
            eigen::as_const(x_new),
            eigen::as_const(u_new),
            VEG_FWD(k),
            stack);
      }

      if (do_linesearch) {
        k = cost_seq_aug(
            eigen::as_mut(costs_new_traj),
            new_traj_storage,
            old_mults,
            mu,
            VEG_FWD(k),
            stack);

        if ((costs_new_traj - costs_old_traj).sum() <= 0) {
          success = true;
        } else {
          step *= 0.5;
        }
      } else {
        success = true;
      }
    }

    return {elems, VEG_FWD(k), step};
  }

  template <typename Mults>
  auto cost_seq_aug(
      eigen::view<scalar, colvec> out,
      trajectory const& traj,
      Mults const& mults,
      scalar mu,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    auto csp = self.constraint.output_space();

    auto ce_storage =
        stack.make_new_for_overwrite(veg::tag<scalar>, csp.max_dim()).unwrap();
    auto pe_storage =
        stack.make_new_for_overwrite(veg::tag<scalar>, csp.max_dim()).unwrap();

    for (i64 t = traj.index_begin(); t < traj.index_end(); ++t) {
      VEG_BIND(auto, (x, u), traj[t]);
      auto ce =
          eigen::as_mut(eigen::slice_to_vec(ce_storage).topRows(csp.dim(t)));
      auto pe =
          eigen::as_mut(eigen::slice_to_vec(pe_storage).topRows(csp.dim(t)));

      auto l = self.cost.eval(t, x, u, stack);
      k = self.constraint.eval_to(ce, t, x, u, VEG_FWD(k), stack);
      mults.eq.eval_to(pe, t, x, stack);

      if (ce.rows() > 0) {
        fmt::print("{}\n", ce.norm());
      }

      out[t - traj.index_begin()] =
          l + eigen::dot(pe, ce) + (mu / 2) * eigen::dot(ce, ce);
    }

    auto x = traj.x_f();
    out[traj.index_end() - traj.index_begin()] = self.cost.eval_final(x, stack);

    return k;
  }

  struct layout {
    Dynamics dynamics;
    Cost cost;
    Constraint constraint;
  } self;
};

namespace make {
namespace fn {
struct ddp_fn {
  template <typename Dynamics, typename Cost, typename Constraint>
  auto operator()(Dynamics dynamics, Cost cost, Constraint constraint) const
      -> ::ddp::ddp<Dynamics, Cost, Constraint> {
    return {{
        VEG_FWD(dynamics),
        VEG_FWD(cost),
        VEG_FWD(constraint),
    }};
  }
};
} // namespace fn
__VEG_ODR_VAR(ddp, fn::ddp_fn);
} // namespace make

} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_DDP_HPP_CAAKIEJQS */
