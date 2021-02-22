#ifndef DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS
#define DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS

#include "ddp/dynamics.hpp"

namespace ddp {
using namespace veg::literals;

template <typename Dynamics, typename Fn>
struct config_constraint {
  struct layout {
    Dynamics const& dynamics;
    Fn target_generator;
    mem_req generator_mem_req;
  } self;

  using key = typename Dynamics::key;
  using scalar = typename Dynamics::scalar;

  auto dynamics() const -> decltype(auto) { return self.dynamics; }

  auto output_space() const {
    return vector_space<scalar>{{dynamics().self.model.tangent_dim()}};
  }
  auto state_space() const { return dynamics().state_space(); }
  auto control_space() const { return dynamics().control_space(); }

  auto eval_to_req() const -> mem_req { return self.generator_mem_req; }

  auto eval_to(
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {
    auto _target = self.target_generator(t, stack);
    auto target = eigen::as_const(_target);

    if (target.rows() == 0) {
      return k;
    }

    VEG_DEBUG_ASSERT_ALL_OF( //
        (x.rows() == state_space().dim(t)),
        (u.rows() == control_space().dim(t)),
        (out.rows() == dynamics().self.model.tangent_dim()),
        (target.rows() == dynamics().self.model.config_dim()));

    (void)u;
    auto nq = dynamics().self.model.config_dim();

    dynamics().self.model.difference(
        out, target, eigen::split_at_row(x, nq)[0_c]);

    return k;
  }

  auto d_eval_to_req() const -> mem_req { return eval_to_req(); }

  auto d_eval_to(
      view<scalar, colmat> out_x,
      view<scalar, colmat> out_u,
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    auto _target = self.target_generator(t, stack);
    auto target = eigen::as_const(_target);

    if (target.rows() == 0) {
      return k;
    }

    VEG_DEBUG_ASSERT_ALL_OF( //
        (x.rows() == state_space().dim(t)),
        (u.rows() == control_space().dim(t)),
        (out_x.rows() == dynamics().self.model.tangent_dim()),
        (out_x.cols() == dynamics().state_space().ddim(t)),
        (out_u.rows() == dynamics().self.model.tangent_dim()),
        (out_u.cols() == dynamics().control_space().ddim(t)),
        (out.rows() == dynamics().self.model.tangent_dim()),
        (target.rows() == dynamics().self.model.config_dim()));

    (void)u;
    auto nq = dynamics().self.model.config_dim();
    auto nv = dynamics().self.model.tangent_dim();

    auto xq = eigen::split_at_row(x, nq)[0_c];

    VEG_BIND(auto, (out_xq, out_xv), eigen::split_at_col(out_x, nv));

    dynamics().self.model.difference(out, target, xq);
    dynamics().self.model.d_difference_dq_finish(out_xq, target, xq);

    out_xv.setZero();
    out_u.setZero();

    return k;
  }
};

namespace make {
namespace fn {
struct config_constraint_fn {
  template <typename Dynamics, typename Fn>
  auto
  operator()(Dynamics const& dynamics, Fn target_gen, mem_req gen_mem_req) const
      -> config_constraint<Dynamics, Fn> {
    return {{dynamics, VEG_FWD(target_gen), gen_mem_req}};
  }
};
} // namespace fn
VEG_ODR_VAR(config_constraint, fn::config_constraint_fn);
} // namespace make

template <typename Dynamics, typename Fn>
struct velocity_constraint {
  struct layout {
    Dynamics const& dynamics;
    Fn target_generator;
    mem_req generator_mem_req;
  } self;

  using key = typename Dynamics::key;
  using scalar = typename Dynamics::scalar;

  auto dynamics() const -> decltype(auto) { return self.dynamics; }

  auto output_space() const {
    return vector_space<scalar>{{dynamics().self.model.tangent_dim()}};
  }
  auto state_space() const { return dynamics().state_space(); }
  auto control_space() const { return dynamics().control_space(); }

  auto eval_to_req() const -> mem_req { return self.generator_mem_req; }

  auto eval_to(
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {
    auto _target = self.target_generator(t, stack);
    auto target = eigen::as_const(_target);

    if (target.rows() == 0) {
      return k;
    }

    VEG_DEBUG_ASSERT_ALL_OF( //
        (x.rows() == state_space().dim(t)),
        (u.rows() == control_space().dim(t)),
        (out.rows() == dynamics().self.model.tangent_dim()),
        (target.rows() == dynamics().self.model.tangent_dim()));

    (void)u;
    auto nq = dynamics().self.model.config_dim();

    eigen::sub_to(out, target, eigen::split_at_row(x, nq)[1_c]);

    return k;
  }

  auto d_eval_to_req() const -> mem_req { return eval_to_req(); }

  auto d_eval_to(
      view<scalar, colmat> out_x,
      view<scalar, colmat> out_u,
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    auto _target = self.target_generator(t, stack);
    auto target = eigen::as_const(_target);

    if (target.rows() == 0) {
      return k;
    }

    VEG_DEBUG_ASSERT_ALL_OF( //
        (x.rows() == state_space().dim(t)),
        (u.rows() == control_space().dim(t)),
        (out_x.rows() == dynamics().self.model.tangent_dim()),
        (out_x.cols() == dynamics().state_space().ddim(t)),
        (out_u.rows() == dynamics().self.model.tangent_dim()),
        (out_u.cols() == dynamics().control_space().ddim(t)),
        (out.rows() == dynamics().self.model.tangent_dim()),
        (target.rows() == dynamics().self.model.config_dim()));

    (void)u;
    auto nq = dynamics().self.model.config_dim();
    auto nv = dynamics().self.model.tangent_dim();

    VEG_BIND(auto, (out_xq, out_xv), eigen::split_at_col(out_x, nv));

    eigen::sub_to(out, target, eigen::split_at_row(x, nq)[1_c]);

    out_xq.setZero();
    out_xv.setIdentity();
    out_u.setZero();

    return k;
  }
};

// FIXME
VEG_INSTANTIATE_CLASS(
    config_constraint,
    pinocchio_dynamics_free<double>,
    veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>);
VEG_INSTANTIATE_CLASS(
    velocity_constraint,
    pinocchio_dynamics_free<double>,
    veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>);

template <typename Constraint>
struct constraint_advance_time {
  struct layout {
    Constraint constr;
  } self;

  using key = typename Constraint::key;
  using scalar = typename Constraint::scalar;

  auto dynamics() const -> decltype(auto) { return self.constr.dynamics(); }

  auto state_space() const { return self.constr.state_space(); }
  auto control_space() const { return self.constr.control_space(); }
  auto output_space() const { return self.constr.output_space(); }

  auto eval_to_req() const -> mem_req {
    return mem_req::sum_of({
        mem_req::max_of({
            self.constr.eval_to_req(),
            dynamics().eval_to_req(),
        }),

        {veg::tag<scalar>, dynamics().state_space().max_dim()},
    });
  }
  auto eval_to(
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {
    auto _x_n =
        stack.make_new(veg::tag<scalar>, dynamics().output_space().dim(t))
            .unwrap();

    auto x_n = eigen::slice_to_vec(_x_n); // x_{t+1}
    k = dynamics().eval_to(x_n, t, x, u, VEG_FWD(k), stack);
    k = self.constr.eval_to(
        out, t + 1, eigen::as_const(x_n), u, VEG_FWD(k), stack);
    return k;
  }

  auto d_eval_to_req() const -> mem_req {
    auto nx = dynamics().state_space().max_dim();
    auto ndx = dynamics().state_space().max_ddim();
    auto ndu = dynamics().control_space().max_ddim();
    auto nde = output_space().max_ddim();

    return mem_req::sum_of({

        mem_req::max_of({
            self.constr.d_eval_to_req(),
            dynamics().d_eval_to_req(),
        }),

        {veg::tag<scalar>,
         (nx          //
          + ndx       //
          + ndx * ndx //
          + ndx * ndu //
          + nde * ndx //
          + nde * ndu //
          )},
    });
  }

  auto d_eval_to(
      view<scalar, colmat> out_x,
      view<scalar, colmat> out_u,
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    auto nx = dynamics().output_space().dim(t);
    auto ndx = dynamics().output_space().ddim(t);
    auto ndu1 = dynamics().control_space().ddim(t);
    auto ndu2 = dynamics().control_space().ddim(t + 1);
    auto nde = output_space().ddim(t + 1);

    constexpr auto tag = veg::tag<scalar>;

    auto _x_n = stack.make_new(tag, nx).unwrap();
    auto x_n = eigen::slice_to_vec(_x_n);

    auto _fx_n = stack.make_new(tag, ndx * ndx).unwrap();
    auto _fu_n = stack.make_new(tag, ndx * ndu1).unwrap();
    auto fx_n = eigen::slice_to_mat(_fx_n, ndx, ndx);
    auto fu_n = eigen::slice_to_mat(_fu_n, ndx, ndu1);

    k = dynamics().d_eval_to(fx_n, fu_n, x_n, t, x, u, VEG_FWD(k), stack);

    auto _eq_n_x = stack.make_new(tag, nde * ndx).unwrap();
    auto _eq_n_u = stack.make_new(tag, nde * ndu2).unwrap();
    auto eq_n_x = eigen::slice_to_mat(_eq_n_x, nde, ndx);
    auto eq_n_u = eigen::slice_to_mat(_eq_n_u, nde, ndu2);

    k = self.constr.d_eval_to(
        eq_n_x, eq_n_u, out, t + 1, eigen::as_const(x_n), u, VEG_FWD(k), stack);

    VEG_DEBUG_ASSERT_ELSE(
        "control should have no effect on the constraint value",
        eq_n_u.isConstant(0));

    out_x.setZero();
    out_u.setZero();

    eigen::mul_add_to_noalias(out_x, eq_n_x, fx_n);
    eigen::mul_add_to_noalias(out_u, eq_n_x, fu_n);

    return k;
  }
};

// FIXME
VEG_INSTANTIATE_CLASS(
    constraint_advance_time,
    config_constraint<
        pinocchio_dynamics_free<double>,
        veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>>);

template <typename Constr1, typename Constr2>
struct concat_constraint {
  struct layout {
    Constr1 constr1;
    Constr2 constr2;
  } self;

  static_assert(
      VEG_SAME_AS(typename Constr1::scalar, typename Constr2::scalar), "");
  static_assert(VEG_SAME_AS(typename Constr1::key, typename Constr2::key), "");

  using key = typename Constr1::key;
  using scalar = typename Constr1::scalar;

  auto dynamics() const -> decltype(auto) { return self.constr1.dynamics(); }

  auto state_space() const { return self.constr1.state_space(); }
  auto control_space() const { return self.constr1.control_space(); }
  auto output_space() const { return self.constr1.output_space(); }

  auto eval_to_req() const -> mem_req {
    return mem_req::max_of({
        self.constr1.eval_to_req(),
        self.constr2.eval_to_req(),
    });
  }
  auto eval_to(
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    VEG_BIND(
        auto,
        (out1, out2),
        eigen::split_at_row(out, self.constr1.output_space().dim(t)));

    k = self.constr1.eval_to(out1, t, x, u, VEG_FWD(k), stack);
    k = self.constr2.eval_to(out2, t, x, u, VEG_FWD(k), stack);

    return k;
  }

  auto d_eval_to_req() const -> mem_req {
    return mem_req::max_of({
        self.constr1.d_eval_to_req(),
        self.constr2.d_eval_to_req(),
    });
  }

  auto d_eval_to(
      view<scalar, colmat> out_x,
      view<scalar, colmat> out_u,
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> x,
      view<scalar const, colvec> u,
      key k,
      veg::dynamic_stack_view stack) const -> key {

    VEG_BIND(
        auto,
        (out1, out2),
        eigen::split_at_row(out, self.constr1.output_space().dim(t)));
    VEG_BIND(
        auto,
        (out_x1, out_x2),
        eigen::split_at_row(out_x, self.constr1.output_space().ddim(t)));
    VEG_BIND(
        auto,
        (out_u1, out_u2),
        eigen::split_at_row(out_u, self.constr1.output_space().ddim(t)));

    k = self.constr1.d_eval_to(
        out_x1, out_u1, out1, t, x, u, VEG_FWD(k), stack);
    k = self.constr2.d_eval_to(
        out_x2, out_u2, out2, t, x, u, VEG_FWD(k), stack);

    return k;
  }
};

// FIXME
VEG_INSTANTIATE_CLASS(
    concat_constraint,
    config_constraint<
        pinocchio_dynamics_free<double>,
        veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>>,
    config_constraint<
        pinocchio_dynamics_free<double>,
        veg::fn_ref<view<double const, colvec>(i64, veg::dynamic_stack_view)>>);

} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_CONSTRAINT_HPP_GKXIDUUFS */
