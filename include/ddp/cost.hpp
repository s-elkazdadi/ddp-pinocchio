#ifndef DDP_PINOCCHIO_COST_HPP_FVE04HDWS
#define DDP_PINOCCHIO_COST_HPP_FVE04HDWS

#include "ddp/internal/eigen.hpp"

namespace ddp {

template <typename T>
struct quadratic_cost_fixed_size {
  struct layout {
    std::vector<T> q;
    std::vector<T> Q;
    std::vector<T> r;
    std::vector<T> R;

    std::vector<T> rf;
    std::vector<T> Rf;
  } self;

  auto eval_to_req() const -> mem_req { return {veg::tag<T>, 0}; }
  auto d_eval_to_req() const -> mem_req {
    return {
        veg::tag<T>,
        veg::narrow<i64>(
            veg::meta::max_of({self.q.size(), self.r.size(), self.rf.size()}))};
  }
  auto dd_eval_to_req() const -> mem_req { return {veg::tag<T>, 0}; }

  auto eval_final(view<T const, colvec> x, veg::dynamic_stack_view stack) const
      -> T {

    auto nx = veg::narrow<i64>(self.rf.size());

    VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx);

    T out(int(0));
    {
      auto _tmp1 = stack.make_new(veg::tag<T>, x.rows()).unwrap();
      auto tmp1 = eigen::slice_to_vec(_tmp1);
      eigen::mul_add_to_noalias(tmp1, eigen::slice_to_mat(self.Rf, nx, nx), x);
      out += eigen::dot(eigen::slice_to_vec(self.rf), x);
    }

    return out;
  }

  void d_eval_final_to(
      view<T, colvec> out_x,
      view<T const, colvec> x,
      veg::dynamic_stack_view stack) const {

    (void)stack;
    auto nx = veg::narrow<i64>(self.r.size());

    VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx);

    eigen::assign(out_x, eigen::slice_to_vec(self.rf));
    eigen::mul_add_to_noalias(out_x, eigen::slice_to_mat(self.Rf, nx, nx), x);
  }

  void dd_eval_final_to(
      view<T, colmat> out_xx,
      view<T const, colvec> x,
      veg::dynamic_stack_view stack) const {

    (void)stack;
    auto nx = veg::narrow<i64>(self.rf.size());

    VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx);
    eigen::assign(out_xx, eigen::slice_to_mat(self.Rf, nx, nx));
  }

  auto eval(
      i64 t,
      view<T const, colvec> x,
      view<T const, colvec> u,
      veg::dynamic_stack_view stack) const -> T {

    (void)t;

    auto nu = veg::narrow<i64>(self.q.size());
    auto nx = veg::narrow<i64>(self.r.size());

    VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx, u.rows() == nu);

    T out(int(0));
    {
      auto _tmp1 = stack.make_new(veg::tag<T>, x.rows()).unwrap();
      auto tmp1 = eigen::slice_to_vec(_tmp1);
      eigen::mul_add_to_noalias(tmp1, eigen::slice_to_mat(self.R, nx, nx), x);
      out += eigen::dot(eigen::slice_to_vec(self.r), x);
    }
    {
      auto _tmp1 = stack.make_new(veg::tag<T>, u.rows()).unwrap();
      auto tmp1 = eigen::slice_to_vec(_tmp1);
      eigen::mul_add_to_noalias(tmp1, eigen::slice_to_mat(self.Q, nu, nu), u);
      out += eigen::dot(eigen::slice_to_vec(self.q), u);
    }
    return out;
  }

  void d_eval_to(
      view<T, colvec> out_x,
      view<T, colvec> out_u,
      i64 t,
      view<T const, colvec> x,
      view<T const, colvec> u,
      veg::dynamic_stack_view stack) const {

    (void)stack, (void)t;

    auto nu = veg::narrow<i64>(self.q.size());
    auto nx = veg::narrow<i64>(self.r.size());

    VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx, u.rows() == nu);

    eigen::assign(out_x, eigen::slice_to_vec(self.r));
    eigen::assign(out_u, eigen::slice_to_vec(self.q));
    eigen::mul_add_to_noalias(out_x, eigen::slice_to_mat(self.R, nx, nx), x);
    eigen::mul_add_to_noalias(out_u, eigen::slice_to_mat(self.Q, nu, nu), u);
  }

  void dd_eval_to(
      view<T, colmat> out_xx,
      view<T, colmat> out_ux,
      view<T, colmat> out_uu,
      i64 t,
      view<T const, colvec> x,
      view<T const, colvec> u,
      veg::dynamic_stack_view stack) const {

    (void)t, (void)x, (void)u, (void)stack;

    auto nu = veg::narrow<i64>(self.q.size());
    auto nx = veg::narrow<i64>(self.r.size());

    VEG_DEBUG_ASSERT_ALL_OF(x.rows() == nx, u.rows() == nu);

    eigen::assign(out_xx, eigen::slice_to_mat(self.R, nx, nx));
    eigen::assign(out_uu, eigen::slice_to_mat(self.Q, nu, nu));
    out_ux.setZero();
  }
};

VEG_INSTANTIATE_CLASS(quadratic_cost_fixed_size, double);

namespace make {
namespace fn {
struct quadratic_cost_fixed_size_fn {
  template <typename MatV, typename T = typename MatV::Scalar>
  auto operator()(
      MatV q,
      eigen::view<T const, colmat> Q,
      eigen::view<T const, colvec> r,
      eigen::view<T const, colmat> R,
      eigen::view<T const, colvec> rf,
      eigen::view<T const, colmat> Rf) const -> quadratic_cost_fixed_size<T> {
    auto nq = q.size();
    auto nr = r.size();
    VEG_DEBUG_ASSERT_ALL_OF(
        Q.rows() == nq,
        Q.cols() == nq,
        R.rows() == nr,
        R.cols() == nr,
        rf.rows() == nr,
        Rf.rows() == nr);

    auto to_vec = [](auto const& v) {
      return std::vector<T>(v.data(), v.data() + v.size());
    };

    return {{
        to_vec(q),
        to_vec(Q),
        to_vec(r),
        to_vec(R),
        to_vec(rf),
        to_vec(Rf),
    }};
  }
};
} // namespace fn
VEG_ODR_VAR(quadratic_cost_fixed_size, fn::quadratic_cost_fixed_size_fn);
} // namespace make

} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_COST_HPP_FVE04HDWS */
