#ifndef DDP_PINOCCHIO_FUNCTION_MODELS_HPP_EPD2SYHQS
#define DDP_PINOCCHIO_FUNCTION_MODELS_HPP_EPD2SYHQS

#include "ddp/internal/matrix_seq.hpp"
#include "ddp/space.hpp"

namespace ddp {
namespace internal {

template <typename Scalar, typename State_Space, typename Output_Space>
struct affine_function_seq {
  using state_space = State_Space;
  using output_space = Output_Space;
  using scalar = Scalar;

  static_assert(VEG_SAME_AS(scalar, typename State_Space::scalar), "");
  static_assert(VEG_SAME_AS(scalar, typename Output_Space::scalar), "");

  struct layout {
    mat_seq<scalar, colvec> origin;
    mat_seq<scalar, colvec> val;
    mat_seq<scalar, colmat> jac;
    state_space in;
    output_space out;
  } self;

  explicit affine_function_seq(layout l) : self{VEG_FWD(l)} {}

  affine_function_seq(i64 begin, i64 end, state_space in, output_space out)
      : self{
            mat_seq<scalar, colvec>{
                {begin,
                 end,
                 [&](i64 t) { return idx::dims<colvec>{in.dim(t)}; }}},
            mat_seq<scalar, colvec>{
                {begin,
                 end,
                 [&](i64 t) { return idx::dims<colvec>{out.dim(t)}; }}},
            mat_seq<scalar, colmat>{
                {begin,
                 end,
                 [&](i64 t) {
                   return idx::dims<colmat>{out.dim(t), in.dim(t)};
                 }}},
            VEG_FWD(in),
            VEG_FWD(out),
        } {}

  auto update_origin_req() const -> mem_req {
    return mem_req::sum_of({
        {veg::tag<scalar>,
         (self.in.max_ddim()                            // diff
          + self.in.max_ddim() * self.in.max_ddim()     // diff_jac
          + self.out.max_ddim() * self.in.max_ddim())}, // tmp

        mem_req::max_of({
            self.in.difference_req(),
            self.in.d_difference_d_finish_req(),
        }),
    });
  }

  auto eval_to_req() const -> mem_req {
    return mem_req::sum_of({
        {veg::tag<scalar>, self.in.max_ddim()},
        self.in.difference_req(),
    });
  }
  void eval_to(
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> in,
      veg::dynamic_stack_view stack) const {
    auto _tmp = stack.make_new(veg::tag<scalar>, self.in.ddim(t)).unwrap();
    auto tmp = eigen::slice_to_vec(_tmp);

    self.in.difference(tmp, t, self.origin[t], in, stack);
    eigen::assign(out, self.val[t]);
    eigen::mul_add_to_noalias(out, self.jac[t], tmp);
  }

  void update_origin(
      mat_seq<scalar, colvec> const& new_traj, veg::dynamic_stack_view stack) {

    auto begin = self.origin.index_begin();
    auto end = self.origin.index_end();

    VEG_DEBUG_ASSERT_ALL_OF(
        (new_traj.index_end() == end), (new_traj.index_begin() == begin));

    for (i64 t = begin; t < end; ++t) {
      auto origin = self.origin[t];
      auto val = self.val[t];
      auto jac = self.jac[t];

      auto new_origin = new_traj[t];

      auto ndi = self.in.ddim(t);
      auto ndo = self.out.ddim(t);

      auto _diff = stack.make_new(veg::tag<scalar>, ndi).unwrap();
      auto _diff_jac = stack.make_new(veg::tag<scalar>, ndi * ndi).unwrap();
      auto _tmp = stack.make_new(veg::tag<scalar>, ndo * ndi).unwrap();

      auto diff = eigen::slice_to_vec(_diff);
      auto diff_jac = eigen::slice_to_mat(_diff_jac, ndi, ndi);
      auto tmp = eigen::slice_to_mat(_tmp, ndo, ndi);

      self.in.difference(diff, t, eigen::as_const(origin), new_origin, stack);
      self.in.d_difference_d_finish(
          diff_jac, t, eigen::as_const(origin), new_origin, stack);

      eigen::mul_add_to_noalias(val, jac, diff);

      tmp.setZero();
      eigen::mul_add_to_noalias(tmp, jac, diff_jac);
      eigen::assign(jac, tmp);
      eigen::assign(origin, new_origin);
    }
  }
};

template <typename Scalar, typename Output_Space>
struct constant_function_seq {
  using output_space = Output_Space;
  using scalar = Scalar;

  static_assert(VEG_SAME_AS(scalar, typename Output_Space::scalar), "");

  struct layout {
    mat_seq<scalar, colvec> val;
    output_space out;
  } self;

  explicit constant_function_seq(layout l) : self{VEG_FWD(l)} {}

  constant_function_seq(i64 begin, i64 end, output_space out)
      : self{
            mat_seq<scalar, colvec>{{
                begin,
                end,
                [&](i64 t) { return idx::dims<colvec>{out.dim(t)}; },
            }},
            out,
        } {}

  auto update_origin_req() const -> mem_req { return {veg::tag<scalar>, 0}; }
  auto eval_to_req() const -> mem_req { return {veg::tag<scalar>, 0}; }

  void eval_to(
      view<scalar, colvec> out,
      i64 t,
      view<scalar const, colvec> in,
      veg::dynamic_stack_view stack) const {

    (void)stack, (void)in, (void)t;
    eigen::assign(out, self.val[t]);
  }

  void update_origin(
      mat_seq<scalar, colvec> const& new_traj, veg::dynamic_stack_view stack) {
    (void)this, (void)new_traj, (void)stack;
  }
};

// FIXME
VEG_INSTANTIATE_CLASS(
    affine_function_seq, double, vector_space<double>, vector_space<double>);
VEG_INSTANTIATE_CLASS(constant_function_seq, double, vector_space<double>);

} // namespace internal
} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_FUNCTION_MODELS_HPP_EPD2SYHQS */
