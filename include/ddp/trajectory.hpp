#ifndef DDP_PINOCCHIO_TRAJECTORY_HPP_VQYVAKHAS
#define DDP_PINOCCHIO_TRAJECTORY_HPP_VQYVAKHAS

#include "ddp/internal/matrix_seq.hpp"

namespace ddp {

template <typename T>
struct trajectory {
  static_assert(!std::is_const<T>::value, "");

  using seq = internal::mat_seq<T, colvec>;
  struct layout {
    seq x;
    seq u;
  } self;
  explicit trajectory(layout l) : self{VEG_FWD(l)} {}

  trajectory(idx::idx<colvec> x, idx::idx<colvec> u)
      : self{seq{VEG_FWD(x)}, seq{VEG_FWD(u)}} {
    VEG_ASSERT_ALL_OF( //
        (x.index_begin() == u.index_begin()),
        (x.index_end() == u.index_end() + 1));
  }

  auto index_begin() const -> i64 { return self.u.index_begin(); }
  auto index_end() const -> i64 { return self.u.index_end(); }

  auto operator[](i64 t) const
      -> veg::tuple<view<T const, colvec>, view<T const, colvec>> {
    return {elems, self.x[t], self.u[t]};
  }
  auto operator[](i64 t) -> veg::tuple<view<T, colvec>, view<T, colvec>> {
    return {elems, self.x[t], self.u[t]};
  }
  auto x(i64 t) -> view<T, colvec> { return {self.x[t]}; }
  auto x(i64 t) const -> view<T const, colvec> { return {self.x[t]}; }
  auto u(i64 t) -> view<T, colvec> { return {self.u[t]}; }
  auto u(i64 t) const -> view<T const, colvec> { return {self.u[t]}; }
  auto x_f() const -> view<T const, colvec> {
    return self.x[self.x.index_end() - 1];
  }
  auto x_f() -> view<T, colvec> { return self.x[self.x.index_end() - 1]; }
};

} // namespace ddp

#endif /* end of include guard DDP_PINOCCHIO_TRAJECTORY_HPP_VQYVAKHAS */
