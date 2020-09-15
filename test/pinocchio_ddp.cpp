#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, "  ", "\n", "", "")
#include "ddp/detail/utils.hpp"

#include "ddp/pinocchio_model.hpp"
#include "ddp/ddp.hpp"
#include "ddp/ddp_bwd.ipp"
#include "ddp/ddp_fwd.ipp"

#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <boost/multiprecision/mpfr.hpp>
#include <chrono>

#if 1
using scalar_t = boost::multiprecision::number<
    boost::multiprecision::backends::mpfr_float_backend<500, boost::multiprecision::allocate_stack>,
    boost::multiprecision::et_off>;
#else
using scalar_t = double;
#endif

using namespace ddp;

namespace gsl {
template <typename T>
using owner = T;
} // namespace gsl

struct file_t {
  gsl::owner<std::FILE*> ptr;
  file_t(file_t const&) = delete;
  file_t(file_t&&) = delete;
  auto operator=(file_t const&) -> file_t& = delete;
  auto operator=(file_t &&) -> file_t& = delete;
  file_t(char const* file, char const* mode) noexcept : ptr{std::fopen(file, mode)} {
    if (ptr == nullptr) {
      std::terminate();
    }
  }
  ~file_t() { std::fclose(ptr); }
};

struct chronometer_t {
  static file_t const file;
  using clock_t = std::chrono::steady_clock;
  clock_t::time_point m_begin;
  clock_t::time_point m_end;
  char const* m_message;

  chronometer_t(chronometer_t const&) = delete;
  chronometer_t(chronometer_t&&) = delete;
  auto operator=(chronometer_t const&) -> chronometer_t& = delete;
  auto operator=(chronometer_t &&) -> chronometer_t& = delete;

  explicit chronometer_t(char const* message) : m_begin{}, m_message{message} { m_begin = clock_t::now(); }
  ~chronometer_t() {
    m_end = clock_t::now();
    fmt::print(
        file.ptr,
        "finished: {} | {} elapsed\n",
        m_message,
        std::chrono::duration<double, std::milli>(m_end - m_begin));
  }
};
file_t const chronometer_t::file = {"/tmp/chrono.log", "w"};

template <typename Out, typename XX, typename UX, typename UU, typename DX, typename DU>
void add_second_order_term(Out&& out, XX const& xx, UX const& ux, UU const& uu, DX const& dx, DU const& du) {
  DDP_ASSERT(out.cols() == 1, "wrong dimensions");
  DDP_ASSERT(out.rows() == xx.outdim().value(), "wrong dimensions");
  DDP_ASSERT(out.rows() == ux.outdim().value(), "wrong dimensions");
  DDP_ASSERT(out.rows() == uu.outdim().value(), "wrong dimensions");

  DDP_ASSERT(dx.rows() == xx.indiml().value(), "wrong dimensions");
  DDP_ASSERT(dx.rows() == xx.indimr().value(), "wrong dimensions");
  DDP_ASSERT(dx.rows() == ux.indimr().value(), "wrong dimensions");

  DDP_ASSERT(du.rows() == uu.indiml().value(), "wrong dimensions");
  DDP_ASSERT(du.rows() == uu.indimr().value(), "wrong dimensions");
  DDP_ASSERT(du.rows() == ux.indiml().value(), "wrong dimensions");

  for (index_t j = 0; j < dx.rows(); ++j) {
    for (index_t i = 0; i < dx.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * dx(i) * xx(k, i, j) * dx(j);
      }
    }
  }

  for (index_t j = 0; j < du.rows(); ++j) {
    for (index_t i = 0; i < du.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += 0.5 * du(i) * uu(k, i, j) * du(j);
      }
    }
  }

  for (index_t j = 0; j < dx.rows(); ++j) {
    for (index_t i = 0; i < du.rows(); ++i) {
      for (index_t k = 0; k < out.rows(); ++k) {
        out(k) += du(i) * ux(k, i, j) * dx(j);
      }
    }
  }
}

struct problem_t {
  using scalar_t = ::scalar_t;
  using model_t = pinocchio::model_t<scalar_t>;
  using state_indexer_t = indexing::regular_indexer_t<dyn_index>;
  using dstate_indexer_t = indexing::regular_indexer_t<dyn_index>;
  using control_indexer_t = indexing::regular_indexer_t<dyn_index>;
  using eq_indexer_t = indexing::regular_indexer_t<dyn_index>;

  using trajectory_t = trajectory::trajectory_t<scalar_t, state_indexer_t, control_indexer_t>;
  using derivative_storage_t = ddp::derivative_storage_t<scalar_t, control_indexer_t, eq_indexer_t, dstate_indexer_t>;

  using x_mut = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, state_indexer_t>>;
  using eq_mut = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, eq_indexer_t>>;

  using x_const = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, state_indexer_t> const>;
  using u_const = eigen::view_t<eigen::matrix_from_idx_t<scalar_t, control_indexer_t> const>;

  auto state_indexer(index_t begin, index_t end) const -> state_indexer_t {
    return indexing::vec_regular_indexer(begin, end, dyn_index{m_model.configuration_dim() + m_model.tangent_dim()});
  }
  auto dstate_indexer(index_t begin, index_t end) const -> dstate_indexer_t {
    return indexing::vec_regular_indexer(begin, end, dyn_index{m_model.tangent_dim() + m_model.tangent_dim()});
  }

  auto lf(x_const x) const -> scalar_t {
    detail::unused(this, x);
    return 0;
  }
  auto l(index_t t, x_const x, u_const u) const -> scalar_t {
    detail::unused(this, t, x);
    return 0.5 * u.squaredNorm();
  }
  void eval_f_to(x_mut x_out, index_t t, x_const x, u_const u) const {
    detail::unused(t);
    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();

    // v_out = dt * v_in
    x_out.bottomRows(nv) = dt * x.bottomRows(nv);

    // q_out = q_in + v_out
    //       = q_in + dt * v_in
    m_model.integrate(                             //
        eigen::as_mut_view(x_out.topRows(nq)),     //
        eigen::as_const_view(x.topRows(nq)),       //
        eigen::as_const_view(x_out.bottomRows(nv)) //
    );

    // v_out = acc
    m_model.dynamics_aba(                         //
        eigen::as_mut_view(x_out.bottomRows(nv)), //
        eigen::as_const_view(x.topRows(nq)),      //
        eigen::as_const_view(x.bottomRows(nv)),   //
        u                                         //
    );

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    x_out.bottomRows(nv) = x.bottomRows(nv) + x_out.bottomRows(nv) * dt;
  }
  void eval_eq_to(eq_mut eq_out, index_t t, x_const x, u_const u) const {
    detail::unused(t, u);
    vec_t x_n{m_model.configuration_dim() + m_model.tangent_dim()};
    vec_t x_nn{m_model.configuration_dim() + m_model.tangent_dim()};
    eval_f_to(eigen::as_mut_view(x_n), t, x, u);
    eval_f_to(eigen::as_mut_view(x_nn), t, eigen::as_const_view(x_n), u);

    // eq = q_in - q0
    m_model.difference(                                                 //
        eq_out,                                                         //
        eigen::as_const_view(m_eq_ref),                                 //
        eigen::as_const_view(x_nn.topRows(m_model.configuration_dim())) //
    );
  }

  using vec_t = Eigen::Matrix<scalar_t, -1, 1>;
  using mat_t = Eigen::Matrix<scalar_t, -1, -1>;
  using mat_mut_view_t = eigen::view_t<mat_t>;
  using mat_const_view_t = eigen::view_t<mat_t>;
  using tensor_t = tensor::tensor_t<scalar_t, dyn_index, dyn_index, dyn_index>;
  using tensor_mut_view_t = tensor::tensor_view_t<scalar_t, dyn_index, dyn_index, dyn_index>;
  using tensor_const_view_t = tensor::tensor_view_t<scalar_t, dyn_index, dyn_index, dyn_index>;

  // TODO extract common code from compute_{f,eq}_derivatives
  // TODO need commutator for hessian finite differences with jacobian
  void compute_eq_derivatives( //
      tensor_mut_view_t eq_xx, //
      tensor_mut_view_t eq_ux, //
      tensor_mut_view_t eq_uu, //
      mat_mut_view_t eq_x,     //
      mat_mut_view_t eq_u,     //
      eq_mut eq_v,             //
      index_t t,               //
      x_const x,               //
      u_const u                //
  ) const {

    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();
    index_t ne = eq_v.rows();

    tensor_t fxx_n{dyn_index{2 * nv}, dyn_index{2 * nv}, dyn_index{2 * nv}};
    tensor_t fux_n{dyn_index{2 * nv}, dyn_index{nv}, dyn_index{2 * nv}};
    tensor_t fuu_n{dyn_index{2 * nv}, dyn_index{nv}, dyn_index{nv}};

    mat_t fx_n{2 * nv, 2 * nv};
    mat_t fu_n{2 * nv, nv};

    tensor_t fxx_nn{dyn_index{2 * nv}, dyn_index{2 * nv}, dyn_index{2 * nv}};
    tensor_t fux_nn{dyn_index{2 * nv}, dyn_index{nv}, dyn_index{2 * nv}};
    tensor_t fuu_nn{dyn_index{2 * nv}, dyn_index{nv}, dyn_index{nv}};

    mat_t fx_nn{2 * nv, 2 * nv};
    mat_t fu_nn{2 * nv, nv};

    vec_t x_n{m_model.configuration_dim() + m_model.tangent_dim()};
    vec_t x_nn{m_model.configuration_dim() + m_model.tangent_dim()};
    eval_f_to(eigen::as_mut_view(x_n), t, x, u);
    eval_f_to(eigen::as_mut_view(x_nn), t, eigen::as_const_view(x_n), u);

    compute_f_derivatives(
        fxx_n.as_mut_view(),
        fux_n.as_mut_view(),
        fuu_n.as_mut_view(),
        eigen::as_mut_view(fx_n),
        eigen::as_mut_view(fu_n),
        t,
        x,
        u,
        eigen::as_const_view(x_n));

    compute_f_derivatives(
        fxx_nn.as_mut_view(),
        fux_nn.as_mut_view(),
        fuu_nn.as_mut_view(),
        eigen::as_mut_view(fx_nn),
        eigen::as_mut_view(fu_nn),
        t + 1,
        eigen::as_const_view(x_n),
        u,
        eigen::as_const_view(x_nn));

    // eq(t, x, u) = f(t+1, f(t, x, u), u)
    {
      m_model.difference( //
          eq_v,
          eigen::as_const_view(m_eq_ref),
          eigen::as_const_view(x_nn.topRows(nq)));
    }
    // first derivatives
    {
      auto d_diff = m_model._d_difference_dq_finish(m_eq_ref, x_nn.topRows(nq));

      eq_x.noalias() = d_diff * (fx_nn * fx_n).topRows(nv);
      eq_u.noalias() = d_diff * (fx_nn * fu_n).topRows(nv);
    }
    // second derivatives
    {
      vec_t eq1{ne};
      vec_t x1 = x;
      vec_t u1 = u;

      vec_t dx = vec_t::Zero(nv + nv);
      vec_t du = vec_t::Zero(nv);

      using std::sqrt;
      scalar_t eps = sqrt(sqrt(std::numeric_limits<scalar_t>::epsilon()));
      scalar_t eps2 = eps * eps;

      // compute diagonal
      for (index_t i = 0; i < 3 * nv; ++i) {
        bool at_x = (i < (2 * nv));
        index_t idx = at_x ? i : (i - 2 * nv);

        scalar_t& in_var = at_x ? dx[idx] : du[idx];

        auto eq_col = at_x ? eigen::as_const_view(eq_x.col(idx)) : eigen::as_const_view(eq_u.col(idx));
        auto tensor = at_x ? eq_xx.as_dynamic() : eq_uu.as_dynamic();

        in_var = eps;

        m_model.integrate(
            eigen::as_mut_view(x1.topRows(nq)),
            eigen::as_const_view(x.topRows(nq)),
            eigen::as_const_view(dx.topRows(nv)));
        x1.bottomRows(nv) = x.bottomRows(nv) + dx.bottomRows(nv);
        u1 = u + du;

        eval_eq_to(eigen::as_mut_view(eq1), t, eigen::as_const_view(x1), eigen::as_const_view(u1));
        eq1 -= eq_v;
        eq1 -= eps * eq_col;
        eq1 *= 2;

        for (index_t k = 0; k < ne; ++k) {
          tensor(k, idx, idx) = eq1[k] / eps2;
        }

        in_var = 0;
      }

      // compute non diagonal part
      // ei H ej = ((ei + ej) H (ei + ej) - ei H ei - ej H ej) / 2
      for (index_t i = 0; i < 3 * nv; ++i) {
        bool at_x_1 = (i < 2 * nv);
        index_t idx_1 = at_x_1 ? i : (i - 2 * nv);

        scalar_t& in_var_1 = at_x_1 ? dx[idx_1] : du[idx_1];
        auto eq_col_1 = at_x_1 ? eigen::as_const_view(eq_x.col(idx_1)) : eigen::as_const_view(eq_u.col(idx_1));

        auto tensor_1 = at_x_1 //
                            ? eq_xx.as_dynamic()
                            : eq_uu.as_dynamic();

        in_var_1 = eps;

        for (index_t j = i + 1; j < 3 * nv; ++j) {
          bool at_x_2 = (j < 2 * nv);
          index_t idx_2 = at_x_2 ? j : (j - 2 * nv);

          scalar_t& in_var_2 = at_x_2 ? dx[idx_2] : du[idx_2];
          auto eq_col_2 = at_x_2 ? eigen::as_const_view(eq_x.col(idx_2)) : eigen::as_const_view(eq_u.col(idx_2));

          auto tensor_2 = at_x_2 //
                              ? eq_xx.as_dynamic()
                              : eq_uu.as_dynamic();

          auto tensor = at_x_1 ? (at_x_2 //
                                      ? eq_xx.as_dynamic()
                                      : eq_ux.as_dynamic())
                               : (at_x_2 //
                                      ? (assert(false), eq_uu.as_dynamic())
                                      : eq_uu.as_dynamic());

          in_var_2 = eps;

          m_model.integrate(
              eigen::as_mut_view(x1.topRows(nq)),
              eigen::as_const_view(x.topRows(nq)),
              eigen::as_const_view(dx.topRows(nv)));
          x1.bottomRows(nv) = x.bottomRows(nv) + dx.bottomRows(nv);
          u1 = u + du;

          eval_eq_to(eigen::as_mut_view(eq1), t, eigen::as_const_view(x1), eigen::as_const_view(u1));
          eq1 -= eq_v;
          eq1 -= eps * eq_col_1;
          eq1 -= eps * eq_col_2;
          eq1 *= 2;

          for (index_t k = 0; k < ne; ++k) {
            tensor(k, idx_2, idx_1) =              //
                0.5 * (eq1[k] / eps2               //
                       - tensor_1(k, idx_1, idx_1) //
                       - tensor_2(k, idx_2, idx_2));

            if (at_x_1 == at_x_2) {
              tensor(k, idx_1, idx_2) = tensor(k, idx_2, idx_1);
            }
          }

          in_var_2 = 0;
        }

        in_var_1 = 0;
      }
    }
  }

  void compute_commutator_jacobian(mat_mut_view_t jac, eigen::view_t<vec_t const> dq) const {
    // jac (nv, nv)
    //
    // jac × dy = [ dq , dy ]
    //          = ((Id + dv) + dy) - ((Id + dy) + dv)

    using std::sqrt;
    scalar_t eps = sqrt(std::numeric_limits<scalar_t>::epsilon());

    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();

    vec_t q_0{nq};
    vec_t q_1{nq};
    vec_t q_2{nq};
    vec_t tmp{nq};

    m_model.neutral_configuration(eigen::as_mut_view(q_0));

    vec_t dy = vec_t::Zero(nv);

    for (index_t i = 0; i < nv; ++i) {
      dy[i] = eps;

      q_1 = q_0;
      m_model.integrate(eigen::as_mut_view(q_1), eigen::as_const_view(q_0), eigen::as_const_view(dq));
      tmp = q_1;
      m_model.integrate(eigen::as_mut_view(q_1), eigen::as_const_view(tmp), eigen::as_const_view(dy));

      q_2 = q_0;
      m_model.integrate(eigen::as_mut_view(q_2), eigen::as_const_view(q_0), eigen::as_const_view(dy));
      tmp = q_2;
      m_model.integrate(eigen::as_mut_view(q_2), eigen::as_const_view(tmp), eigen::as_const_view(dq));

      m_model.difference(eigen::as_mut_view(jac.col(i)), eigen::as_const_view(q_1), eigen::as_const_view(q_2));
      jac.col(i) /= eps;

      dy[i] = 0;
    }
  }
  void compute_eq_derivatives_first( //
      mat_mut_view_t eq_x,           //
      mat_mut_view_t eq_u,           //
      eq_mut eq_v,                   //
      index_t t,                     //
      x_const x,                     //
      u_const u                      //
  ) const {

    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();

    mat_t fx_n{2 * nv, 2 * nv};
    mat_t fu_n{2 * nv, nv};

    tensor_t fxx_nn{dyn_index{2 * nv}, dyn_index{2 * nv}, dyn_index{2 * nv}};
    tensor_t fux_nn{dyn_index{2 * nv}, dyn_index{nv}, dyn_index{2 * nv}};
    tensor_t fuu_nn{dyn_index{2 * nv}, dyn_index{nv}, dyn_index{nv}};

    mat_t fx_nn{2 * nv, 2 * nv};
    mat_t fu_nn{2 * nv, nv};

    vec_t x_n{m_model.configuration_dim() + m_model.tangent_dim()};
    vec_t x_nn{m_model.configuration_dim() + m_model.tangent_dim()};
    eval_f_to(eigen::as_mut_view(x_n), t, x, u);
    eval_f_to(eigen::as_mut_view(x_nn), t + 1, eigen::as_const_view(x_n), u);

    compute_f_derivatives_first(eigen::as_mut_view(fx_n), eigen::as_mut_view(fu_n), t, x, u);

    compute_f_derivatives_first(
        eigen::as_mut_view(fx_nn),
        eigen::as_mut_view(fu_nn),
        t + 1,
        eigen::as_const_view(x_n),
        u);

    // eq(t, x, u) = f(t+1, f(t, x, u), u)
    {
      m_model.difference( //
          eq_v,
          eigen::as_const_view(m_eq_ref),
          eigen::as_const_view(x_nn.topRows(nq)));
    }

    // first derivatives
    {
      auto d_diff = m_model._d_difference_dq_finish(m_eq_ref, x_nn.topRows(nq));

      eq_x.noalias() = d_diff * (fx_nn * fx_n).topRows(nv);
      eq_u.noalias() = d_diff * (fx_nn * fu_n).topRows(nv);
    }
  }

  void compute_f_derivatives_first( //
      mat_mut_view_t fx,            //
      mat_mut_view_t fu,            //
      index_t t,                    //
      x_const x,                    //
      u_const u                     //
  ) const {
    (void)t;
    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();

    auto q = eigen::as_const_view(x.topRows(nq));
    auto v = eigen::as_const_view(x.bottomRows(nv));

    // v_out = dt * v_in;
    fx.col(0).bottomRows(nv) = dt * v;
    auto dt_v = eigen::as_const_view(fx.col(0).bottomRows(nv));

    // q_out = q_in + dt * v_in
    m_model.d_integrate_dq(eigen::as_mut_view(fx.topLeftCorner(nv, nv)), q, dt_v);
    m_model.d_integrate_dv(eigen::as_mut_view(fx.topRightCorner(nv, nv)), q, dt_v);
    fx.topRightCorner(nv, nv) *= dt;

    // v_out = acc
    fu.topRows(nv).setZero();
    m_model.d_dynamics_aba( //
        eigen::as_mut_view(fx.bottomLeftCorner(nv, nv)),
        eigen::as_mut_view(fx.bottomRightCorner(nv, nv)),
        eigen::as_mut_view(fu.bottomRows(nv)),
        q,
        v,
        u);

    // v_out = v_in + dt * v_out
    //       = v_in + dt * acc
    fx.bottomRows(nv) *= dt;
    fx.bottomRightCorner(nv, nv) += mat_t::Identity(nv, nv);
    fu.bottomRows(nv) *= dt;
  }

  void compute_eq_derivatives_2( //
      tensor_mut_view_t eq_xx,   //
      tensor_mut_view_t eq_ux,   //
      tensor_mut_view_t eq_uu,   //
      mat_mut_view_t eq_x,       //
      mat_mut_view_t eq_u,       //
      eq_mut eq_val,             //
      index_t t,                 //
      x_const x,                 //
      u_const u) const {

    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();
    index_t ne = eq_val.rows();

    compute_eq_derivatives_first(eq_x, eq_u, eq_val, t, x, u);
    // compute second derivatives
    {

      mat_t eq_x_{ne, nv + nv};
      mat_t eq_u_{ne, nv};
      vec_t eq_{ne};

      vec_t x_ = x;
      vec_t u_ = u;

      vec_t dx = vec_t::Zero(nv + nv);
      vec_t du = vec_t::Zero(nv);

      using std::sqrt;
      scalar_t eps = sqrt(std::numeric_limits<scalar_t>::epsilon());

      for (index_t i = 0; i < 3 * nv; ++i) {
        bool at_x = (i < (2 * nv));
        index_t idx = at_x ? i : (i - 2 * nv);

        scalar_t& in_var = at_x ? dx[idx] : du[idx];

        in_var = eps;

        m_model.integrate(
            eigen::as_mut_view(x_.topRows(nq)),
            eigen::as_const_view(x.topRows(nq)),
            eigen::as_const_view(dx.topRows(nv)));
        x_.bottomRows(nv) = x.bottomRows(nv) + dx.bottomRows(nv);
        u_ = u + du;

        compute_eq_derivatives_first(
            eigen::as_mut_view(eq_x_),
            eigen::as_mut_view(eq_u_),
            eigen::as_mut_view(eq_),
            t,
            eigen::as_const_view(x_),
            eigen::as_const_view(u_));

        // mat_t commutator_jac{nv, nv};
        // compute_commutator_jacobian(eigen::as_mut_view(commutator_jac),
        // eigen::as_const_view(dx.topRows(nv)));

        if (at_x) {

          for (index_t k = 0; k < ne; ++k) {
            for (index_t j = 0; j < nv + nv; ++j) {
              eq_xx(k, j, idx) = (eq_x_(k, j) - eq_x(k, j)) / eps;
            }
            for (index_t j = 0; j < nv; ++j) {
              eq_ux(k, j, idx) = (eq_u_(k, j) - eq_u(k, j)) / eps;
            }
          }

          // mat_t tmp = -0.5 * eq_x.leftCols(nv) * commutator_jac;
          // for (index_t k = 0; k < ne; ++k) {
          //   for (index_t j = 0; j < nv; ++j) {
          //     eq_xx(k, j, idx) += tmp(k, j);
          //   }
          // }

        } else {
          for (index_t k = 0; k < ne; ++k) {
            for (index_t j = 0; j < nv; ++j) {
              eq_uu(k, j, idx) = (eq_u_(k, j) - eq_u(k, j)) / eps;
            }
          }
        }

        in_var = 0;
      }
    }
  }

  void compute_f_derivatives_2( //
      tensor_mut_view_t fxx,    //
      tensor_mut_view_t fux,    //
      tensor_mut_view_t fuu,    //
      mat_mut_view_t fx,        //
      mat_mut_view_t fu,        //
      index_t t,                //
      x_const x,                //
      u_const u,                //
      x_const x_next            //
  ) const {
    (void)x_next;

    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();

    compute_f_derivatives_first(fx, fu, t, x, u);
    // compute second derivatives
    {
      mat_t fx_{nv + nv, nv + nv};
      mat_t fu_{nv + nv, nv};

      vec_t x_ = x;
      vec_t u_ = u;

      vec_t dx = vec_t::Zero(nv + nv);
      vec_t du = vec_t::Zero(nv);

      using std::sqrt;
      scalar_t eps = sqrt(std::numeric_limits<scalar_t>::epsilon());

      for (index_t i = 0; i < 3 * nv; ++i) {
        bool at_x = (i < (2 * nv));
        index_t idx = at_x ? i : (i - 2 * nv);

        scalar_t& in_var = at_x ? dx[idx] : du[idx];

        in_var = eps;

        m_model.integrate(
            eigen::as_mut_view(x_.topRows(nq)),
            eigen::as_const_view(x.topRows(nq)),
            eigen::as_const_view(dx.topRows(nv)));
        x_.bottomRows(nv) = x.bottomRows(nv) + dx.bottomRows(nv);
        u_ = u + du;

        compute_f_derivatives_first(
            eigen::as_mut_view(fx_),
            eigen::as_mut_view(fu_),
            t,
            eigen::as_const_view(x_),
            eigen::as_const_view(u_));

        // mat_t commutator_jac{nv, nv};
        // compute_commutator_jacobian(eigen::as_mut_view(commutator_jac),
        // eigen::as_const_view(dx.topRows(nv)));

        if (at_x) {

          for (index_t k = 0; k < nv + nv; ++k) {
            for (index_t j = 0; j < nv + nv; ++j) {
              fxx(k, j, idx) = (fx_(k, j) - fx(k, j)) / eps;
            }
            for (index_t j = 0; j < nv; ++j) {
              fux(k, j, idx) = (fu_(k, j) - fu(k, j)) / eps;
            }
          }

          // mat_t tmp = -0.5 * fx.leftCols(nv) * commutator_jac;
          // for (index_t k = 0; k < nv + nv; ++k) {
          //   for (index_t j = 0; j < nv; ++j) {
          //     fxx(k, j, idx) -= 0.5 * tmp(k, j);
          //   }
          // }

        } else {
          for (index_t k = 0; k < nv + nv; ++k) {
            for (index_t j = 0; j < nv; ++j) {
              fuu(k, j, idx) = (fu_(k, j) - fu(k, j)) / eps;
            }
          }
        }

        in_var = 0;
      }
    }
  }

  void compute_f_derivatives( //
      tensor_mut_view_t fxx,  //
      tensor_mut_view_t fux,  //
      tensor_mut_view_t fuu,  //
      mat_mut_view_t fx,      //
      mat_mut_view_t fu,      //
      index_t t,              //
      x_const x,              //
      u_const u,              //
      x_const x_next          //
  ) const {

    index_t nq = m_model.configuration_dim();
    index_t nv = m_model.tangent_dim();

    auto q = eigen::as_const_view(x.topRows(nq));
    auto v = eigen::as_const_view(x.bottomRows(nv));

    // compute first derivatives
    {
      // v_out = dt * v_in;
      fx.col(0).bottomRows(nv) = dt * v;
      auto dt_v = eigen::as_const_view(fx.col(0).bottomRows(nv));

      // q_out = q_in + dt * v_in
      m_model.d_integrate_dq(eigen::as_mut_view(fx.topLeftCorner(nv, nv)), q, dt_v);
      m_model.d_integrate_dv(eigen::as_mut_view(fx.topRightCorner(nv, nv)), q, dt_v);
      fx.topRightCorner(nv, nv) *= dt;

      // v_out = acc
      fu.topRows(nv).setZero();
      m_model.d_dynamics_aba( //
          eigen::as_mut_view(fx.bottomLeftCorner(nv, nv)),
          eigen::as_mut_view(fx.bottomRightCorner(nv, nv)),
          eigen::as_mut_view(fu.bottomRows(nv)),
          q,
          v,
          u);

      // v_out = v_in + dt * v_out
      //       = v_in + dt * acc
      fx.bottomRows(nv) *= dt;
      fx.bottomRightCorner(nv, nv) += mat_t::Identity(nv, nv);
      fu.bottomRows(nv) *= dt;
    }
    // compute second derivatives
    // dx.T H dx = 2 * ((f(x + dx) - f(x)) - J dx)
    {
      auto f0 = x_next;

      vec_t f1{nq + nv};
      vec_t v1{nv};

      vec_t x1 = x;
      vec_t u1 = u;

      vec_t dx = vec_t::Zero(nv + nv);
      vec_t du = vec_t::Zero(nv);

      using std::sqrt;
      scalar_t eps = sqrt(sqrt(std::numeric_limits<scalar_t>::epsilon()));
      scalar_t eps2 = eps * eps;
      // compute diagonal of hessian
      for (index_t i = 0; i < 3 * nv; ++i) {
        bool at_x = (i < (2 * nv));
        index_t idx = at_x ? i : (i - 2 * nv);

        scalar_t& in_var = at_x ? dx[idx] : du[idx];
        auto f_col = at_x ? eigen::as_const_view(fx.col(idx)) : eigen::as_const_view(fu.col(idx));
        auto tensor = at_x ? fxx.as_dynamic() : fuu.as_dynamic();

        in_var = eps;

        m_model.integrate(
            eigen::as_mut_view(x1.topRows(nq)),
            eigen::as_const_view(x.topRows(nq)),
            eigen::as_const_view(dx.topRows(nv)));
        x1.bottomRows(nv) = x.bottomRows(nv) + dx.bottomRows(nv);
        u1 = u + du;

        eval_f_to(eigen::as_mut_view(f1), t, eigen::as_const_view(x1), eigen::as_const_view(u1));

        // f(x + dx) - f(x)
        auto df = eigen::as_mut_view(f1.bottomRows(nv + nv));
        // configuration part with model_t::difference
        m_model.difference(
            eigen::as_mut_view(v1),
            eigen::as_const_view(f0.topRows(nq)),
            eigen::as_const_view(f1.topRows(nq)));
        df.topRows(nv) = v1;
        // velocity part
        df.bottomRows(nv) = f1.bottomRows(nv) - f0.bottomRows(nv);

        // (f(x + dx) - f(x)) - J dx
        // dx = eps * e_i => J dx = eps * J.col(i)
        df -= eps * f_col;

        // 2 * ((f(x + dx) - f(x)) - J dx)
        df *= 2;

        for (index_t k = 0; k < nv + nv; ++k) {
          tensor(k, idx, idx) = df[k] / eps2;
        }

        in_var = 0;
      }

      // compute non diagonal part
      // ei H ej = ((ei + ej) H (ei + ej) - ei H ei - ej H ej) / 2
      for (index_t i = 0; i < 3 * nv; ++i) {
        bool at_x_1 = (i < 2 * nv);
        index_t idx_1 = at_x_1 ? i : (i - 2 * nv);

        scalar_t& in_var_1 = at_x_1 ? dx[idx_1] : du[idx_1];
        auto f_col_1 = at_x_1 ? eigen::as_const_view(fx.col(idx_1)) : eigen::as_const_view(fu.col(idx_1));

        auto tensor_1 = at_x_1 //
                            ? fxx.as_dynamic()
                            : fuu.as_dynamic();

        in_var_1 = eps;

        for (index_t j = i + 1; j < 3 * nv; ++j) {
          bool at_x_2 = (j < 2 * nv);
          index_t idx_2 = at_x_2 ? j : (j - 2 * nv);

          scalar_t& in_var_2 = at_x_2 ? dx[idx_2] : du[idx_2];
          auto f_col_2 = at_x_2 ? eigen::as_const_view(fx.col(idx_2)) : eigen::as_const_view(fu.col(idx_2));

          auto tensor_2 = at_x_2 //
                              ? fxx.as_dynamic()
                              : fuu.as_dynamic();

          auto tensor = at_x_1 ? (at_x_2 //
                                      ? fxx.as_dynamic()
                                      : fux.as_dynamic())
                               : (at_x_2 //
                                      ? (assert(false), fuu.as_dynamic())
                                      : fuu.as_dynamic());

          in_var_2 = eps;

          m_model.integrate(
              eigen::as_mut_view(x1.topRows(nq)),
              eigen::as_const_view(x.topRows(nq)),
              eigen::as_const_view(dx.topRows(nv)));
          x1.bottomRows(nv) = x.bottomRows(nv) + dx.bottomRows(nv);
          u1 = u + du;

          eval_f_to(eigen::as_mut_view(f1), t, eigen::as_const_view(x1), eigen::as_const_view(u1));

          // f(x + dx) - f(x)
          auto df = eigen::as_mut_view(f1.bottomRows(nv + nv));
          // configuration part with model_t::difference
          m_model.difference(
              eigen::as_mut_view(v1),
              eigen::as_const_view(f0.topRows(nq)),
              eigen::as_const_view(f1.topRows(nq)));
          df.topRows(nv) = v1;
          // velocity part
          df.bottomRows(nv) = f1.bottomRows(nv) - f0.bottomRows(nv);

          // (f(x + dx) - f(x)) - J dx
          // dx = eps * e_i => J dx = eps * J.col(i)
          df -= eps * f_col_1;
          df -= eps * f_col_2;

          // 2 * ((f(x + dx) - f(x)) - J dx)
          df *= 2;

          // [v1, f1] == f(x + eps * ei + eps * ej) - f(x)

          for (index_t k = 0; k < nv + nv; ++k) {
            tensor(k, idx_2, idx_1) =              //
                0.5 * (df[k] / eps2                //
                       - tensor_1(k, idx_1, idx_1) //
                       - tensor_2(k, idx_2, idx_2));

            if (at_x_1 == at_x_2) {
              tensor(k, idx_1, idx_2) = tensor(k, idx_2, idx_1);
            }
          }

          in_var_2 = 0;
        }

        in_var_1 = 0;
      }
    }
  }

  auto compute_derivatives(
      derivative_storage_t& derivs, trajectory_t const& traj, bool use_order2_finite_diff = false) const {

    derivs.lfx.setZero();
    derivs.lfxx.setZero();

    // clang-format off
    for (auto zipped : ranges::zip(
          derivs.lx, derivs.lu, derivs.lxx, derivs.lux, derivs.luu,
          derivs.fx, derivs.fu, derivs.fxx, derivs.fux, derivs.fuu,
          derivs.eq_val, derivs.eq_x, derivs.eq_u, derivs.eq_xx, derivs.eq_ux, derivs.eq_uu,
          traj)) {
      DDP_BIND(auto&&, (
          lx, lu, lxx, lux, luu,
          fx, fu, fxx, fux, fuu,
          eq_v, eq_x, eq_u, eq_xx, eq_ux, eq_uu,
          xu), zipped);
      // clang-format on

      assert(not xu.x().hasNaN());
      assert(not xu.x_next().hasNaN());
      assert(not xu.u().hasNaN());

      lx.get().setZero();
      lxx.get().setZero();
      lux.get().setZero();
      lu.get() = xu.u().transpose();
      luu.get().setIdentity();

      if (use_order2_finite_diff) {
        chronometer_t c("computing f derivatives");
        compute_f_derivatives(
            fxx.get(),
            fux.get(),
            fuu.get(),
            fx.get(),
            fu.get(),
            xu.current_index(),
            xu.x(),
            xu.u(),
            xu.x_next());
      } else {
        chronometer_t c("computing f derivatives");
        compute_f_derivatives_2(
            fxx.get(),
            fux.get(),
            fuu.get(),
            fx.get(),
            fu.get(),
            xu.current_index(),
            xu.x(),
            xu.u(),
            xu.x_next());
      }
      if (use_order2_finite_diff) {
        chronometer_t c("computing eq derivatives");
        compute_eq_derivatives(
            eq_xx.get(),
            eq_ux.get(),
            eq_uu.get(),
            eq_x.get(),
            eq_u.get(),
            eq_v.get(),
            xu.current_index(),
            xu.x(),
            xu.u());
      } else {
        chronometer_t c("computing eq derivatives");
        compute_eq_derivatives_2(
            eq_xx.get(),
            eq_ux.get(),
            eq_uu.get(),
            eq_x.get(),
            eq_u.get(),
            eq_v.get(),
            xu.current_index(),
            xu.x(),
            xu.u());
      }

      // finite difference check
      {
        assert(not eq_v.get().hasNaN());
        assert(not eq_x.get().hasNaN());
        assert(not eq_u.get().hasNaN());
        assert(not eq_xx.get().has_nan());
        assert(not eq_ux.get().has_nan());
        assert(not eq_uu.get().has_nan());

        assert(not fx.get().hasNaN());
        assert(not fu.get().hasNaN());
        assert(not fxx.get().has_nan());
        assert(not fux.get().has_nan());
        assert(not fuu.get().has_nan());

        auto nq = m_model.configuration_dim();
        auto nv = m_model.tangent_dim();
        auto ne = eq_v.get().rows();

        auto t = xu.current_index();
        auto x = xu.x();
        auto u = xu.u();

        scalar_t l;
        vec_t f{nq + nv};
        vec_t eq{ne};

        scalar_t l_;
        vec_t f_{nq + nv};
        vec_t eq_{ne};

        scalar_t eps_x = 1e-20;
        scalar_t eps_u = 1e-20;

        vec_t dx = eps_x * vec_t::Random(nv + nv);
        vec_t du = eps_u * vec_t::Random(nv);

        vec_t x_{nq + nv};
        m_model.integrate(
            eigen::as_mut_view(x_.topRows(nq)),
            eigen::as_const_view(x.topRows(nq)),
            eigen::as_const_view(dx.topRows(nv)));
        x_.bottomRows(nv) = x.bottomRows(nv) + dx.bottomRows(nv);

        vec_t u_ = u + du;

        l = this->l(t, x, u);
        eval_f_to(eigen::as_mut_view(f), t, x, u);
        eval_eq_to(eigen::as_mut_view(eq), t, x, u);

        l_ = this->l(t, eigen::as_const_view(x_), eigen::as_const_view(u_));
        eval_f_to(eigen::as_mut_view(f_), t, eigen::as_const_view(x_), eigen::as_const_view(u_));
        eval_eq_to(eigen::as_mut_view(eq_), t, eigen::as_const_view(x_), eigen::as_const_view(u_));

        scalar_t dl;
        vec_t df{nv + nv};
        vec_t deq{ne};

        scalar_t ddl;
        vec_t ddf{nv + nv};
        vec_t ddeq{ne};

        scalar_t dddl;
        vec_t dddf{nv + nv};
        vec_t dddeq{ne};

        dl = l_ - l;
        m_model.difference(
            eigen::as_mut_view(df.topRows(nv)),
            eigen::as_const_view(f.topRows(nq)),
            eigen::as_const_view(f_.topRows(nq)));
        df.bottomRows(nv) = f_.bottomRows(nv) - f.bottomRows(nv);
        deq = eq_ - eq;

        ddl = dl - (lx.get() * dx + lu.get() * du).value();
        ddf = df - (fx.get() * dx + fu.get() * du);
        ddeq = deq - (eq_x.get() * dx + eq_u.get() * du);

        dddl = ddl - (0.5 * dx.transpose() * lxx.get() * dx   //
                      + 0.5 * du.transpose() * luu.get() * du //
                      + du.transpose() * lux.get() * dx)
                         .value();
        dddf = -ddf;
        dddeq = -ddeq;

        add_second_order_term(dddf, fxx.get(), fux.get(), fuu.get(), dx, du);
        add_second_order_term(dddeq, eq_xx.get(), eq_ux.get(), eq_uu.get(), dx, du);

        dddf = -dddf;
        dddeq = -dddeq;

        if (l != 0) {
          assert(fabs(ddl) / fabs(dl) < sqrt(eps_x + eps_u));
          assert(fabs(dddl) / fabs(ddl) < sqrt(eps_x + eps_u));
        }

        assert(ddf.norm() / df.norm() < sqrt(eps_x + eps_u));
        assert(dddf.norm() / ddf.norm() < sqrt(eps_x + eps_u));

        assert(ddeq.norm() / deq.norm() < sqrt(eps_x + eps_u));
        assert(dddeq.norm() / ddeq.norm() < sqrt(eps_x + eps_u));
      }
    }
  }

  auto index_begin() const { return m_begin; }
  auto index_end() const { return m_end; }

  model_t const& m_model;
  index_t m_begin;
  index_t m_end;
  scalar_t dt;
  vec_t m_eq_ref = m_model._neutral_configuration();
};

auto main() -> int {

  using vec_t = Eigen::Matrix<scalar_t, -1, 1>;

#if 0
  auto model = problem_t::model_t::all_joints_test_model();
  constexpr static index_t horizon = 1;
#else
  auto model = problem_t::model_t{"~/pinocchio/models/others/robots/ur_description/urdf/ur5_gripper.urdf"};
  constexpr static index_t horizon = 5;
#endif
  auto nq = model.configuration_dim();
  auto nv = model.tangent_dim();
  auto x_init = [&] {
    Eigen::Matrix<scalar_t, Eigen::Dynamic, 1> x{nq + nv};

    model.neutral_configuration(eigen::as_mut_view(x.topRows(nq)));
    x.bottomRows(nv).setZero();
    return x;
  }();

  problem_t prob{model, 0, horizon, 0.01};
  auto u_idx = indexing::vec_regular_indexer(0, horizon, dyn_index{nv});
  auto eq_idx = indexing::vec_regular_indexer(0, horizon, dyn_index{nv});

  struct control_generator_t {
    using u_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::control_indexer_t>;
    using x_mat_t = eigen::matrix_from_idx_t<scalar_t, problem_t::state_indexer_t>;
    problem_t::control_indexer_t const& m_u_idx;
    index_t m_current_index = 0;
    u_mat_t m_value{m_u_idx.rows(m_current_index).value()};

    auto operator()() const -> eigen::view_t<u_mat_t const> { return eigen::as_const_view(m_value); }
    void next(eigen::view_t<x_mat_t const>) {
      ++m_current_index;
      m_value.resize(m_u_idx.rows(m_current_index).value());
    }
  };

  ddp_solver_t<problem_t> solver{model, prob, u_idx, eq_idx, x_init};

  {
    auto traj = solver.make_trajectory(control_generator_t{u_idx});

    for (auto xu : traj) {
      fmt::print( //
          "q : {}\n"
          "v : {}\n"
          "u : {}\n"
          "\n",
          xu.x().topRows(nq).transpose(),
          xu.x().bottomRows(nv).transpose(),
          xu.u().transpose());
    }

    auto d = solver.uninit_derivative_storage();
    auto d2 = solver.uninit_derivative_storage();
    prob.compute_derivatives(d, traj, true);
    prob.compute_derivatives(d2, traj, false);

    // testing derivatives
    auto eps_x = 1e-20;
    auto eps_u = 1e-20;
    for (auto zipped : ranges::zip(traj, d.f(), d.eq(), d2.eq())) {
      DDP_BIND(auto, (xu, f, eq, eq2), zipped);

      auto t = xu.current_index();
      auto x = xu.as_const().x();
      auto u = xu.as_const().u();

      vec_t x_h = x.eval();
      vec_t u_h = u.eval();

      vec_t dx = eps_x * vec_t::Random(nv + nv);
      vec_t du = eps_u * vec_t::Random(nv);

      // set up u + du, x + dx
      {
        x_h.bottomRows(nv) += dx.bottomRows(nv);
        model.integrate( //
            eigen::as_mut_view(x_h.topRows(nq)),
            eigen::as_const_view(x.topRows(nq)),
            eigen::as_const_view(dx.topRows(nv)));
        u_h += du;
      }

      auto print_norm_vec = [](auto const& v) { fmt::print("{} | {}\n", v.norm(), v.transpose()); };
      {
        vec_t f1{nq + nv};
        prob.eval_f_to(eigen::as_mut_view(f1), t, eigen::as_const_view(x_h), eigen::as_const_view(u_h));

        vec_t df{nv + nv};

        df.bottomRows(nv) = (f1 - xu.x_next()).bottomRows(nv);

        model.difference(
            eigen::as_mut_view(df.topRows(nv)),
            eigen::as_const_view(xu.x_next().topRows(nq)),
            eigen::as_const_view(f1.topRows(nq)));

        vec_t dxx = vec_t::Zero(nv + nv);
        vec_t dux = vec_t::Zero(nv + nv);
        vec_t duu = vec_t::Zero(nv + nv);

        for (index_t k = 0; k < nv + nv; ++k) {
          for (index_t i = 0; i < nv + nv; ++i) {
            for (index_t j = 0; j < nv + nv; ++j) {
              dxx[k] += f.xx(k, i, j) * dx[i] * dx[j];
            }
          }
          for (index_t i = 0; i < nv; ++i) {
            for (index_t j = 0; j < nv; ++j) {
              duu[k] += f.uu(k, i, j) * du[i] * du[j];
            }
          }
          for (index_t i = 0; i < nv; ++i) {
            for (index_t j = 0; j < nv + nv; ++j) {
              dux[k] += f.ux(k, i, j) * du[i] * dx[j];
            }
          }
        }

        fmt::print("{}\n", df.transpose());
        fmt::print("{}\n", (df - f.x * dx - f.u * du).transpose());
        fmt::print("{}\n", (df - f.x * dx - f.u * du - dxx / 2 - duu / 2 - dux).transpose());
      }

      {
        auto ne = eq.val.rows();
        vec_t eq1{ne};
        prob.eval_eq_to(eigen::as_mut_view(eq1), t, eigen::as_const_view(x_h), eigen::as_const_view(u_h));

        vec_t deq{ne};

        deq = eq1 - eq.val;

        vec_t dxx = vec_t::Zero(ne);
        vec_t dux = vec_t::Zero(ne);
        vec_t duu = vec_t::Zero(ne);

        for (index_t k = 0; k < ne; ++k) {
          for (index_t i = 0; i < nv + nv; ++i) {
            for (index_t j = 0; j < nv + nv; ++j) {
              dxx[k] += eq.xx(k, i, j) * dx[i] * dx[j];
            }
          }
          for (index_t i = 0; i < nv; ++i) {
            for (index_t j = 0; j < nv; ++j) {
              duu[k] += eq.uu(k, i, j) * du[i] * du[j];
            }
          }
          for (index_t i = 0; i < nv; ++i) {
            for (index_t j = 0; j < nv + nv; ++j) {
              dux[k] += eq.ux(k, i, j) * du[i] * dx[j];
            }
          }
        }

        print_norm_vec(deq);
        print_norm_vec(deq - eq.x * dx - eq.u * du);
        print_norm_vec(deq - eq.x * dx - eq.u * du - dxx / 2 - duu / 2 - dux);
      }

      {
        auto ne = eq2.val.rows();
        vec_t eq1{ne};
        prob.eval_eq_to(eigen::as_mut_view(eq1), t, eigen::as_const_view(x_h), eigen::as_const_view(u_h));

        vec_t deq{ne};

        deq = eq1 - eq2.val;

        vec_t dxx = vec_t::Zero(ne);
        vec_t dux = vec_t::Zero(ne);
        vec_t duu = vec_t::Zero(ne);

        for (index_t k = 0; k < ne; ++k) {
          for (index_t i = 0; i < nv + nv; ++i) {
            for (index_t j = 0; j < nv + nv; ++j) {
              dxx[k] += eq2.xx(k, i, j) * dx[i] * dx[j];
            }
          }
          for (index_t i = 0; i < nv; ++i) {
            for (index_t j = 0; j < nv; ++j) {
              duu[k] += eq2.uu(k, i, j) * du[i] * du[j];
            }
          }
          for (index_t i = 0; i < nv; ++i) {
            for (index_t j = 0; j < nv + nv; ++j) {
              dux[k] += eq2.ux(k, i, j) * du[i] * dx[j];
            }
          }
        }

        print_norm_vec(deq);
        print_norm_vec(deq - eq2.x * dx - eq2.u * du);
        print_norm_vec(deq - eq2.x * dx - eq2.u * du - dxx / 2 - duu / 2 - dux);
      }

      fmt::print("\n");
    }
  }

  {
    using std::pow;

    constexpr auto M = method::primal_dual_affine_multipliers;

    auto derivs = solver.uninit_derivative_storage();

    scalar_t const mu_init = 1e20;
    scalar_t mu = mu_init;
    scalar_t w = 1 / mu_init;
    scalar_t n = 1 / pow(mu_init, static_cast<scalar_t>(0.1L));
    scalar_t reg = 0;
    auto traj = solver.make_trajectory(control_generator_t{u_idx});
    auto new_traj = traj.clone();
    auto mults = solver.zero_multipliers<M>();

    prob.compute_derivatives(derivs, traj);
    auto bres = solver.backward_pass<M>(traj, mults, reg, mu, derivs);

    mu = bres.mu;
    // reg = bres.reg;
    for (auto fb : bres.feedback) {
      fmt::print("val: {}\n", fb.val().transpose());
      fmt::print("jac:\n{}\n\n", fb.val().transpose());
    }

    auto step = solver.forward_pass<M>(new_traj, traj, mults, bres, true);

    traj = new_traj.clone();

    for (auto xu : traj) {
      fmt::print("x: {}\nu: {}\n", xu.x().transpose(), xu.u().transpose());
    }

    for (index_t t = 0; t < 50; ++t) {
      auto mult_update_rv = solver.update_derivatives<M>(derivs, bres.feedback, mults, traj, mu, w, n);
      switch (mult_update_rv) {
      case mult_update_attempt_result_e::no_update:
        break;
      case mult_update_attempt_result_e::update_failure:
        mu *= 10;
        break;
      case mult_update_attempt_result_e::update_success:
        n /= pow(mu, static_cast<scalar_t>(0.9L));
        w /= mu;
        fmt::print("updated multipliers\n");
      }

      bres = solver.backward_pass<M>(traj, mults, reg, mu, derivs);
      mu = bres.mu;
      // reg = bres.reg;
      fmt::print("mu: {:20}   reg: {:20}\n", mu, reg);

      step = solver.forward_pass<M>(new_traj, traj, mults, bres, true);
      if (step >= 0.5) {
        // reg /= 2;
      }

      traj = new_traj.clone();
      fmt::print("step: {}\n", step);
      fmt::print("eq: ");
      for (auto eq : derivs.eq()) {
        fmt::print("{} ", eq.val.norm());
      }
      fmt::print("\n");
    }
  }
}
