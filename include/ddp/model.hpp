#ifndef MODEL_HPP_NMIGPXUO
#define MODEL_HPP_NMIGPXUO

#include "ddp/detail/utils.hpp"
#include "ddp/zero.hpp"

namespace ddp {

template <typename T, typename JX, typename JU, typename HXX, typename HUX, typename HUU>
struct bivariate_quadratic_model_t {

  T m_val;   // scalar
  JX m_jx;   // [1, x_dim]
  JU m_ju;   // [1, u_dim]
  HXX m_hxx; // [x_dim, x_dim]
  HUX m_hux; // [u_dim, x_dim]
  HUU m_huu; // [u_dim, u_dim]

  using scalar_t = T;

  static_assert(JX::RowsAtCompileTime == 1, "");
  static_assert(JU::RowsAtCompileTime == 1, "");

  static_assert(JX::ColsAtCompileTime == HXX::RowsAtCompileTime, "");
  static_assert(JX::ColsAtCompileTime == HXX::ColsAtCompileTime, "");

  static_assert(JU::ColsAtCompileTime == HUU::RowsAtCompileTime, "");
  static_assert(JU::ColsAtCompileTime == HUU::ColsAtCompileTime, "");

  static_assert(JU::ColsAtCompileTime == HUX::RowsAtCompileTime, "");
  static_assert(JX::ColsAtCompileTime == HUX::ColsAtCompileTime, "");

  static_assert(std::is_same<T, typename JX::Scalar>::value, "");
  static_assert(std::is_same<T, typename JU::Scalar>::value, "");
  static_assert(std::is_same<T, typename HXX::Scalar>::value, "");
  static_assert(std::is_same<T, typename HUX::Scalar>::value, "");
  static_assert(std::is_same<T, typename HUU::Scalar>::value, "");

  auto x_dim() const noexcept -> eigen::col_kind<JX> { return eigen::cols(m_jx); }
  auto u_dim() const noexcept -> eigen::col_kind<JU> { return eigen::cols(m_ju); }

  auto eval(                                                             //
      eigen::view_t<Eigen::Matrix<T, JX::ColsAtCompileTime, 1> const> x, //
      eigen::view_t<Eigen::Matrix<T, JU::ColsAtCompileTime, 1> const> u  //
  ) const -> scalar_t {

    assert(x.rows() == m_jx.cols());
    assert(u.rows() == m_ju.cols());

    return m_val +                            //
           (m_jx * x).value() +               //
           (m_ju * u).value() +               //
           u.transpose() * m_hux * x +        //
           0.5 * (x.transpose() * m_hxx * x + //
                  u.transpose() * m_huu * u);
  }

  auto
  j_x(eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const> x, //
      eigen::view_t<Eigen::Matrix<scalar_t, JU::ColsAtCompileTime, 1> const> u  //
  ) const noexcept -> decltype(m_jx + x.transpose() * m_hxx + u.transpose() * m_hux) {
    assert(x.rows() == m_jx.cols());
    assert(u.rows() == m_ju.cols());

    return m_jx + x.transpose() * m_hxx + u.transpose() * m_hux;
  }

  auto
  j_u(eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const> x, //
      eigen::view_t<Eigen::Matrix<scalar_t, JU::ColsAtCompileTime, 1> const> u  //
  ) const noexcept -> decltype(m_ju + u.transpose() * m_huu + x.transpose() * m_hux.transpose()) {
    assert(x.rows() == m_jx.cols());
    assert(u.rows() == m_ju.cols());

    return m_ju + u.transpose() * m_huu + x.transpose() * m_hux.transpose();
  }

  auto h_xx(
      eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const>, //
      eigen::view_t<Eigen::Matrix<scalar_t, JU::ColsAtCompileTime, 1> const>  //
  ) const noexcept -> decltype(eigen::as_const_view(m_hxx)) {
    return eigen::as_const_view(m_hxx);
  }

  auto h_ux(
      eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const>, //
      eigen::view_t<Eigen::Matrix<scalar_t, JU::ColsAtCompileTime, 1> const>  //
  ) const noexcept -> decltype(eigen::as_const_view(m_hux)) {
    return eigen::as_const_view(m_hxx);
  }

  auto h_uu(
      eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const>, //
      eigen::view_t<Eigen::Matrix<scalar_t, JU::ColsAtCompileTime, 1> const>  //
  ) const noexcept -> decltype(eigen::as_const_view(m_huu)) {
    return eigen::as_const_view(m_hxx);
  }

  struct derivs_1st_t {
    scalar_t val;
    decltype((-m_jx).eval()) x;
    decltype((-m_ju).eval()) u;
  };

  struct derivs_2nd_t {
    scalar_t val;
    decltype((-m_jx).eval()) x;
    decltype((-m_ju).eval()) u;
    decltype(eigen::as_const_view(m_hxx)) xx;
    decltype(eigen::as_const_view(m_hux)) ux;
    decltype(eigen::as_const_view(m_huu)) uu;
  };

  auto derivs_1st(
      eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const> x, //
      eigen::view_t<Eigen::Matrix<scalar_t, JU::ColsAtCompileTime, 1> const> u  //
  ) const -> derivs_1st_t {
    return {
        this->eval(x, u),
        this->j_x(x, u).eval(),
        this->j_u(x, u).eval(),
    };
  }

  auto derivs_2nd(
      eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const> x, //
      eigen::view_t<Eigen::Matrix<scalar_t, JU::ColsAtCompileTime, 1> const> u  //
  ) const -> derivs_2nd_t {
    return {
        this->eval(x, u),
        this->j_x(x, u).eval(),
        this->j_u(x, u).eval(),
        eigen::as_const_view(m_hxx),
        eigen::as_const_view(m_hux),
        eigen::as_const_view(m_huu),
    };
  }
};

template <typename T, typename JX, typename HXX>
struct univariate_quadratic_model_t {

  T m_val;   // scalar
  JX m_jx;   // [1, x_dim]
  HXX m_hxx; // [x_dim, x_dim]

  using scalar_t = T;

  static_assert(JX::RowsAtCompileTime == 1, "");

  static_assert(JX::ColsAtCompileTime == HXX::RowsAtCompileTime, "");
  static_assert(JX::ColsAtCompileTime == HXX::ColsAtCompileTime, "");

  static_assert(std::is_same<T, typename JX::Scalar>::value, "");
  static_assert(std::is_same<T, typename HXX::Scalar>::value, "");

  auto x_dim() const noexcept -> eigen::col_kind<JX> { return eigen::cols(m_jx); }

  auto eval(eigen::view_t<Eigen::Matrix<T, JX::ColsAtCompileTime, 1> const> x) const -> scalar_t {

    assert(x.rows() == m_jx.cols());

    return m_val +              //
           (m_jx * x).value() + //
           0.5 * (x.transpose() * m_hxx * x);
  }

  auto j_x(eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const> x) const noexcept
      -> decltype(m_jx + x.transpose() * m_hxx) {
    assert(x.rows() == m_jx.cols());
    return m_jx + x.transpose() * m_hxx;
  }

  auto h_xx(eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const>) const noexcept
      -> decltype(eigen::as_const_view(m_hxx)) {
    return eigen::as_const_view(m_hxx);
  }

  struct derivs_1st_t {
    scalar_t val;
    decltype((-m_jx).eval()) x;
  };

  struct derivs_2nd_t {
    scalar_t val;
    decltype((-m_jx).eval()) x;
    decltype(eigen::as_const_view(m_hxx)) xx;
  };

  auto derivs_1st(eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const> x) const -> derivs_1st_t {
    return {
        this->eval(x),
        this->j_x(x).eval(),
    };
  }

  auto derivs_2nd(eigen::view_t<Eigen::Matrix<scalar_t, JX::ColsAtCompileTime, 1> const> x) const -> derivs_2nd_t {
    return {
        this->eval(x),
        this->j_x(x).eval(),
        this->j_u(x).eval(),
        eigen::as_const_view(m_hxx),
    };
  }
};

template <typename T, typename JX, typename JU, typename HXX, typename HUX, typename HUU>
auto bivariate_quadratic_model(T val, JX jx, JU ju, HXX hxx, HUX hux, HUU huu) noexcept
    -> bivariate_quadratic_model_t<T, JX, JU, HXX, HUX, HUU> {
  return {
      DDP_MOVE(val),
      DDP_MOVE(jx),
      DDP_MOVE(ju),
      DDP_MOVE(hxx),
      DDP_MOVE(hux),
      DDP_MOVE(huu),
  };
}

template <typename T, typename JX, typename HXX>
auto univariate_quadratic_model(T val, JX jx, HXX hxx) noexcept -> univariate_quadratic_model_t<T, JX, HXX> {
  return {
      DDP_MOVE(val),
      DDP_MOVE(jx),
      DDP_MOVE(hxx),
  };
}
} // namespace ddp

#endif /* end of include guard MODEL_HPP_NMIGPXUO */
