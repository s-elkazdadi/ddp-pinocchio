#ifndef HANDLE_AS_MPFR_HPP_H5QMPTM4
#define HANDLE_AS_MPFR_HPP_H5QMPTM4

#include "mpfr/detail/mpfr.hpp"
#include "mpfr/detail/prologue.hpp"

namespace mpfr {
namespace _ {

template <typename T> struct void_impl { using type = void; };

template <typename T> struct to_mpfr_ptr { using type = void; };
template <precision_t P> struct to_mpfr_ptr<mp_float_t<P>> { using type = mpfr_ptr; };
template <precision_t P> struct to_mpfr_ptr<mp_float_t<P>&> { using type = mpfr_ptr; };
template <precision_t P> struct to_mpfr_ptr<mp_float_t<P> const> { using type = mpfr_srcptr; };
template <precision_t P> struct to_mpfr_ptr<mp_float_t<P> const&> { using type = mpfr_srcptr; };

template <typename Enable, typename T, typename... Args> struct invocable_impl {
  static constexpr bool value = false;
  static constexpr bool nothrow_value = false;
  using type = void;
};

#define MPFR_CXX_DECLVAL(...) (*(static_cast < __VA_ARGS__ && (*)() noexcept > (nullptr)))()

template <typename Fn, typename... Args>
struct invocable_impl<
    typename void_impl<decltype(MPFR_CXX_DECLVAL(Fn)(MPFR_CXX_DECLVAL(Args)...))>::type,
    Fn,
    Args...> {
  static constexpr bool value = true;
  static constexpr bool nothrow_value = noexcept(MPFR_CXX_DECLVAL(Fn)(MPFR_CXX_DECLVAL(Args)...));
  using type = decltype(MPFR_CXX_DECLVAL(Fn)(MPFR_CXX_DECLVAL(Args)...));
};

#undef MPFR_CXX_DECLVAL

template <typename Fn, typename... Args> struct is_invocable : invocable_impl<void, Fn, Args...> {};

template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T&> { using type = T; };

template <typename T> struct is_const { static constexpr bool value = false; };
template <typename T> struct is_const<T const> { static constexpr bool value = true; };

namespace meta {

template <typename... Ts> struct type_list;
template <typename T> struct call_operator_signature;

template <typename T, typename Ret, typename... Args>
struct call_operator_signature<Ret (T::*)(Args...)> {
  using args = type_list<Args...>;
};

template <typename T, typename Ret, typename... Args>
struct call_operator_signature<Ret (T::*)(Args...) const> {
  using args = type_list<Args...>;
};

template <typename T> struct void_impl { using type = void; };
template <typename T> using void_t = typename void_impl<T>::type;

template <typename T, typename Enable = void> struct callable_info {
  static constexpr bool ambiguous = true;
  using args = type_list<>;
};
template <typename T> struct callable_info<T, void_t<decltype(&T::operator())>> {
  static constexpr bool ambiguous = false;
  using args = typename call_operator_signature<decltype(&T::operator())>::args;
};

template <typename Ret, typename... Args> struct callable_info<Ret (*)(Args...), void> {
  static constexpr bool ambiguous = false;
  using args = type_list<Args...>;
};

template <typename Ret, typename... Args> struct callable_info<Ret(Args...), void> {
  static constexpr bool ambiguous = false;
  using args = type_list<Args...>;
};

template <typename T> struct is_mpfr_ptr { static constexpr bool value = false; };
template <> struct is_mpfr_ptr<mpfr_ptr> { static constexpr bool value = true; };
template <> struct is_mpfr_ptr<mpfr_srcptr> { static constexpr bool value = true; };

template <typename T> struct all_mpfr_ptr;
template <typename... Ts> struct all_mpfr_ptr<type_list<Ts...>> {
  static constexpr bool value = all_of({is_mpfr_ptr<Ts>::value...});
};

template <bool Ambiguous> struct impl_handle_as_mpfr_t_ambiguity_dispatch {
  // true case
  template <typename Return, bool No_Except, typename Fn, typename... Args>
  static auto run(Fn&& fn, Args&... args) noexcept(No_Except) -> Return {
    return static_cast<Fn&&>(fn)(_::into_mpfr<                                               //
                                 _::is_const<                                                //
                                     typename _::remove_reference<Args>::type                //
                                     >::value                                                //
                                 >::get_pointer(_::into_mpfr<                                //
                                                _::is_const<                                 //
                                                    typename _::remove_reference<Args>::type //
                                                    >::value                                 //
                                                >::get_mpfr(args))...);
  }
};

template <typename Ptr> struct add_const_if;

template <> struct add_const_if<mpfr_srcptr> { template <typename T> using type = T const; };
template <> struct add_const_if<mpfr_ptr> { template <typename T> using type = T; };

template <bool All_Mpfr_Ptr> struct all_mpfr_ptr_dispatch {
  // true case
  template <typename Return, bool No_Except, typename Fn, typename... Args>
  static auto run(Fn&& fn, Args&... args) noexcept(No_Except) -> Return {
    using callable_info = meta::callable_info<typename remove_reference<Fn>::type>;
    return all_mpfr_ptr_dispatch::run2<Return, No_Except>(
        static_cast<typename callable_info::args*>(nullptr), static_cast<Fn&&>(fn), args...);
  }

  template <typename Return, bool No_Except, typename Fn, typename... Args, typename... Fn_Params>
  static auto run2(type_list<Fn_Params...>* /*unused*/, Fn&& fn, Args&... args) noexcept(No_Except) -> Return {

    return impl_handle_as_mpfr_t_ambiguity_dispatch<true>::run<Return, No_Except>(
        static_cast<Fn&&>(fn),
        static_cast<typename add_const_if<Fn_Params>::template type<Args>&>(args)...);
  }
};

template <> struct all_mpfr_ptr_dispatch<false> {
  template <typename Return, bool No_Except, typename Fn, typename... Args>
  static auto run(Fn&& fn, Args&... args) noexcept(No_Except) -> Return {
    return impl_handle_as_mpfr_t_ambiguity_dispatch<true>::run<Return, No_Except>(
        static_cast<Fn&&>(fn), args...);
  }
};

template <> struct impl_handle_as_mpfr_t_ambiguity_dispatch<false> {
  template <typename Return, bool No_Except, typename Fn, typename... Args>
  static auto run(Fn&& fn, Args&... args) noexcept(No_Except) -> Return {
    using callable_info = meta::callable_info<typename remove_reference<Fn>::type>;
    return all_mpfr_ptr_dispatch<all_mpfr_ptr<typename callable_info::args>::value>::
        template run<Return, No_Except>(static_cast<Fn&&>(fn), args...);
  }
};

} // namespace meta

template <bool No_Except, typename Fn, typename... Args>
auto impl_handle_as_mpfr_t(Fn&& fn, Args&... args) noexcept(No_Except) ->
    typename ::mpfr::_::is_invocable<Fn, typename mpfr::_::to_mpfr_ptr<Args>::type...>::type {
  return //
      meta::impl_handle_as_mpfr_t_ambiguity_dispatch<
          meta::callable_info<typename remove_reference<Fn>::type>::ambiguous>::template run< //
          typename ::mpfr::_::is_invocable<Fn, typename mpfr::_::to_mpfr_ptr<Args>::type...>::
              type, //
          No_Except //
          >         //
      (static_cast<Fn&&>(fn), args...);
}

} // namespace _
} // namespace mpfr

#include "mpfr/detail/epilogue.hpp"

#endif /* end of include guard HANDLE_AS_MPFR_HPP_H5QMPTM4 */
