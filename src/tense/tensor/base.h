/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2021, Shahriar Rezghi <shahriar25.ss@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <tense/tensor/expr.h>

#include <limits>
#include <random>

#define _USE_MATH_DEFINES
#include <cmath>

#define _UNARY0(NAME, TYPE, FUNC)                           \
    auto NAME() const                                       \
    {                                                       \
        return unary<TYPE>([](auto val1) { return FUNC; }); \
    }

#define _UNARY1(NAME, TYPE, VTYPE, FUNC)                        \
    auto NAME(VTYPE val2) const                                 \
    {                                                           \
        return unary<TYPE>([val2](auto val1) { return FUNC; }); \
    }

#define _BINARY(NAME, TYPE, FUNC)                                                     \
    template <typename Expr2, typename = IsExpr<Expr2>>                               \
    auto NAME(const Expr2 &expr2) const                                               \
    {                                                                                 \
        return binary<TYPE, Expr2>(expr2, [](auto val1, auto val2) { return FUNC; }); \
    }

#define _REDUCE0(NAME, TYPE, INIT, FUNC)                           \
    auto NAME(Size dim = 0) const                                  \
    {                                                              \
        return reduce<TYPE>(                                       \
            INIT, [](auto val1, auto val2) { return FUNC; }, dim); \
    }

#define _REDUCE1(NAME, TYPE, VTYPE, INIT, FUNC)                        \
    auto NAME(VTYPE val3, Size dim = 0) const                          \
    {                                                                  \
        return reduce<TYPE>(                                           \
            INIT, [val3](auto val1, auto val2) { return FUNC; }, dim); \
    }

#define UNARY0(NAME, FUNC) _UNARY0(NAME, typename Derived::Type, FUNC)
#define UNARY1(NAME, TYPE, FUNC) _UNARY1(NAME, typename Derived::Type, TYPE, FUNC)
#define BINARY(NAME, FUNC) _BINARY(NAME, typename Derived::Type, FUNC)
#define REDUCE0(NAME, INIT, FUNC) _REDUCE0(NAME, typename Derived::Type, INIT, FUNC)
#define REDUCE1(NAME, TYPE, INIT, FUNC) _REDUCE1(NAME, TYPE, typename Derived::Type, INIT, FUNC)

#define OPERATOR0(NAME, OP) \
    auto operator OP() const { return derived().NAME(); }

#define OPERATOR1(NAME, RNAME, OP)                                       \
    template <typename Expr2, typename = IsExpr<Expr2>>                  \
    auto operator OP(const Expr2 &expr2) const                           \
    {                                                                    \
        return derived().NAME(expr2);                                    \
    }                                                                    \
    auto operator OP(Type expr2) const { return derived().NAME(expr2); } \
    friend auto operator OP(Type expr2, const Derived &expr1) { return expr1.RNAME(expr2); }

namespace Tense::TensorImpl
{
template <typename Type, typename Derived>
class Base : Expr
{
    const Derived &derived() const { return *static_cast<const Derived *>(this); }

public:
    template <typename T, typename Func>
    auto unary(Func func) const
    {
        return Unary<T, Derived, Func>(derived(), func);
    }
    template <typename T, typename Expr2, typename Func, typename = IsExpr<Expr2>>
    auto binary(const Expr2 &expr2, Func func) const
    {
        auto shape1 = Helper::remove(derived().shape());
        auto shape2 = Helper::remove(expr2.shape());
        Size size = Helper::check(shape1, shape2);
        return Binary<T, Derived, Expr2, Func>(derived(), expr2, func, shape1.size() >= shape2.size(),
                                               shape1.size() <= shape2.size(), size);
    }
    template <typename T, typename Func>
    auto reduce(T init, Func func, Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        auto shape = Helper::left(derived().shape(), dim);
        return Reduce<T, Derived, Func>(derived(), func, shape, size, init);
    }
    auto view(const Shape &indexes) const
    {
        auto shape = derived().shape();
        TENSE_TASSERT(indexes.size(), <=, shape.size(), "view", "Input dimension can't be higher than tensor dimension")
        auto index = Helper::view(shape, indexes);
        shape = Helper::right(shape, indexes.size());
        return View<Derived>(derived(), shape, index);
    }
    auto strided(const Shape &indexes) const
    {
        const auto &shape = derived().shape();
        auto dim = shape.size() - indexes.size();
        TENSE_TASSERT(indexes.size(), <=, shape.size(), "stride",
                      "Input dimension can't be higher than tensor dimension")
        auto index = Helper::item(shape, indexes), step = Helper::elems(shape, dim);
        return Strided<Derived>(derived(), Helper::left(shape, dim), index, step);
    }
    auto repeat(Shape shape) const
    {
        Helper::check(shape);
        const Shape &shape2 = derived().shape();
        shape.insert(shape.end(), shape2.begin(), shape2.end());
        return Repeat<Derived>(derived(), shape, Helper::elems(derived().shape()));
    }

    template <int P>
    auto pow() const
    {
        return Power<P, Derived>(derived());
    }
    auto flip(Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        return Flip<Derived>(derived(), size);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto indirect(const Expr2 &expr2) const
    {
        TENSE_TASSERT(derived().shape(), ==, expr2.shape(), "indirect", "Shapes of tensors must be equal")
        return Indirect<Derived, Expr2>(derived(), expr2);
    }
    template <typename T>
    auto type() const
    {
        if constexpr (IsComplex<Type>::value && !IsComplex<T>::value)
            return derived().real().template type<T>();
        else
            return Convert<T, Derived>(derived());
    }
    auto reshape(Shape shape) const
    {
        Helper::replace(shape, Helper::elems(derived().shape()));
        return Reshape<Derived>(derived(), shape);
    }

    template <typename Func>
    auto where(Func func, Type val = Type(0)) const
    {
        return FWhere<Derived, Func>(derived(), func, val);
    }
    template <typename Func>
    auto iwhere(Func func, Type val = Type(0)) const
    {
        return IWhere<Derived, Func>(derived(), func, val);
    }
    template <typename Func, typename Expr2, typename = IsExpr<Expr2>>
    auto where(Func func, const Expr2 &expr2) const
    {
        TENSE_TASSERT(this->shape(), ==, expr2.shape(), "where", "Shape of tensors must be equal")
        return FEWhere<Derived, Expr2, Func>(derived(), expr2, func);
    }
    template <typename Func, typename Expr2, typename = IsExpr<Expr2>>
    auto iwhere(Func func, const Expr2 &expr2) const
    {
        TENSE_TASSERT(this->shape(), ==, expr2.shape(), "iwhere", "Shape of tensors must be equal")
        return IEWhere<Derived, Expr2, Func>(derived(), expr2, func);
    }

    auto all(Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        auto shape = Helper::left(derived().shape(), dim);
        return All<Derived>(derived(), shape, size);
    }
    auto any(Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        auto shape = Helper::left(derived().shape(), dim);
        return Any<Derived>(derived(), shape, size);
    }
    auto minidx(Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        auto shape = Helper::left(derived().shape(), dim);
        return MinIdx<Derived>(derived(), shape, size);
    }
    auto maxidx(Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        auto shape = Helper::left(derived().shape(), dim);
        return MaxIdx<Derived>(derived(), shape, size);
    }

    auto polar() const { return Polar<Derived>(derived()); }
    auto expr() const { return Self<Derived>(derived()); }
    auto eval() const { return Tensor<Type>(derived()); }
    Type item() const { return derived()[0]; }

    template <typename Major>
    auto matrix() const
    {
        const auto &shape = derived().shape();
        TENSE_TASSERT(shape.size(), ==, 2, "matrix", "Dimension of tensor must be 2")
        return MatrixImpl::ToMatrix<Major, Derived>(derived(), shape[0], shape[1]);
    }
    template <typename... Args>
    Type operator()(Args &&...args) const
    {
        Shape indexes({std::forward<Args>(args)...});
        TENSE_TASSERT(derived().shape().size(), ==, indexes.size(), "operator()",
                      "Tensor and indexes dimension must be the same")
        auto index = Helper::item(derived().shape(), indexes);
        return derived()[index];
    }

    ////////////////////////////////////////////////////////////////////////////

    template <int P>
    auto norm() const
    {
        static_assert(P > 0, "P can't be zero in Matrix::norm");

        if constexpr (P == Inf)
            return derived().abs().max();
        else if constexpr (P == 1)
            return derived().abs().sum();
        else if constexpr (P == 2)
            return derived().abs().square().sum().sqrt();
        else
            return derived().abs().template pow<P>().sum().pow(1.f / P);
    }
    template <Size P, typename Expr2, typename = IsExpr<Expr2>>
    auto distance(const Expr2 &expr2) const
    {
        return derived().sub(expr2).template norm<P>();
    }
    template <Size P>
    auto distance(Type expr2) const
    {
        return derived().sub(expr2).template norm<P>();
    }

    auto var(Size dof = 0, Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        auto expr1 = derived().square().sum(dim);
        auto expr2 = derived().sum(dim).square() / size;
        return (expr1 - expr2) / (size - dof);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto cov(const Expr2 &expr2, Size dof = 0, Size dim = 0) const
    {
        TENSE_TASSERT(this->shape(), ==, expr2.shape(), "cov", "Shapes of tensors must be the same")

        auto size = Helper::check(derived().shape(), dim);
        auto _expr1 = derived().mul(expr2).sum(dim);
        auto _expr2 = derived().sum(dim) * expr2.sum(dim) / size;
        return (_expr1 - _expr2) / (size - dof);
    }
    auto mean(Size dim = 0) const
    {
        auto size = Helper::check(derived().shape(), dim);
        auto shape = Helper::left(derived().shape(), dim);
        auto func = [](auto val1, auto val2) { return val1 + val2; };
        return Reduce<Type, Derived, decltype(func)>(derived(), shape, size) / size;
    }
    auto normalize(Size dim = 0) const
    {
        auto min = derived().min(dim), dist = (derived().max(dim) - min);
        return (derived() - min) / dist;
    }
    auto standize(Size dof = 0, Size dim = 0) const
    {
        auto mean = derived().mean(dim);
        auto std = derived().std(dof, dim);
        return (derived() - mean) / std;
    }

    auto std(Size dof = 0, Size dim = 0) const { return derived().var(dof, dim).sqrt(); }
    auto contains(Type expr2, Size dim = 0) const { return derived().eq(expr2).any(dim); }
    auto equal(Type expr2, Size dim = 0) const { return derived().eq(expr2).all(dim); }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto equal(const Expr2 &expr2, Size dim = 0) const
    {
        return derived().eq(expr2).all(dim);
    }
    auto close(Type expr2, typename FromComplex<Type>::Type expr3 = 1e-5) const
    {
        return derived().sub(expr2).abs().le(expr3).all();
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto close(const Expr2 &expr2, typename FromComplex<Type>::Type expr3 = 1e-5) const
    {
        return derived().sub(expr2).abs().le(expr3).all();
    }
    auto abs() const
    {
        return unary<typename FromComplex<Type>::Type>([](auto val1) { return std::abs(val1); });
    }
    auto clip(Type val2, Type val3) const
    {
        if (val2 > val3) std::swap(val2, val3);
        return unary<Type>([val2, val3](auto val1) { return std::min(std::max(val1, val2), val3); });
    }

    ////////////////////////////////////////////////////////////////////////////

    static auto init(const Shape &shape, const std::vector<Type> &list)
    {
        Helper::check(shape);
        TENSE_TASSERT(Helper::elems(shape), ==, list.size(), "init", "List size must be equal to input size");
        return Initial<Type>(shape, list);
    }
    static auto init(const Shape &shape, const std::initializer_list<Type> &list)
    {
        return init(shape, std::vector<Type>(list));
    }
    static auto init(const Shape &shape, Type val)
    {
        Helper::check(shape);
        return Constant<Type>(shape, val);
    }
    template <typename Dist>
    static auto dist(const Shape &shape, Dist dist)
    {
        Helper::check(shape);
        return Distribution<Type, Dist>(shape, dist);
    }
    static auto seq(const Shape &shape, Type start, Type end)
    {
        Helper::check(shape);
        auto step = (end - start) / Helper::elems(shape);
        return Sequence<Type>(shape, start, step);
    }
    static auto cat(const std::vector<Tensor<Type>> &list, Size dim = 0)
    {
        auto shape = list[0].shape();
        TENSE_TASSERT(list.empty(), ==, false, "cat", "Input list can't be empty")
        for (const auto &tensor : list)
            TENSE_TASSERT(tensor.shape(), ==, shape, "cat", "Tensors must have the same shape")
        TENSE_TASSERT(shape.size(), >=, dim, "cat", "Input dimension can't be higher than dimension of input tensors")

        auto size = Helper::elems(shape, dim);
        if (dim == 0)
            shape.insert(shape.begin(), list.size());
        else
            shape[dim - 1] *= list.size();
        return Concat<Type>(list, shape, size);
    }

    ////////////////////////////////////////////////////////////////////////////

    static auto numin(const Shape &shape) { return init(shape, std::numeric_limits<Type>::min()); }
    static auto numax(const Shape &shape) { return init(shape, std::numeric_limits<Type>::max()); }
    static auto lowest(const Shape &shape) { return init(shape, std::numeric_limits<Type>::lowest()); }
    static auto epsilon(const Shape &shape) { return init(shape, std::numeric_limits<Type>::epsilon()); }
    static auto inf(const Shape &shape) { return init(shape, std::numeric_limits<Type>::infinity()); }
    static auto nan(const Shape &shape) { return init(shape, std::numeric_limits<Type>::quiet_NaN()); }
    static auto zeros(const Shape &shape) { return init(shape, 0); }
    static auto ones(const Shape &shape) { return init(shape, 1); }
    static auto e(const Shape &shape) { return init(shape, M_E); }
    static auto pi(const Shape &shape) { return init(shape, M_PI); }
    static auto sqrt2(const Shape &shape) { return init(shape, M_SQRT2); }

    ////////////////////////////////////////////////////////////////////////////

    static auto uniform(const Shape &shape, Type a = 0, Type b = 1)
    {
        if constexpr (std::is_integral<Type>::value)
            return dist(shape, std::uniform_int_distribution<Type>(a, b));
        else if constexpr (std::is_floating_point<Type>::value)
            return dist(shape, std::uniform_real_distribution<Type>(a, b));
        throw std::runtime_error("Data type not supported for uniform operation in tensor::uniform.");
    }
    static auto bernoulli(const Shape &shape, double p = 0.5)  //
    {
        return dist(shape, std::bernoulli_distribution(p));
    }
    static auto binomial(const Shape &shape, int t, double p = 0.5)  //
    {
        return dist(shape, std::binomial_distribution(t, p));
    }
    static auto geometric(const Shape &shape, double p = 0.5)  //
    {
        return dist(shape, std::geometric_distribution(p));
    }
    static auto pascal(const Shape &shape, int k, double p = 0.5)  //
    {
        return dist(shape, std::negative_binomial_distribution(k, p));
    }
    static auto poisson(const Shape &shape, double mean = 1)  //
    {
        return dist(shape, std::poisson_distribution(mean));
    }
    static auto exponential(const Shape &shape, double lambda = 1)  //
    {
        return dist(shape, std::exponential_distribution(lambda));
    }
    static auto gamma(const Shape &shape, double alpha = 1, double beta = 1)  //
    {
        return dist(shape, std::gamma_distribution(alpha, beta));
    }
    static auto weibull(const Shape &shape, double a = 1, double b = 1)  //
    {
        return dist(shape, std::weibull_distribution(a, b));
    }
    static auto extremevalue(const Shape &shape, double a = 0, double b = 1)  //
    {
        return dist(shape, std::extreme_value_distribution(a, b));
    }
    static auto normal(const Shape &shape, double mean = 0, double std = 1)  //
    {
        return dist(shape, std::normal_distribution(mean, std));
    }
    static auto lognormal(const Shape &shape, double m = 0, double s = 1)  //
    {
        return dist(shape, std::lognormal_distribution(m, s));
    }
    static auto chisquared(const Shape &shape, double n = 1)  //
    {
        return dist(shape, std::chi_squared_distribution(n));
    }
    static auto cauchy(const Shape &shape, double a = 0, double b = 1)  //
    {
        return dist(shape, std::cauchy_distribution(a, b));
    }
    static auto fisherf(const Shape &shape, double m = 1, double n = 1)  //
    {
        return dist(shape, std::fisher_f_distribution(m, n));
    }
    static auto studentt(const Shape &shape, double n = 1)  //
    {
        return dist(shape, std::student_t_distribution(n));
    }

    ////////////////////////////////////////////////////////////////////////////

    UNARY0(cos, std::cos(val1))
    UNARY0(sin, std::sin(val1))
    UNARY0(tan, std::tan(val1))
    UNARY0(acos, std::acos(val1))
    UNARY0(asin, std::asin(val1))
    UNARY0(atan, std::atan(val1))
    UNARY0(cosh, std::cosh(val1))
    UNARY0(sinh, std::sinh(val1))
    UNARY0(tanh, std::tanh(val1))
    UNARY0(acosh, std::acosh(val1))
    UNARY0(asinh, std::asinh(val1))
    UNARY0(exp, std::exp(val1))
    UNARY0(log, std::log(val1))
    UNARY0(log2, std::log2(val1))
    UNARY0(log10, std::log10(val1))
    UNARY0(exp2, std::exp2(val1))
    UNARY0(expm1, std::expm1(val1))
    UNARY0(ilogb, std::ilogb(val1))
    UNARY0(log1p, std::log1p(val1))
    UNARY0(sqrt, std::sqrt(val1))
    UNARY0(cbrt, std::cbrt(val1))
    UNARY0(erf, std::erf(val1))
    UNARY0(erfc, std::erfc(val1))
    UNARY0(tgamma, std::tgamma(val1))
    UNARY0(lgamma, std::lgamma(val1))
    UNARY0(ceil, std::ceil(val1))
    UNARY0(floor, std::floor(val1))
    UNARY0(trunc, std::trunc(val1))
    UNARY0(round, std::round(val1))
    UNARY0(lround, std::lround(val1))
    UNARY0(llround, std::llround(val1))
    UNARY0(rint, std::rint(val1))
    UNARY0(lrint, std::lrint(val1))
    UNARY0(llrint, std::llrint(val1))
    UNARY0(nearbyint, std::nearbyint(val1))
    UNARY0(conj, std::conj(val1))
    UNARY0(proj, std::proj(val1))
    UNARY0(neg, -val1)
    UNARY0(pos, +val1)
    UNARY0(_not, !val1)
    UNARY0(square, val1 *val1)
    UNARY0(cube, val1 *val1 *val1)
    UNARY0(frac, val1 - std::floor(val1))
    UNARY0(ln, std::log(val1))
    UNARY0(rev, 1 / val1)
    UNARY0(rsqrt, 1 / std::sqrt(val1))
    UNARY0(relu, std::fmax(0, val1))
    UNARY0(sigmoid, 1 / (1 + std::exp(-val1)))
    UNARY0(deg2rad, val1 *M_PI / 180)
    UNARY0(rad2deg, val1 * 180 / M_PI)

    ////////////////////////////////////////////////////////////////////////////

    UNARY1(atan2, Type, std::atan2(val1, val2))
    UNARY1(fdim, Type, std::fdim(val1, val2))
    UNARY1(ldexp, int, std::ldexp(val1, val2))
    UNARY1(scalbn, int, std::scalbn(val1, val2))
    UNARY1(scalbln, long, std::scalbln(val1, val2))
    UNARY1(pow, Type, std::pow(val1, val2))
    UNARY1(hypot, Type, std::hypot(val1, val2))
    UNARY1(remainder, Type, std::remainder(val1, val2))
    UNARY1(copysign, Type, std::copysign(val1, val2))
    UNARY1(nextafter, Type, std::nextafter(val1, val2))
    UNARY1(nexttoward, Type, std::nexttoward(val1, val2))
    UNARY1(fmin, Type, std::fmax(val1, val2))
    UNARY1(fmax, Type, std::fmin(val1, val2))
    UNARY1(add, Type, val1 + val2)
    UNARY1(sub, Type, val1 - val2)
    UNARY1(mul, Type, val1 *val2)
    UNARY1(div, Type, val1 / val2)
    UNARY1(mod, Type, val1 % val2)
    UNARY1(_and, Type, val1 &val2)
    UNARY1(_or, Type, val1 | val2)
    UNARY1(_xor, Type, val1 ^ val2)
    UNARY1(lshift, Size, val1 << val2)
    UNARY1(rshift, Size, val1 >> val2)
    UNARY1(revsub, Type, val2 - val1)
    UNARY1(revdiv, Type, val2 / val1)
    UNARY1(revmod, Type, val2 % val1)
    UNARY1(revlshift, Type, val2 << val1)
    UNARY1(revlrshift, Type, val2 >> val1)
    UNARY1(heaviside, Type, val1 < 0 ? 0 : (val1 > 0 ? 1 : val2))

    ////////////////////////////////////////////////////////////////////////////

    _UNARY0(arg, typename Type::value_type, std::arg(val1))
    _UNARY0(norm, typename Type::value_type, std::norm(val1))
    _UNARY0(real, typename Type::value_type, val1.real())
    _UNARY0(imag, typename Type::value_type, val1.imag())
    _UNARY0(isnan, bool, std::isnan(val1))
    _UNARY0(isinf, bool, std::isinf(val1))
    _UNARY0(sign, bool, std::signbit(val1))
    _UNARY0(zero, bool, val1 == 0)
    _UNARY0(nonzero, bool, val1 != 0)
    _UNARY1(gt, bool, Type, val1 > val2)
    _UNARY1(ge, bool, Type, val1 >= val2)
    _UNARY1(lt, bool, Type, val1 < val2)
    _UNARY1(le, bool, Type, val1 <= val2)
    _UNARY1(eq, bool, Type, val1 == val2)
    _UNARY1(ne, bool, Type, val1 != val2)
    _BINARY(gt, bool, val1 > val2)
    _BINARY(ge, bool, val1 >= val2)
    _BINARY(lt, bool, val1 < val2)
    _BINARY(le, bool, val1 <= val2)
    _BINARY(eq, bool, val1 == val2)
    _BINARY(ne, bool, val1 != val2)
    _BINARY(complex, std::complex<Type>, std::complex(val1, val2))
    _BINARY(lshift, Size, val1 << val2)
    _BINARY(rshift, Size, val1 >> val2)

    ////////////////////////////////////////////////////////////////////////////

    BINARY(add, val1 + val2)
    BINARY(sub, val1 - val2)
    BINARY(mul, val1 *val2)
    BINARY(div, val1 / val2)
    BINARY(mod, val1 % val2)
    BINARY(_and, val1 &val2)
    BINARY(_or, val1 | val2)
    BINARY(_xor, val1 ^ val2)
    BINARY(atan2, std::atan2(val1, val2))
    BINARY(pow, std::pow(val1, val2))
    BINARY(remainder, std::remainder(val1, val2))
    BINARY(fmin, std::fmin(val1, val2))
    BINARY(fmax, std::fmax(val1, val2))
    BINARY(mask, val2 ? val1 : 0)
    BINARY(heaviside, val1 < 0 ? 0 : (val1 > 0 ? 1 : val2))

    ////////////////////////////////////////////////////////////////////////////

    REDUCE0(sum, 0, val1 + val2)
    REDUCE0(prod, 1, val1 *val2)
    REDUCE0(max, std::numeric_limits<Type>::min(), std::max(val1, val2))
    REDUCE0(min, std::numeric_limits<Type>::max(), std::min(val1, val2))
    _REDUCE1(count, Size, Type, 0, val1 + (val2 == val3))

    ////////////////////////////////////////////////////////////////////////////

    OPERATOR0(neg, -)
    OPERATOR0(pos, +)
    OPERATOR0(_not, !)
    OPERATOR0(_not, ~)
    OPERATOR1(lt, gt, <)
    OPERATOR1(le, ge, <=)
    OPERATOR1(gt, lt, >)
    OPERATOR1(ge, le, >=)
    OPERATOR1(eq, eq, ==)
    OPERATOR1(ne, ne, !=)
    OPERATOR1(add, add, +)
    OPERATOR1(mul, mul, *)
    OPERATOR1(sub, revsub, -)
    OPERATOR1(div, revdiv, /)
    OPERATOR1(mod, revmod, %)
    OPERATOR1(_and, _and, &)
    OPERATOR1(_or, _or, |)
    OPERATOR1(_xor, _xor, ^)
    OPERATOR1(lshift, revlshift, <<)
    OPERATOR1(rshift, revrshift, >>)
};
}  // namespace Tense::TensorImpl

#undef _UNARY0
#undef _UNARY1
#undef _BINARY
#undef _REDUCE0
#undef _REDUCE1
#undef UNARY0
#undef UNARY1
#undef BINARY
#undef REDUCE0
#undef _REDUCE1
#undef _OPERATOR0
#undef _OPERATOR1
#undef OPERATOR1
