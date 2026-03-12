/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2026, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
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

#pragma once

#include <tense/vector/expr.h>
#include <tense/vector/extr.h>

#include <algorithm>
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
    auto NAME(const Expr2& expr2) const                                               \
    {                                                                                 \
        return binary<TYPE, Expr2>(expr2, [](auto val1, auto val2) { return FUNC; }); \
    }

#define _REDUCE0(NAME, TYPE, INIT, FUNC)                                      \
    auto NAME() const                                                         \
    {                                                                         \
        return reduce<TYPE>(INIT, [](auto val1, auto val2) { return FUNC; }); \
    }

#define _REDUCE1(NAME, TYPE, VTYPE, INIT, FUNC)                                   \
    auto NAME(VTYPE val3) const                                                   \
    {                                                                             \
        return reduce<TYPE>(INIT, [val3](auto val1, auto val2) { return FUNC; }); \
    }

#define OPERATOR0(NAME, OP) \
    auto operator OP() const { return derived().NAME(); }

#define OPERATOR1(NAME, RNAME, OP)                                       \
    template <typename Expr2, typename = IsExpr<Expr2>>                  \
    auto operator OP(const Expr2& expr2) const                           \
    {                                                                    \
        return derived().NAME(expr2);                                    \
    }                                                                    \
    auto operator OP(Type expr2) const { return derived().NAME(expr2); } \
    friend auto operator OP(Type expr2, const Derived& expr1) { return expr1.RNAME(expr2); }

#define UNARY0(NAME, FUNC) _UNARY0(NAME, typename Derived::Type, FUNC)
#define UNARY1(NAME, TYPE, FUNC) _UNARY1(NAME, typename Derived::Type, TYPE, FUNC)
#define BINARY(NAME, FUNC) _BINARY(NAME, typename Derived::Type, FUNC)
#define REDUCE0(NAME, INIT, FUNC) _REDUCE0(NAME, typename Derived::Type, INIT, FUNC)
#define REDUCE1(NAME, TYPE, INIT, FUNC) _REDUCE1(NAME, TYPE, typename Derived::Type, INIT, FUNC)

namespace Tense::VectorImpl
{
template <typename Type, typename Derived>
class Base : Expr
{
    const Derived& derived() const { return *static_cast<const Derived*>(this); }
    Size size() const { return derived()->size(); }

    auto _repeat(Size size) const
    {
        TENSE_TASSERT(size, >, 0, "repeat", "Input size can't be zero")
        return Repeat<Derived>(derived(), size * this->size());
    }

public:
    template <typename T, typename Func>
    auto unary(Func func) const
    {
        return Unary<T, Derived, Func>(derived(), func);
    }
    template <typename T, typename Expr2, typename Func, typename = IsExpr<Expr2>>
    auto binary(const Expr2& expr2, Func func) const
    {
        TENSE_VASSERT(this->size(), ==, expr2.size(), "binary", "Size of vectors must be equal");

        if constexpr (IsHeavy<Derived>::value && IsHeavy<Expr2>::value)
            return eval().template binary<T>(expr2.eval(), func);
        else if constexpr (IsHeavy<Derived>::value)
            return eval().template binary<T>(expr2, func);
        else if constexpr (IsHeavy<Expr2>::value)
            return binary<T>(expr2.eval(), func);
        else
            return Binary<T, Derived, Expr2, Func>(derived(), expr2, func);
    }

    auto repeat(Size size) const
    {
        if constexpr (IsHeavy<Derived>::value)
            return derived().eval()._repeat(size);
        else
            return derived()._repeat(size);
    }

    template <typename T, typename Func>
    auto reduce(T init, Func func) const
    {
        return Reduce<T, Derived, Func>(derived(), func, init).item();
    }

    auto block(Size i, Size size) const
    {
        TENSE_TASSERT(size, >, 0, "block", "Input size can't be zero")
        TENSE_TASSERT(i + size, <=, this->size(), "block", "Block bounds can't be out of range")
        return SBlock<Derived>(derived(), i, size);
    }
    auto elem(Size size) const
    {
        TENSE_TASSERT(size, <, this->size(), "elem", "Input row can't be out of range")
        return SElem<Derived>(derived(), size);
    }

    auto index(Cut cut) const
    {
        if (cut.step == 0) cut = {this->size()};
        TENSE_TASSERT(cut.start, <=, this->size(), "index", "Input row start can't be out of range")
        TENSE_TASSERT(cut.end, <=, this->size(), "index", "Input row end can't be out of range")
        return RIndex<Derived>(derived(), cut);
    }
    auto index(const std::vector<Size>& indices) const { return VIndex<Derived>(derived(), indices); }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto rindirect(const Expr2& expr2)
    {
        TENSE_TASSERT(size(), ==, expr2.size(), "indirect", "Size of vectors must be equal")
        return Indirect<Derived, Expr2>(derived(), expr2);
    }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto cat0(const Expr2& expr2) const
    {
        return Cat0<Derived, Expr2>(derived(), expr2, this->size() + expr2.size());
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto cat1(const Expr2& expr2) const
    {
        TENSE_TASSERT(this->size(), ==, expr2.size(), "ircat", "Size of vectors must be equal")
        return Cat1<Derived, Expr2>(derived(), expr2);
    }

    auto turn(Size size) const
    {
        TENSE_TASSERT(size, <=, this->size(), "turn", "Input size can't be higher than vector size")
        return Turn<Derived>(derived(), this->size() - size);
    }

    auto minidx() const { return MinIdx<Derived>(derived()); }
    auto maxidx() const { return MaxIdx<Derived>(derived()); }

    auto ball(Size size) const
    {
        TENSE_TASSERT(size, >, 0, "ball", "Input size can't be zero")
        TENSE_TASSERT(this->size() % size, ==, 0, "ball", "Vector size must be divisable by input size")
        return BAny<std::false_type, Derived>(derived(), size);
    }
    auto all() const { return Any<std::false_type, Derived>(derived()).item(); }

    auto bany(Size size) const
    {
        TENSE_TASSERT(size, >, 0, "bany", "Input size can't be zero")
        TENSE_TASSERT(this->size() % size, ==, 0, "bany", "Vector size must be divisable by input size")
        return BAny<std::true_type, Derived>(derived(), size);
    }
    auto any() const { return Any<std::true_type, Derived>(derived()).item(); }

    auto bflip(Size size) const
    {
        TENSE_TASSERT(size, >, 0, "bflip", "Input size can't be zero")
        TENSE_TASSERT(this->size() % size, ==, 0, "bflip", "Vector size must be divisable by input size")
        return BFlip<Derived>(derived(), size);
    }
    auto flip() const { return Flip<Derived>(derived()); }

    template <int P>
    auto pow() const
    {
        return Power<P, Derived>(derived());
    }
    template <typename T>
    auto type() const
    {
        if constexpr (IsComplex<Type>::value && !IsComplex<T>::value)
            return derived().real().template type<T>();
        else
            return Convert<T, Derived>(derived());
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
    auto where(Func func, const Expr2& expr2) const
    {
        TENSE_TASSERT(this->size(), ==, expr2.size(), "where", "Size of vectors must be equal")
        return FEWhere<Derived, Expr2, Func>(derived(), expr2, func);
    }
    template <typename Func, typename Expr2, typename = IsExpr<Expr2>>
    auto iwhere(Func func, const Expr2& expr2) const
    {
        TENSE_TASSERT(this->size(), ==, expr2.size(), "iwhere", "Size of vectors must be equal")
        return IEWhere<Derived, Expr2, Func>(derived(), expr2, func);
    }

    auto polar() const { return Polar<Derived>(derived()); }

    auto expr() const { return Self<Derived>(derived()); }
    auto eval() const { return Vector<Type>(derived()); }
    auto copy() const { return eval(); }
    Type item() const { return derived()[0]; }
    Shape shape() const { return {size()}; }
    auto tensor() const { return TensorImpl::ToTensor<Derived>(derived(), {derived().size()}); }

    ////////////////////////////////////////////////////////////////////////////

    auto left(Size size) { return block(0, size); }
    auto right(Size size) { return block(this->size() - size, size); }

    template <Size P>
    auto bnorm(Size size) const
    {
        static_assert(P > 0, "P can't be zero in vector::bnorm");

        if constexpr (P == Inf)
            return derived().abs().bmax(size);
        else if constexpr (P == 1)
            return derived().abs().bsum(size);
        else if constexpr (P == 2)
            return derived().abs().square().bsum(size).sqrt();
        else
            return derived().abs().template pow<P>().bsum(size).pow(1.f / P);
    }
    template <Size P>
    auto norm() const
    {
        static_assert(P > 0, "P can't be zero in vector::norm");

        if constexpr (P == Inf)
            return derived().abs().max();
        else if constexpr (P == 1)
            return derived().abs().sum();
        else if constexpr (P == 2)
            return std::sqrt(derived().abs().square().sum());
        else
            return std::pow(derived().abs().template pow<P>().sum(), 1.f / P);
    }

    template <Size P, typename Expr2, typename = IsExpr<Expr2>>
    auto bdistance(Size size, const Expr2& expr2) const
    {
        return derived().sub(expr2).template bnorm<P>(size);
    }
    template <Size P, typename Expr2, typename = IsExpr<Expr2>>
    auto distance(const Expr2& expr2) const
    {
        return derived().sub(expr2).template norm<P>();
    }

    template <Size P>
    auto bdistance(Size size, Type expr2) const
    {
        return derived().sub(expr2).template bnorm<P>(size);
    }
    template <Size P>
    auto distance(Type expr2) const
    {
        return derived().sub(expr2).template norm<P>();
    }

    auto bvar(Size size, Size dof = 0) const
    {
        auto expr1 = derived().square().bsum(size);
        auto expr2 = derived().bsum(size).square() / size;
        return (expr1 - expr2) / (size - dof);
    }
    auto var(Size dof = 0) const
    {
        Size size = this->size();
        auto expr1 = derived().square().sum();
        auto expr2 = std::pow(derived().sum(), 2) / size;
        return (expr1 - expr2) / (size - dof);
    }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto bcov(const Expr2& expr2, Size size, Size dof = 0) const
    {
        TENSE_TASSERT(this->size(), ==, expr2.size(), "bcov", "Size of vectors must be the same")

        auto _expr1 = derived().mul(expr2).bsum(size);
        auto _expr2 = derived().bsum(size).mul(expr2.bsum()) / size;
        return (_expr1 - _expr2) / (size - dof);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto cov(const Expr2& expr2, Size dof = 0) const
    {
        TENSE_TASSERT(this->size(), ==, expr2.size(), "cov", "Size of vectors must be the same")

        Size size = this->size();
        auto _expr1 = derived().mul(expr2).sum();
        auto _expr2 = derived().sum() * expr2.sum() / size;
        return (_expr1 - _expr2) / (size - dof);
    }
    auto cov(Size dof = 0) const
    {
        auto mat = (derived() - derived().mean()).eval();
        return mat.adjoint() * mat / (mat.size() - dof);
    }
    // TODO cov with self?

    auto bmean(Size size) const { return derived().bsum(size) / size; }
    auto mean() const { return derived().sum() / (this->size()); }

    auto bstd(Size size, Size dof = 0) const { return derived().bvar(size, dof).sqrt(); }
    auto std(Size dof = 0) const { return std::sqrt(derived().var(dof)); }

    auto bcontains(Size size, Type expr2) const { return derived().eq(expr2).bany(size); }
    auto contains(Type expr2) const { return derived().eq(expr2).any(); }

    auto bequal(Size size, Type expr2) const { return derived().eq(expr2).ball(size); }
    auto equal(Type expr2) const { return derived().eq(expr2).all(); }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto bequal(Size size, const Expr2& expr2) const
    {
        return derived().eq(expr2).ball(size);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto equal(const Expr2& expr2) const
    {
        return derived().eq(expr2).all();
    }

    auto bnormalize(Size size) const
    {
        auto min = derived().bmin(size).eval();
        auto dist = derived().bmax(size).sub(min).eval();
        return (derived() - min) / dist;
    }
    auto normalize() const
    {
        auto min = derived().min();
        auto dist = (derived().max() - min);
        return (derived() - min) / dist;
    }

    auto bstandize(Size size, Size dof = 0) const
    {
        auto mean = derived().bmean(size).eval();
        auto std = derived().bstd(size, dof).eval();
        return (derived() - mean) / std;
    }
    auto standize(Size dof = 0) const
    {
        auto mean = derived().mean();
        auto std = derived().std(dof);
        return (derived() - mean) / std;
    }

    auto close(Type expr2, typename FromComplex<Type>::Type expr3 = 1e-5) const
    {
        return derived().sub(expr2).abs().le(expr3).all();
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto close(const Expr2& expr2, typename FromComplex<Type>::Type expr3 = 1e-5) const
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
        return unary<Type>([val2, val3](auto val1) { return std::clamp(val1, val2, val3); });
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto dot(const Expr2& expr2) const
    {
        const auto& expr1 = derived();
        TENSE_TASSERT(this->size(), ==, expr2.size(), "dot", "Vector sizes must be the same")
        return expr1.mul(expr2).sum();
    }

    ////////////////////////////////////////////////////////////////////////////

    static auto init(Size size, Type val)
    {
        TENSE_TASSERT(size, >, 0, "init", "Input size can't be zero")
        return Constant<Type>(size, val);
    }
    template <typename Dist>
    static auto dist(Size size, Dist dist)
    {
        TENSE_TASSERT(size, >, 0, "dist", "Input size can't be zero")
        return Distribution<Type, Dist>(size, dist);
    }
    static auto seq(Size size, Type start, Type end)
    {
        TENSE_TASSERT(size, >, 0, "seq", "Input size can't be zero")
        auto step = (end - start) / size;
        return Sequence<Type>(size, start, step);
    }
    static auto strided(Size size, Type* data, Size rstride, Size cstride)
    {
        TENSE_TASSERT(size, >, 0, "seq", "Input size can't be zero")
        return Strided<Type>(size, data, rstride, cstride);
    }
    template <Size Count>
    static auto stat(Type value = Type(0))
    {
        static_assert(Count > 0, "Size can't be zero in vector::static");
        return Static<Type, Count>(value);
    }
    template <Size Count>
    static auto stat(const std::vector<Type>& list)
    {
        static_assert(Count > 0, "Size can't be zero in vector::static");
        return Static<Type, Count>(list);
    }
    template <Size Count>
    static auto stat(const std::initializer_list<Type>& list)
    {
        TENSE_TASSERT(list.size(), ==, Count, "stat", "Vector size must match initializer list")
        auto target = stat<Count>();
        Eval::assign(target, list);
        return target;
    }

    ////////////////////////////////////////////////////////////////////////////

    static auto numin(Size size) { return init(size, std::numeric_limits<Type>::min()); }
    static auto numax(Size size) { return init(size, std::numeric_limits<Type>::max()); }
    static auto lowest(Size size) { return init(size, std::numeric_limits<Type>::lowest()); }
    static auto epsilon(Size size) { return init(size, std::numeric_limits<Type>::epsilon()); }
    static auto inf(Size size) { return init(size, std::numeric_limits<Type>::infinity()); }
    static auto nan(Size size) { return init(size, std::numeric_limits<Type>::quiet_NaN()); }
    static auto zeros(Size size) { return init(size, 0); }
    static auto ones(Size size) { return init(size, 1); }
    static auto e(Size size) { return init(size, M_E); }
    static auto pi(Size size) { return init(size, M_PI); }
    static auto sqrt2(Size size) { return init(size, M_SQRT2); }

    ////////////////////////////////////////////////////////////////////////////

    static auto uniform(Size size, Type a = 0, Type b = 1)
    {
        if constexpr (std::is_integral<Type>::value)
            return dist(size, std::uniform_int_distribution<Type>(a, b));
        else if constexpr (std::is_floating_point<Type>::value)
            return dist(size, std::uniform_real_distribution<Type>(a, b));
        else
            throw std::runtime_error("Data type not supported in vector::uniform.");
    }
    static auto bernoulli(Size size, double p = 0.5) { return dist(size, std::bernoulli_distribution(p)); }
    static auto binomial(Size size, int t, double p = 0.5) { return dist(size, std::binomial_distribution(t, p)); }
    static auto geometric(Size size, double p = 0.5) { return dist(size, std::geometric_distribution(p)); }
    static auto _pascal(Size size, int k, double p = 0.5)
    {
        return dist(size, std::negative_binomial_distribution(k, p));
    }
    static auto poisson(Size size, double mean = 1) { return dist(size, std::poisson_distribution(mean)); }
    static auto exponential(Size size, double lambda = 1) { return dist(size, std::exponential_distribution(lambda)); }
    static auto gamma(Size size, double alpha = 1, double beta = 1)
    {
        return dist(size, std::gamma_distribution(alpha, beta));
    }
    static auto weibull(Size size, double a = 1, double b = 1) { return dist(size, std::weibull_distribution(a, b)); }
    static auto extremevalue(Size size, double a = 0, double b = 1)
    {
        return dist(size, std::extreme_value_distribution(a, b));
    }
    static auto normal(Size size, double mean = 0, double std = 1)
    {
        return dist(size, std::normal_distribution(mean, std));
    }
    static auto lognormal(Size size, double m = 0, double s = 1)
    {
        return dist(size, std::lognormal_distribution(m, s));
    }
    static auto chisquared(Size size, double n = 1) { return dist(size, std::chi_squared_distribution(n)); }
    static auto cauchy(Size size, double a = 0, double b = 1) { return dist(size, std::cauchy_distribution(a, b)); }
    static auto fisherf(Size size, double m = 1, double n = 1) { return dist(size, std::fisher_f_distribution(m, n)); }
    static auto studentt(Size size, double n = 1) { return dist(size, std::student_t_distribution(n)); }

    ////////////////////////////////////////////////////////////////////////////

    template <typename Func>
    auto sort(Func func) const
    {
        auto target = derived().copy();
        Backend::sort(target, func);
        return target;
    }
    auto sort() const
    {
        return sort([](auto i, auto j) { return i < j; });
    }
    template <typename Func>
    auto sortidx(Func func) const
    {
        return Backend::sortidx(derived(), func);
    }
    auto sortidx() const
    {
        return sortidx([](auto i, auto j) { return i < j; });
    }

    auto shuffle() const
    {
        auto target = derived().copy();
        Backend::shuffle(target);
        return target;
    }
    auto shuffleidx() const { return Backend::shuffle(derived()); }

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
    UNARY0(proj, std::proj(val1))
    UNARY0(neg, -val1)
    UNARY0(pos, +val1)
    UNARY0(_not, !val1)
    UNARY0(square, val1* val1)
    UNARY0(cube, val1 * val1 * val1)
    UNARY0(frac, val1 - std::floor(val1))
    UNARY0(ln, std::log(val1))
    UNARY0(rev, 1 / val1)
    UNARY0(rsqrt, 1 / std::sqrt(val1))
    UNARY0(relu, std::fmax(0, val1))
    UNARY0(sigmoid, 1 / (1 + std::exp(-val1)))
    UNARY0(deg2rad, val1* M_PI / 180)
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
    UNARY1(mul, Type, val1* val2)
    UNARY1(div, Type, val1 / val2)
    UNARY1(mod, Type, val1 % val2)
    UNARY1(_and, Type, val1& val2)
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
    BINARY(mul, val1* val2)
    BINARY(div, val1 / val2)
    BINARY(mod, val1 % val2)
    BINARY(_and, val1& val2)
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
    REDUCE0(prod, 1, val1* val2)
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
    OPERATOR1(sub, revsub, -)
    OPERATOR1(mul, mul, *)
    OPERATOR1(div, revdiv, /)
    OPERATOR1(mod, revmod, %)
    OPERATOR1(_and, _and, &)
    OPERATOR1(_or, _or, |)
    OPERATOR1(_xor, _xor, ^)
    OPERATOR1(lshift, revlshift, <<)
    OPERATOR1(rshift, revrshift, >>)
};
}  // namespace Tense::VectorImpl

#undef _UNARY0
#undef _UNARY1
#undef _BINARY
#undef _REDUCE0
#undef _REDUCE1
#undef _SQUARE
#undef SQUARE
#undef UNARY0
#undef UNARY1
#undef BINARY
#undef REDUCE0
#undef REDUCE1
#undef _OPERATOR0
#undef _OPERATOR1
#undef OPERATOR1
