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

#pragma once

#include <blasw/blasw.h>
#include <tense/matrix/struct.h>

#include <algorithm>
#include <numeric>

namespace Tense::MatrixImpl
{
namespace Backend
{
template <typename Expr1, typename Func>
void rsort(Expr1& expr1, Func func)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();

    if constexpr (std::is_same<Major, Col>::value)
        throw std::runtime_error("Not Implemented.");
    else
#pragma omp parallel for
        for (Size i = 0; i < rows; ++i)
        {
            auto data = &expr1(i, 0);
            std::sort(data, data + cols, func);
        }
}

template <typename Expr1, typename Func>
void csort(Expr1& expr1, Func func)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();

    if constexpr (std::is_same<Major, Col>::value)
#pragma omp parallel for
        for (Size j = 0; j < cols; ++j)
        {
            auto data = &expr1(0, j);
            std::sort(data, data + rows, func);
        }
    else
        throw std::runtime_error("Not Implemented.");
}

template <typename Expr1, typename Func>
auto rsortidx(Expr1& expr1, Func func)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();
    Matrix<Major, Size> index(rows, cols);

    if constexpr (std::is_same<Major, Col>::value)
        throw std::runtime_error("Not Implemented.");
    else
#pragma omp parallel for
        for (Size i = 0; i < rows; ++i)
        {
            auto comp = [&](auto j, auto k) { return func(expr1(i, j), expr1(i, k)); };
            auto data = &index(i, 0);
            std::iota(data, data + cols, 0);
            std::sort(data, data + cols, comp);
        }

    return index;
}

template <typename Expr1, typename Func>
auto csortidx(Expr1& expr1, Func func)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();
    Matrix<Major, Size> index(rows, cols);

    if constexpr (std::is_same<Major, Col>::value)
#pragma omp parallel for
        for (Size j = 0; j < cols; ++j)
        {
            auto comp = [&](auto i, auto k) { return func(expr1(i, j), expr1(k, j)); };
            auto data = &index(0, j);
            std::iota(data, data + rows, 0);
            std::sort(data, data + rows, comp);
        }
    else
        throw std::runtime_error("Not Implemented.");

    return index;
}

template <typename Expr1>
void rshuffle(Expr1& expr1)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();

    if constexpr (std::is_same<Major, Col>::value)
        throw std::runtime_error("Not Implemented.");
    else
#pragma omp parallel for
        for (Size i = 0; i < rows; ++i)
        {
            auto data = &expr1(i, 0);
            std::random_shuffle(data, data + cols);
        }
}

template <typename Expr1>
void cshuffle(Expr1& expr1)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();

    if constexpr (std::is_same<Major, Col>::value)
#pragma omp parallel for
        for (Size j = 0; j < cols; ++j)
        {
            auto data = &expr1(0, j);
            std::random_shuffle(data, data + rows);
        }
    else
        throw std::runtime_error("Not Implemented.");
}

template <typename Expr1>
void shuffle(Expr1& expr1)
{
    auto size = expr1.size();
    auto data = &expr1(0, 0);
    std::random_shuffle(data, data + size);
}

template <typename Expr1>
auto rshuffleidx(Expr1& expr1)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();
    Matrix<Major, Size> index(rows, cols);

    if constexpr (std::is_same<Major, Col>::value)
        throw std::runtime_error("Not Implemented.");
    else
#pragma omp parallel for
        for (Size i = 0; i < rows; ++i)
        {
            auto data = &index(i, 0);
            std::iota(data, data + cols, 0);
            std::random_shuffle(data, data + cols);
        }

    return index;
}

template <typename Expr1>
auto cshuffleidx(Expr1& expr1)
{
    using Major = typename Expr1::Major;
    Size rows = expr1.rows(), cols = expr1.cols();
    Matrix<Major, Size> index(rows, cols);

    if constexpr (std::is_same<Major, Col>::value)
#pragma omp parallel for
        for (Size j = 0; j < cols; ++j)
        {
            auto data = &index(0, j);
            std::iota(data, data + rows, 0);
            std::random_shuffle(data, data + rows);
        }
    else
        throw std::runtime_error("Not Implemented.");

    return index;
}

template <typename Expr1, typename Expr2, typename Expr3>
void multiply(const Expr1& first, const Expr2& second, Expr3& target)
{
    const Size X = first.rows(), Y = first.cols(), Z = second.cols();

    if constexpr (std::is_same<typename Expr1::Major, Col>::value)
        for (Size j = 0; j < Z; ++j)
            for (Size k = 0; k < Y; ++k)
                for (Size i = 0; i < X; ++i) target(i, j) += first(i, k) * second(k, j);
    else
        for (Size i = 0; i < X; ++i)
            for (Size k = 0; k < Y; ++k)
                for (Size j = 0; j < Z; ++j) target(i, j) += first(i, k) * second(k, j);
}
}  // namespace Backend

namespace External
{
template <typename T>
struct CheckExpr
{
    typedef char yes[1];
    typedef char no[2];
    template <typename C>
    static yes& test(typename C::Expr*);
    template <typename>
    static no& test(...);
    static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);
};

struct Dummy
{
    using Flag = void;
};
template <typename T, typename E = void>
struct GetExpr
{
    using Type = Dummy;
};
template <typename T>
struct GetExpr<T, typename std::enable_if<CheckExpr<T>::value>::type>
{
    using Type = typename T::Expr;
};

template <typename _T>
constexpr bool istr()
{
    using T = typename _T::Flag;
    return std::is_same<UpperFlag, T>::value || std::is_same<LowerFlag, T>::value ||
           std::is_same<OUpperFlag, T>::value || std::is_same<OLowerFlag, T>::value;
}
template <typename _T>
constexpr bool issy()
{
    using T = typename _T::Flag;
    return std::is_same<USymmFlag, T>::value || std::is_same<LSymmFlag, T>::value;
}
template <typename _T>
constexpr bool ishe()
{
    using T = typename _T::Flag;
    return std::is_same<UHermFlag, T>::value || std::is_same<LHermFlag, T>::value;
}
template <typename _T>
constexpr bool isup()
{
    using T = typename _T::Flag;
    return std::is_same<USymmFlag, T>::value || std::is_same<UHermFlag, T>::value ||
           std::is_same<UpperFlag, T>::value || std::is_same<OUpperFlag, T>::value;
}
template <typename _T>
constexpr bool islw()
{
    using T = typename _T::Flag;
    return std::is_same<LSymmFlag, T>::value || std::is_same<LHermFlag, T>::value ||
           std::is_same<LowerFlag, T>::value || std::is_same<OLowerFlag, T>::value;
}
template <typename _T>
constexpr bool isun()
{
    using T = typename _T::Flag;
    return std::is_same<OUpperFlag, T>::value || std::is_same<OLowerFlag, T>::value;
}
template <typename _T>
constexpr bool ists()
{
    using T = typename _T::Flag;
    return std::is_same<TransFlag, T>::value;
}
template <typename _T>
constexpr bool iscj()
{
    using T = typename _T::Flag;
    return std::is_same<ConjFlag, T>::value;
}

#define DERIVED Matrix<M, T>

template <typename M, typename T>
auto wrge(DERIVED expr)
{
    auto major = std::is_same<M, Row>::value ? Blasw::Major::Row : Blasw::Major::Col;
    if (!expr.valid()) return Blasw::General<T>(nullptr, 0, 0, 0, major, Blasw::State::None);
    return Blasw::General<T>(expr.data(), expr.rows(), expr.cols(), 0, major, Blasw::State::None);
}
template <typename Expr1, typename M, typename T>
auto wrtr(DERIVED expr, bool flip = false)
{
    auto diag = isun<Expr1>() ? Blasw::Diagonal::Unit : Blasw::Diagonal::NonUnit;
    auto major = std::is_same<M, Row>::value ? Blasw::Major::Row : Blasw::Major::Col;
    auto tri = isup<Expr1>() ^ flip ? Blasw::Triangular::Upper : Blasw::Triangular::Lower;
    if (!expr.valid()) return Blasw::Triangle<T>(nullptr, 0, 0, major, Blasw::State::None, tri, diag);
    return Blasw::Triangle<T>(expr.data(), expr.rows(), 0, major, Blasw::State::None, tri, diag);
}
template <typename Expr1, typename M, typename T>
auto wrsy(DERIVED expr, bool flip = false)
{
    auto major = std::is_same<M, Row>::value ? Blasw::Major::Row : Blasw::Major::Col;
    auto tri = isup<Expr1>() ^ flip ? Blasw::Triangular::Upper : Blasw::Triangular::Lower;
    if (!expr.valid()) return Blasw::Symmetric<T>(nullptr, 0, 0, major, tri);
    return Blasw::Symmetric<T>(expr.data(), expr.rows(), 0, major, tri);
}
template <typename Expr1, typename M, typename T>
auto wrhe(DERIVED expr, bool flip = false)
{
    auto major = std::is_same<M, Row>::value ? Blasw::Major::Row : Blasw::Major::Col;
    auto tri = isup<Expr1>() ^ flip ? Blasw::Triangular::Upper : Blasw::Triangular::Lower;
    if (!expr.valid()) return Blasw::Hermitian<T>(expr.data(), expr.rows(), 0, major, tri);
    return Blasw::Hermitian<T>(expr.data(), expr.rows(), 0, major, tri);
}

template <typename Expr1, typename Expr2, typename Expr3>
void multiply(const Expr1& expr1, const Expr2& expr2, Expr3& expr)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using A1 = typename GetExpr<A2>::Type;

    using B3 = Expr2;
    using B2 = typename GetExpr<B3>::Type;
    using B1 = typename GetExpr<B2>::Type;

    using T = typename Expr3::Type;
    using M = typename Expr3::Major;

    if constexpr ((istr<A1>() && ists<A2>() && iscj<A3>()) || (istr<A1>() && ists<A3>() && iscj<A2>()))
    {
        expr = expr2.expr();
        auto op1 = expr1.get().get().get().eval(), op2 = expr;
        Blasw::dot(wrtr<A1>(op1).adjoint(), wrge(op2), 1);
    }
    else if constexpr ((istr<B1>() && ists<B2>() && iscj<B3>()) || (istr<B1>() && ists<B3>() && iscj<B2>()))
    {
        expr = expr1.expr();
        auto op1 = expr, op2 = expr2.get().get().get().eval();
        Blasw::dot(wrge(op1), wrtr<B1>(op2).adjoint(), 1);
    }
    ////////////////////////////////////////////////////////////////////////////
    else if constexpr (istr<A2>() && ists<A3>())
    {
        expr = expr2.expr();
        auto op1 = expr1.get().get().eval(), op2 = expr;
        Blasw::dot(wrtr<A2>(op1).trans(), wrge(op2), 1);
    }
    else if constexpr (istr<B2>() && ists<B3>())
    {
        expr = expr1.expr();
        auto op1 = expr, op2 = expr2.get().get().eval();
        Blasw::dot(wrge(op1), wrtr<B2>(op2).trans(), 1);
    }
    else if constexpr (issy<A2>() && ists<A3>())
    {
        auto op1 = expr1.get().get().eval(), op2 = expr2.eval();
        Blasw::dot(wrsy<A2>(op1), wrge(op2), wrge(expr), 1, 0);
    }
    else if constexpr (issy<B2>() && ists<B3>())
    {
        auto op1 = expr1.eval(), op2 = expr2.get().get().eval();
        Blasw::dot(wrge(op1), wrsy<B2>(op2), wrge(expr), 1, 0);
    }
    ////////////////////////////////////////////////////////////////////////////
    else if constexpr (ists<A2>() && istr<A3>())
    {
        expr = expr2.expr();
        auto op1 = expr1.get().get().eval(), op2 = expr;
        Blasw::dot(wrtr<A3>(op1, true).trans(), wrge(op2), 1);
    }
    else if constexpr (ists<B2>() && istr<B3>())
    {
        expr = expr1.expr();
        auto op1 = expr, op2 = expr2.get().get().eval();
        Blasw::dot(wrge(op1), wrtr<B3>(op2, true).trans(), 1);
    }
    else if constexpr (ists<A2>() && issy<A3>())
    {
        auto op1 = expr1.get().get().eval(), op2 = expr2.eval();
        Blasw::dot(wrsy<A3>(op1, true), wrge(op2), wrge(expr), 1, 0);
    }
    else if constexpr (ists<B2>() && issy<B3>())
    {
        auto op1 = expr1.eval(), op2 = expr2.get().get().eval();
        Blasw::dot(wrge(op1), wrsy<B3>(op2, true), wrge(expr), 1, 0);
    }
    ////////////////////////////////////////////////////////////////////////////
    else if constexpr (istr<A3>())
    {
        expr = expr2.expr();
        auto op1 = expr1.get().eval(), op2 = expr;
        Blasw::dot(wrtr<A3>(op1), wrge(op2), 1);
    }
    else if constexpr (istr<B3>())
    {
        expr = expr1.expr();
        auto op1 = expr, op2 = expr2.get().eval();
        Blasw::dot(wrge(op1), wrtr<B3>(op2), 1);
    }
    else if constexpr (issy<A3>())
    {
        auto op1 = expr1.get().eval(), op2 = expr2.eval();
        Blasw::dot(wrsy<A3>(op1), wrge(op2), wrge(expr), 1, 0);
    }
    else if constexpr (issy<B3>())
    {
        auto op1 = expr1.eval(), op2 = expr2.get().eval();
        Blasw::dot(wrge(op1), wrsy<B3>(op2), wrge(expr), 1, 0);
    }
    ////////////////////////////////////////////////////////////////////////////
    else if constexpr (ists<A2>() && ishe<A3>())
    {
        auto op1 = expr1.get().get().eval(), op2 = expr2.eval();
        Blasw::dot(wrhe<A3>(op1, true), wrge(op2), wrge(expr), 1, 0);
    }
    else if constexpr (ists<B2>() && ishe<B3>())
    {
        auto op1 = expr1.eval(), op2 = expr2.get().get().eval();
        Blasw::dot(wrge(op1), wrhe<B3>(op2, true), wrge(expr), 1, 0);
    }
    else if constexpr (ishe<A3>())
    {
        auto op1 = expr1.get().eval(), op2 = expr2.eval();
        Blasw::dot(wrhe<A3>(op1), wrge(op2), wrge(expr), 1, 0);
    }
    else if constexpr (ishe<B3>())
    {
        auto op1 = expr1.eval(), op2 = expr2.get().eval();
        Blasw::dot(wrhe(op1), wrtr<B3>(op2), wrge(expr), 1, 0);
    }
    ////////////////////////////////////////////////////////////////////////////
    else
    {
        DERIVED op1, op2;
        auto state1 = Blasw::State::None, state2 = Blasw::State::None;

        if constexpr ((ists<A2>() && iscj<A3>()) || (ists<A3>() && iscj<A2>()))
            op1 = expr1.get().get().eval(), state1 = Blasw::State::ConjTrans;
        else if constexpr (ists<A3>())
            op1 = expr1.get().eval(), state1 = Blasw::State::Trans;
        else
            op1 = expr1.eval();

        if constexpr ((ists<B2>() && iscj<B3>()) || (ists<B3>() && iscj<B2>()))
            op2 = expr2.get().get().eval(), state2 = Blasw::State::ConjTrans;
        else if constexpr (ists<B3>())
            op2 = expr2.get().eval(), state2 = Blasw::State::Trans;
        else
            op2 = expr2.eval();

        auto wr1 = wrge(op1), wr2 = wrge(op2);
        wr1.state = state1, wr2.state = state2;
        Blasw::dot(wr1, wr2, wrge(expr), 1, 0);
    }
}

template <typename Expr1>
auto inverse(const Expr1& expr1)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    if constexpr (issy<A3>())
    {
        auto expr = expr1.get().copy();
        if (!Blasw::inverse(wrsy<A3>(expr))) expr.reset();
        return expr;
    }
    else
    {
        auto expr = expr1.copy();
        if (!Blasw::inverse(wrge(expr))) expr.reset();
        return expr;
    }
}

template <typename Expr1>
auto determinant(const Expr1& expr1)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    if constexpr (issy<A3>())
    {
        auto expr = expr1.get().copy();
        return Blasw::determinant(wrsy<A3>(expr));
    }
    else if constexpr (issy<A3>())
    {
        auto expr = expr1.get().copy();
        return Blasw::determinant(wrhe<A3>(expr));
    }
    else
    {
        auto expr = expr1.copy();
        return Blasw::determinant(wrge(expr));
    }
}

template <typename Expr1>
auto lufact(const Expr1& expr1)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    Matrix<M, int> pivot(std::min(expr1.rows(), expr1.cols()), 1);

    if constexpr (issy<A3>())
    {
        auto expr = expr1.get().copy();
        if (!Blasw::lufact(wrsy<A3>(expr), Blasw::vec(pivot.data(), pivot.rows()))) expr.reset(), pivot.reset();
        return std::make_pair(expr, pivot);
    }
    else if constexpr (ishe<A3>())
    {
        auto expr = expr1.get().copy();
        if (!Blasw::lufact(wrhe<A3>(expr), Blasw::vec(pivot.data(), pivot.rows()))) expr.reset(), pivot.reset();
        return std::make_pair(expr, pivot);
    }
    else
    {
        auto expr = expr1.copy();
        if (!Blasw::lufact(wrge(expr), Blasw::vec(pivot.data(), pivot.rows()))) expr.reset(), pivot.reset();
        return std::make_pair(expr, pivot);
    }
}

template <typename Expr1>
auto qrfact(const Expr1& expr1)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    DERIVED pivot(std::min(expr1.rows(), expr1.cols()), 1);
    auto expr = expr1.copy();
    if (!Blasw::qrfact(wrge(expr), Blasw::vec(pivot.data(), pivot.rows()))) expr.reset(), pivot.reset();
    return std::make_pair(expr, pivot);
}

template <typename Expr1>
auto cholesky(const Expr1& expr1)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    static_assert(istr<A3>(), "Matrix must be upper or lower in matrix::cholesky.");
    auto expr = expr1.get().copy();
    auto wr = wrtr<A3>(expr);
    if (!Blasw::cholesky(Blasw::Posdef<T>(wr.data, wr.size, wr.stride, wr.major, wr.tri))) expr.reset();
    return expr;
}

template <typename Expr1, typename Expr2>
auto solve(const Expr1& expr1, const Expr2& expr2)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    auto right = expr2.copy();

    if constexpr (issy<A3>())
    {
        auto expr = expr1.get().copy();
        if (!Blasw::solve(wrsy<A3>(expr), wrge(right))) right.reset();
        return right;
    }
    else if constexpr (ishe<A3>())
    {
        auto expr = expr1.get().copy();
        if (!Blasw::solve(wrhe<A3>(expr), wrge(right))) right.reset();
        return right;
    }
    else
    {
        auto expr = expr1.copy();
        if (!Blasw::solve(wrge(expr), wrge(right))) right.reset();
        return right;
    }
}

template <typename Expr1>
auto eigen(const Expr1& expr1, bool _left, bool _right)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    if constexpr (issy<A3>() && !IsComplex<T>::value)
    {
        DERIVED eigen(expr1.rows(), 1);
        auto wrei = Blasw::vec(eigen.data(), eigen.rows());
        auto expr = expr1.get().copy();
        if (!Blasw::eigen(wrsy<A3>(expr), wrei, _left || _right)) eigen.reset(), expr.reset();
        return std::make_pair(eigen, expr);
    }
    else if constexpr (ishe<A3>())
    {
        Matrix<M, typename FromComplex<T>::Type> eigen(expr1.rows(), 1);
        auto wrei = Blasw::vec(eigen.data(), eigen.rows());
        auto expr = expr1.get().copy();
        if (!Blasw::eigen(wrsy<A3>(expr), wrei, _left || _right)) eigen.reset(), expr.reset();
        return std::make_pair(eigen, expr);
    }
    else
    {
        Matrix<M, typename ToComplex<T>::Type> eigen(expr1.rows(), 1);
        auto wrei = Blasw::vec(eigen.data(), eigen.rows());
        auto expr = expr1.copy();
        auto left = _left ? DERIVED(expr1.rows(), expr1.rows()) : DERIVED();
        auto right = _right ? DERIVED(expr1.cols(), expr1.cols()) : DERIVED();
        if (!Blasw::eigen(wrge(expr), wrei, wrge(left), wrge(right))) eigen.reset(), left.reset(), right.reset();
        return std::make_tuple(eigen, left, right);
    }
}

template <typename Expr1>
auto schur(const Expr1& expr1, bool _vectors)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    Matrix<M, typename ToComplex<T>::Type> eigen(expr1.rows(), 1);
    auto wrei = Blasw::vec(eigen.data(), eigen.rows());
    auto expr = expr1.copy();
    auto vectors = _vectors ? DERIVED(expr1.rows(), expr1.rows()) : DERIVED();
    if (!Blasw::schur(wrge(expr), wrei, wrge(vectors))) eigen.reset(), vectors.reset();
    return std::make_tuple(eigen, vectors);
}

template <typename Expr1, typename _T>
auto rank(const Expr1& expr1, _T epsilon)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;
    auto expr = expr1.copy();
    return Blasw::rank(wrge(expr), epsilon);
}

template <typename Expr1, typename Expr2>
auto lsquares(const Expr1& expr1, const Expr2& expr2)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    auto right = expr2.copy();

    if constexpr (ists<A3>())
    {
        auto expr = expr1.get().copy();
        if (!Blasw::lsquares(wrge(expr).trans(), wrge(right))) return DERIVED();
        return right.block(0, 0, expr1.cols(), right.cols()).eval();
    }
    else
    {
        auto expr = expr1.copy();
        if (!Blasw::lsquares(wrge(expr), wrge(right))) return DERIVED();
        return right.block(0, 0, expr1.cols(), right.cols()).eval();
    }
}

template <typename Expr1>
auto svd(const Expr1& expr1, bool _left, bool _right)
{
    using A3 = Expr1;
    using A2 = typename GetExpr<A3>::Type;
    using T = typename Expr1::Type;
    using M = typename Expr1::Major;

    Matrix<M, typename FromComplex<T>::Type> singular(std::min(expr1.rows(), expr1.cols()), 1);
    auto wrei = Blasw::vec(singular.data(), singular.rows());
    auto expr = expr1.copy();
    auto left = _left ? DERIVED(expr1.rows(), expr1.rows()) : DERIVED();
    auto right = _right ? DERIVED(expr1.cols(), expr1.cols()) : DERIVED();
    if (!Blasw::svd(wrge(expr), wrei, wrge(left), wrge(right))) singular.reset(), left.reset(), right.reset();
    return std::make_tuple(singular, left, right);
}
}  // namespace External
}  // namespace Tense::MatrixImpl
