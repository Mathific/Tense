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

#include <tense/struct.h>

#include <complex>
#include <type_traits>

namespace Tense::MatrixImpl
{
using namespace Tense::Impl;

template <typename T>
using IL1D = std::initializer_list<T>;

template <typename T>
using IL2D = std::initializer_list<std::initializer_list<T>>;

struct Expr
{
};

template <typename Major, typename Type, typename Derived>
class Base;

template <typename T>
using IsExpr = typename std::enable_if<std::is_base_of<Expr, T>::value>::type;

template <typename T>
using NotExpr = typename std::enable_if<!std::is_base_of<Expr, T>::value>::type;

template <typename T>
struct IsMatrix : std::false_type
{
};
template <typename Major, typename Type>
struct IsMatrix<Matrix<Major, Type>> : std::true_type
{
};

////////////////////////////////////////////////////////////////////////////////

struct HeavyFlag
{
};
struct StaticFlag
{
};
struct TransFlag
{
};
struct ConjFlag
{
};
struct USymmFlag
{
};
struct LSymmFlag
{
};
struct UHermFlag
{
};
struct LHermFlag
{
};
struct UpperFlag
{
};
struct LowerFlag
{
};
struct OUpperFlag
{
};
struct OLowerFlag
{
};

template <typename T>
using IsStatic = std::is_same<StaticFlag, typename T::Flag>;

template <typename T>
using IsHeavy = std::is_same<HeavyFlag, typename T::Flag>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Enable = void>
struct ReadableAlias
{
    using Type = const T;
};
template <typename T>
struct ReadableAlias<T, typename std::enable_if<IsStatic<T>::value>::type>
{
    using Type = const T &;
};
template <typename T, typename Enable = void>
struct WritableAlias
{
    using Type = T;
};
template <typename T>
struct WritableAlias<T, typename std::enable_if<IsStatic<T>::value>::type>
{
    using Type = T &;
};

template <typename T, typename S, typename Enable>
struct Alias
{
};
template <typename T>
struct Alias<T, Readable>
{
    using Type = typename ReadableAlias<T>::Type;
};
template <typename T>
struct Alias<T, Writable>
{
    using Type = typename WritableAlias<T>::Type;
};

////////////////////////////////////////////////////////////////////////////////

template <Size I, Size J, Size Rows, Size Cols>
struct StaticEval
{
    template <typename Expr1, typename Expr2>
    static void ceval(Expr1 &expr1, const Expr2 &expr2)
    {
        expr1(I, J) = expr2(I, J);
        if constexpr (I + 1 < Rows)
            StaticEval<I + 1, J, Rows, Cols>::ceval(expr1, expr2);
        else if constexpr (I + 1 == Rows && J + 1 < Cols)
            StaticEval<0, J + 1, Rows, Cols>::ceval(expr1, expr2);
    }
    template <typename Expr1, typename Expr2>
    static void reval(Expr1 &expr1, const Expr2 &expr2)
    {
        expr1(I, J) = expr2(I, J);
        if constexpr (J + 1 < Cols)
            StaticEval<I, J + 1, Rows, Cols>::reval(expr1, expr2);
        else if constexpr (J + 1 == Cols && I + 1 < Rows)
            StaticEval<I + 1, 0, Rows, Cols>::reval(expr1, expr2);
    }
    template <typename Expr1, typename Expr2>
    static void eval(Expr1 &expr1, const Expr2 &expr2)
    {
        if constexpr (std::is_same<typename Expr1::Major, Col>::value)
            ceval(expr1, expr2);
        else
            reval(expr1, expr2);
    }
};

struct Eval
{
    template <typename Expr1>
    static void assign(Expr1 &expr1, IL1D<typename Expr1::Type> list)
    {
        Size i = 0;
        for (auto item : list) expr1(i++, 0) = item;
    }
    template <typename Expr1>
    static void assign(Expr1 &expr1, IL2D<typename Expr1::Type> list)
    {
        Size rows = expr1.rows(), cols = expr1.cols();

        Size i = 0;
        for (auto row : list)
        {
            TENSE_MASSERT(cols, ==, row.size(), "assign", "2D initializer list input has rows with different length")

            Size j = 0;
            for (auto item : row)
            {
                expr1(i, j) = item;
                ++j;
            }
            ++i;
        }
    }
    template <typename Expr1>
    static void assign(Expr1 &expr1, typename Expr1::Type expr2)
    {
        auto rows = expr1.rows(), cols = expr1.cols();
        if constexpr (std::is_same<typename Expr1::Major, Col>::value)
#pragma omp parallel for
            for (Size j = 0; j < cols; ++j)
                for (Size i = 0; i < rows; ++i) expr1(i, j) = expr2;
        else
#pragma omp parallel for
            for (Size i = 0; i < rows; ++i)
                for (Size j = 0; j < cols; ++j) expr1(i, j) = expr2;
    }

    template <typename Expr1, typename Expr2>
    static void eval(Expr1 &expr1, const Expr2 &expr2)
    {
        auto rows = expr1.rows(), cols = expr1.cols();

        if constexpr (std::is_same<typename Expr1::Major, Col>::value)
#pragma omp parallel for
            for (Size j = 0; j < cols; ++j)
                for (Size i = 0; i < rows; ++i) expr1(i, j) = expr2(i, j);
        else
#pragma omp parallel for
            for (Size i = 0; i < rows; ++i)
                for (Size j = 0; j < cols; ++j) expr1(i, j) = expr2(i, j);
    }
};
}  // namespace Tense::MatrixImpl
