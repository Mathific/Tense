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

#include <tense/struct.h>

#include <type_traits>

namespace Tense::VectorImpl
{
using namespace Tense::Impl;

struct Expr
{
};

template <typename Type, typename Derived>
class Base;

template <typename T>
using IsExpr = typename std::enable_if<std::is_base_of<Expr, T>::value>::type;

template <typename T>
using NotExpr = typename std::enable_if<!std::is_base_of<Expr, T>::value>::type;

template <typename T>
struct IsVector : std::false_type
{
};
template <typename Type>
struct IsVector<Vector<Type>> : std::true_type
{
};

////////////////////////////////////////////////////////////////////////////////

struct HeavyFlag
{
};
struct StaticFlag
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
    using Type = const T&;
};
template <typename T, typename Enable = void>
struct WritableAlias
{
    using Type = T;
};
template <typename T>
struct WritableAlias<T, typename std::enable_if<IsStatic<T>::value>::type>
{
    using Type = T&;
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

template <Size I, Size Count>
struct StaticEval
{
    template <typename Expr1, typename Expr2>
    static void eval(Expr1& expr1, const Expr2& expr2)
    {
        expr1[I] = expr2[I];
        if constexpr (I + 1 < Count) StaticEval<I + 1, Count>::eval(expr1, expr2);
    }
};

struct Eval
{
    template <typename Expr1>
    static void assign(Expr1& expr1, std::initializer_list<typename Expr1::Type> list)
    {
        Size i = 0;
        for (auto item : list) expr1[i++] = item;
    }
    template <typename Expr1>
    static void assign(Expr1& expr1, typename Expr1::Type expr2)
    {
        auto size = expr1.size();
        TENSE_PARALLEL_FOR
        for (Size i = 0; i < size; ++i) expr1[i] = expr2;
    }

    template <typename Expr1, typename Expr2>
    static void eval(Expr1& expr1, const Expr2& expr2)
    {
        auto size = expr1.size();
        TENSE_PARALLEL_FOR
        for (Size i = 0; i < size; ++i) expr1[i] = expr2[i];
    }
};
}  // namespace Tense::VectorImpl
