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

#include <algorithm>
#include <complex>
#include <numeric>
#include <type_traits>

namespace Tense::TensorImpl
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
struct IsTensor : std::false_type
{
};
template <typename Type>
struct IsTensor<Tensor<Type>> : std::true_type
{
};

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Enable = void>
struct ReadableAlias
{
    using Type = const T;
};
template <typename T, typename Enable = void>
struct WritableAlias
{
    using Type = T;
};

template <typename T, typename Status, typename Enable>
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

namespace Helper
{
inline Size elems(const Shape &shape, Size margin1 = 0, Size margin2 = 0)
{
    return std::accumulate(shape.begin() + margin1, shape.end() - margin2, Size(1), std::multiplies<Size>());
}
inline Shape left(const Shape &shape, Size dim)
{
    if (dim == 0) return {1};
    return Shape(shape.begin(), shape.begin() + dim);
}
inline Shape right(const Shape &shape, Size dim)
{
    if (dim == shape.size()) return {1};
    return Shape(shape.begin() + dim, shape.end());
}
inline Shape stride(const Shape &shape)
{
    Shape stride(shape.size(), 1);
    for (Size i = shape.size() - 1; i > 0; --i) stride[i - 1] = stride[i] * shape[i];
    return stride;
}
inline Shape remove(Shape shape)
{
    Size i = 0;
    for (; i < shape.size(); ++i)
        if (shape[i] != 1) break;
    return Shape(shape.begin() + i, shape.end());
}
inline void check(const Shape &shape)
{
    for (const auto &item : shape) TENSE_TASSERT(item, >, 0, "check", "Shape item can't be zero")
}
inline Size check(const Shape &shape, Size dim)
{
    TENSE_TASSERT(dim, <, shape.size(), "check", "Dimension must be less than tensor dimension")
    return Helper::elems(shape, dim);
}
inline Size check(const Shape &shape1, const Shape &shape2)
{
    Size diff = std::abs(int64_t(shape1.size()) - int64_t(shape2.size()));
    if (shape1.size() > shape2.size())
    {
        TENSE_TASSERT(right(shape1, diff), ==, shape2, "check", "Tensor shapes must match")
        return std::accumulate(shape1.begin(), shape1.begin() + diff, Size(1), std::multiplies<Size>());
    }
    else
    {
        TENSE_TASSERT(shape1, ==, right(shape2, diff), "check", "Tensor shapes must match")
        return std::accumulate(shape2.begin(), shape2.begin() + diff, Size(1), std::multiplies<Size>());
    }
}
inline void replace(Shape &shape, Size size)
{
    auto it = std::find(shape.begin(), shape.end(), 0);
    if (it != shape.end())
    {
        *it = 1;
        *it = size / elems(shape);
    }
    TENSE_TASSERT(std::find(it, shape.end(), 0) == shape.end(), ==, true, "replace", "Shape item can't be zero")
    TENSE_TASSERT(Helper::elems(shape) == size, ==, true, "replace", "Sizes of shapes must be equal")
}
inline Size view(const Shape &shape, const Shape &indexes)
{
    Size index = 0, size = Helper::elems(shape, 1);
    for (Size i = 0; i < indexes.size(); ++i)
    {
        index += indexes[i] * size;
        if (i + 1 != indexes.size()) size /= shape[i + 1];
    }
    return index;
}
inline Size item(const Shape &shape, const Shape &indexes)
{
    Size index = 0, size = 1, diff = shape.size() - indexes.size();
    for (Size i = indexes.size(); i > 0; --i)
    {
        index += size * indexes[i - 1];
        size *= shape[diff + i - 1];
    }
    return index;
}
}  // namespace Helper

namespace Access
{
template <Size I, Size N, typename Head, typename... Tail>
Size item(const Shape &stride, Head head, Tail &&...tail)
{
    if constexpr (I == N - 1)
        return head;
    else
        return stride[I] * head + item<I + 1, N>(stride, std::forward<Tail>(tail)...);
}
}  // namespace Access

////////////////////////////////////////////////////////////////////////////////

struct Eval
{
    template <typename Expr1>
    static void assign(Expr1 &expr1, typename Expr1::Type expr2)
    {
        auto size = Helper::elems(expr1.shape());
        for (Size i = 0; i < size; ++i) expr1[i] = expr2;
    }
    template <typename Expr1, typename Expr2>
    static void eval(Expr1 &expr1, const Expr2 &expr2)
    {
        auto size = Helper::elems(expr1.shape());
#pragma omp parallel for
        for (Size i = 0; i < size; ++i) expr1[i] = expr2[i];
    }
};
}  // namespace Tense::TensorImpl
