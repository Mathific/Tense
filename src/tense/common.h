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

#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#define TENSE_ALIGNMENT 64

#define TENSE_MASSERT(first, op, second, name, msg)                                                                \
    if (!static_cast<bool>(first op second))                                                                       \
    {                                                                                                              \
        std::stringstream stream;                                                                                  \
        stream << "Error: " << (first) << " " << #op << " " << (second) << " -> " << msg << " in matrix::" << name \
               << ".";                                                                                             \
        throw std::runtime_error(stream.str());                                                                    \
    }

#define TENSE_TASSERT(first, op, second, name, msg)                                                                \
    if (!static_cast<bool>(first op second))                                                                       \
    {                                                                                                              \
        std::stringstream stream;                                                                                  \
        stream << "Error: " << (first) << " " << #op << " " << (second) << " -> " << msg << " in tensor::" << name \
               << ".";                                                                                             \
        throw std::runtime_error(stream.str());                                                                    \
    }

namespace Tense
{
using Size = std::size_t;

constexpr Size Inf = std::numeric_limits<Size>::max();

using Shape = std::vector<Size>;

enum class Mode
{
    Hold,
    Copy,
    Own,
};

struct Row
{
};
struct Col
{
};
struct Readable
{
};
struct Writable
{
};

struct Cut
{
    Size start, step, end;
    inline Cut() : start(0), step(0), end(0) {}
    inline Cut(Size end) : start(0), step(1), end(end) {}
    inline Cut(Size start, Size end) : start(start), end(end) { step = end >= start ? 1 : -1; }
    inline Cut(Size start, Size step, Size end) : start(start), step(step), end(end) {}
};

namespace MatrixImpl
{
struct Expr;

template <typename Major, typename Type>
class Matrix;

template <typename M, typename Expr1>
struct ToMatrix;

template <typename T, typename S, typename Enable = void>
struct Alias;
}  // namespace MatrixImpl

namespace TensorImpl
{
struct Expr;

template <typename Type>
class Tensor;

template <typename Expr1>
struct ToTensor;

template <typename T, typename S, typename Enable = void>
struct Alias;
}  // namespace TensorImpl

template <typename Major, typename Type>
using Matrix = MatrixImpl::Matrix<Major, Type>;

template <typename Type>
using Tensor = TensorImpl::Tensor<Type>;

inline std::ostream &operator<<(std::ostream &os, const Shape &shape)
{
    os << "Shape<";
    for (Size i = 0; i < shape.size(); ++i) os << shape[i] << (i + 1 == shape.size() ? "" : ",");
    return os << ">";
}
}  // namespace Tense
