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

#include <tense/common.h>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace Tense::Impl
{
template <typename T>
using IsWritable = typename std::enable_if<std::is_same<T, Writable>::value>::type;

template <typename Status1, typename Status2>
struct SumWritable
{
    using Type = Readable;
};
template <>
struct SumWritable<Writable, Writable>
{
    using Type = Writable;
};

template <typename Type>
struct IsReal : std::false_type
{
};
template <>
struct IsReal<float> : std::true_type
{
};
template <>
struct IsReal<double> : std::true_type
{
};
template <>
struct IsReal<std::complex<float>> : std::true_type
{
};
template <>
struct IsReal<std::complex<double>> : std::true_type
{
};

template <typename T>
struct IsComplex : std::false_type
{
};
template <typename T>
struct IsComplex<std::complex<T>> : std::true_type
{
};

template <typename T>
struct ToComplex
{
    using Type = std::complex<T>;
};
template <typename T>
struct ToComplex<std::complex<T>>
{
    using Type = std::complex<T>;
};

template <typename T>
struct FromComplex
{
    using Type = T;
};
template <typename T>
struct FromComplex<std::complex<T>>
{
    using Type = T;
};

template <typename T>
std::string TypeName()
{
    if constexpr (IsComplex<T>::value)
        return "c" + TypeName<typename T::value_type>();
    else if (std::is_same<T, bool>::value)
        return "bool";
    else if (std::is_same<T, int8_t>::value)
        return "i8";
    else if (std::is_same<T, int16_t>::value)
        return "i16";
    else if (std::is_same<T, int32_t>::value)
        return "i32";
    else if (std::is_same<T, int64_t>::value)
        return "i64";
    else if (std::is_same<T, uint8_t>::value)
        return "u8";
    else if (std::is_same<T, uint16_t>::value)
        return "u16";
    else if (std::is_same<T, uint32_t>::value)
        return "u32";
    else if (std::is_same<T, uint64_t>::value)
        return "u64";
    else if (std::is_same<T, float>::value)
        return "f32";
    else if (std::is_same<T, double>::value)
        return "f64";
#ifdef __SIZEOF_INT128__
    else if (std::is_same<T, __int128_t>::value)
        return "i128";
    else if (std::is_same<T, __uint128_t>::value)
        return "u128";
#endif
    else
        return "?";
}
}  // namespace Tense::Impl
