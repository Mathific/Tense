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

#include <tense/vector/struct.h>

#include <complex>
#include <random>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#define C ,

#define EXPR(NAME, STATUS, TYPE, ARGS, SIZE)    \
    struct NAME : public Base<TYPE, NAME<ARGS>> \
    {                                           \
        using Type = TYPE;                      \
        using Status = STATUS;                  \
        using Flag = void;                      \
                                                \
        Size size() const { return SIZE; }      \
                                                \
    private:                                    \
        using This = NAME<ARGS>;

#define FLAGGED(NAME, FLAG, STATUS, TYPE, ARGS, SIZE) \
    struct NAME : public Base<TYPE, NAME<ARGS>>       \
    {                                                 \
        using Type = TYPE;                            \
        using Status = STATUS;                        \
        using Flag = FLAG;                            \
                                                      \
        Size size() const { return SIZE; }            \
                                                      \
    private:                                          \
        using This = NAME<ARGS>;

#define OPERATOR(OP)                                                                                         \
    auto& operator OP## = (Type expr2)                                                                       \
    {                                                                                                        \
        Eval::eval(*this, *this OP expr2);                                                                   \
        return *this;                                                                                        \
    }                                                                                                        \
    template <typename _Expr, typename = IsExpr<_Expr>, typename Temp = Status, typename = IsWritable<Temp>> \
    auto& operator OP## = (const _Expr& expr)                                                                \
    {                                                                                                        \
        TENSE_VASSERT(size(), ==, expr.size(), "operator" + #OP, "size of vectors must be equal")            \
        Eval::eval(*this, *this OP expr);                                                                    \
        return *this;                                                                                        \
    }

#define WRITABLE(EXPR)                                                                                           \
    template <typename Temp = Status, typename = IsWritable<Temp>>                                               \
    Type& operator[](Size i)                                                                                     \
    {                                                                                                            \
        EXPR;                                                                                                    \
    }                                                                                                            \
    Type operator[](Size i) const { EXPR; }                                                                      \
                                                                                                                 \
    auto& operator=(Type val)                                                                                    \
    {                                                                                                            \
        Eval::assign(*this, val);                                                                                \
        return *this;                                                                                            \
    }                                                                                                            \
    template <typename _Expr, typename = IsExpr<_Expr>, typename Temp = Status, typename = IsWritable<Temp>>     \
    auto& operator=(const _Expr& expr)                                                                           \
    {                                                                                                            \
        TENSE_VASSERT(size(), ==, expr.size(), "assign", "Size of vectors must be equal")                        \
        Eval::eval(*this, expr);                                                                                 \
        return *this;                                                                                            \
    }                                                                                                            \
    auto& operator=(const std::initializer_list<Type>& list)                                                     \
    {                                                                                                            \
        TENSE_VASSERT(size(), ==, list.size(), "operator=", "Size of vector and initializer list must be equal") \
        Eval::assign(*this, list);                                                                               \
        return *this;                                                                                            \
    }                                                                                                            \
    auto& operator=(const This& other) { return this->operator= <This>(other); }                                 \
                                                                                                                 \
    OPERATOR(+)                                                                                                  \
    OPERATOR(*)                                                                                                  \
    OPERATOR(-)                                                                                                  \
    OPERATOR(/)                                                                                                  \
    OPERATOR(%)                                                                                                  \
    OPERATOR(&)                                                                                                  \
    OPERATOR(|)                                                                                                  \
    OPERATOR(^)                                                                                                  \
    OPERATOR(<<)                                                                                                 \
    OPERATOR(>>)

namespace Tense::VectorImpl
{
// clang-format off
template <typename T, typename Expr1, typename Func>
EXPR(Unary, Readable, T, T C Expr1 C Func, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;

public:
    Unary(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func) {}
    Type operator[](Size i) const { return _func(_expr1[i]); }
};

template <typename T, typename Expr1, typename Expr2, typename Func>
EXPR(Binary, Readable, T, T C Expr1 C Expr2 C Func,
     std::max(_expr1.size(), _expr2.size()))
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;

public:
    Binary(const Expr1 &expr1, const Expr2 &expr2, Func func)
        : _expr1(expr1), _expr2(expr2),
        _func(func) {}
    Type operator[](Size i) const
    {
        return _func(_expr1[i], _expr2[i]);
    }
};

template<typename Expr1, typename Func>
EXPR(Access, typename Expr1::Status, typename Expr1::Type,
     Expr1 C Func, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;

public:
    Access(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func) {}
    WRITABLE(return _func(i, _expr1))
};

template <typename Expr1>
EXPR(Repeat, Readable, typename Expr1::Type, Expr1, _size)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _size, _size2;

public:
    Repeat(const Expr1 &expr1, Size size)
        : _expr1(expr1), _size(size), _size2(expr1.size()) {}
   Type operator[](Size i) const { return _expr1[i % _size2]; }
};


template <typename T, typename Expr1, typename Func>
FLAGGED(BReduce, HeavyFlag, Readable, T, T C Expr1 C Func, _expr1.size() / _size)
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Type _init;
    const Size _size;

public:
    BReduce(const Expr1 &expr1, Func func, Type init, Size size)
        : _expr1(expr1), _func(func), _init(init), _size(size) {}
    Type operator[](Size i) const
    {
        Type val = _init;
        Size si = i * _size;
        for (Size i = si; i < si + _size; ++i) val = _func(val, _expr1[i]);
        return val;
    }
};

template <typename T, typename Expr1, typename Func>
T Reduce(const Expr1 &expr1, Func func, T init)
{
    T val = init;
    const Size size = expr1.size();
    for (Size i=0; i < size; ++i) val = func(val, expr1[i]);
    return val;
}

template <typename Expr1>
EXPR(SBlock, typename Expr1::Status, typename Expr1::Type, Expr1, _size)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _i, _size;

public:
    SBlock(const Expr1 &expr1, Size i, Size size)
        : _expr1(expr1), _i(i), _size(size) {}
    WRITABLE(return _expr1[_i + i])
};

template <typename Expr1>
EXPR(SElem, typename Expr1::Status, typename Expr1::Type, Expr1, 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _i;

public:
    SElem(const Expr1 &expr1, Size i) : _expr1(expr1), _i(i) {}
    WRITABLE(return _expr1[_i])
};

template <typename Expr1>
EXPR(RIndex, typename Expr1::Status, typename Expr1::Type, Expr1, _size)
    typename Alias<Expr1, Status>::Type _expr1;
    SSize _b, _s, _size;

public:
    RIndex(const Expr1 &expr1, Cut cut)
        : _expr1(expr1), _b(cut.start), _s(cut.step)
    {
        _size = (cut.end - cut.start + cut.step - 1) / cut.step;
    }
    WRITABLE(return _expr1[i * _s + _b])
};

template <typename Expr1>
EXPR(VIndex, typename Expr1::Status, typename Expr1::Type, Expr1, _indices.size())
    typename Alias<Expr1, Status>::Type _expr1;
    const std::vector<Size> _indices;

public:
    VIndex(const Expr1 &expr1, const std::vector<Size> &indices)
        : _expr1(expr1), _indices(indices) {}
    WRITABLE(return _expr1[_indices[i]])
};

template <typename Expr1, typename Expr2>
EXPR(Indirect, typename Expr1::Status, typename Expr1::Type,
     Expr1 C Expr2, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;

public:
    Indirect(const Expr1 &expr1, const Expr2 &expr2) : _expr1(expr1), _expr2(expr2) {}
    WRITABLE(return _expr1[_expr2[i]])
};

template <typename Expr1, typename Expr2>
EXPR(Cat0, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
     typename Expr1::Type, Expr1 C Expr2, _size)
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Size _size, _size1;

public:
    Cat0(const Expr1 &expr1, const Expr2 &expr2, Size size)
        : _expr1(expr1), _expr2(expr2), _size(size), _size1(expr1.size()) {}
    WRITABLE(return i < _size1 ? _expr1[i] : _expr2[i - _size1])
};


template <typename Expr1, typename Expr2>
EXPR(Cat1, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Type, Expr1 C Expr2, _expr1.size() * 2)
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;

public:
    Cat1(const Expr1 &expr1, const Expr2 &expr2) : _expr1(expr1), _expr2(expr2) {}
    WRITABLE(return i & 1 ? _expr2[i / 2] : _expr1[i / 2])
};

template <typename Switch, typename Expr1>
FLAGGED(BAny, HeavyFlag, Readable, bool, Switch C Expr1, _expr1.size() / _size)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _size;

public:
    BAny(const Expr1 &expr1, Size size) : _expr1(expr1), _size(size) {}
    Type operator[](Size i) const
    {
        Size si = i * _size;
        for (Size j = si; j < si + _size; ++j)
            if (!Switch::value ^ static_cast<bool>(_expr1[j])) return Switch::value;
        return !Switch::value;
    }
};

template <typename Switch, typename Expr1>
FLAGGED(Any, HeavyFlag, Readable, bool, Switch C Expr1, 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _size;

public:
    Any(const Expr1 &expr1) : _expr1(expr1), _size(expr1.size()) {}
    Type operator[](Size) const
    {
        for (Size i = 0; i < _size; ++i)
            if (!Switch::value ^ static_cast<bool>(_expr1[i])) return Switch::value;
        return !Switch::value;
    }
};

template <typename Expr1, typename Func>
Size Redux(const Expr1 &expr1, Func func)
{
    Size index = 0;
    const Size size = expr1.size();
    typename Expr1::Type val = expr1[0];
    for (Size i = 0; i < size; ++i)
        if (func(expr1[i], val)) index = i, val = expr1[i];
    return index;
}

template <int P, typename Expr1>
EXPR(Power, Readable, typename Expr1::Type, P C Expr1, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Power(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator[](Size i) const { return std::pow(_expr1[i], P); }
};

template <typename Expr1, typename Func>
EXPR(FWhere, typename Expr1::Status, typename Expr1::Type,
     Expr1 C Func, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    Type _val;

public:
    FWhere(const Expr1 &expr1, Func func, Type val) : _expr1(expr1), _func(func), _val(val) {}
    WRITABLE(return _func(_expr1[i]) ? _expr1[i] : _val)
};

template <typename Expr1, typename Func>
EXPR(IWhere, typename Expr1::Status, typename Expr1::Type,
     Expr1 C Func, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    Type _val;

public:
    IWhere(const Expr1 &expr1, Func func, Type val) : _expr1(expr1), _func(func), _val(val) {}
    WRITABLE(return _func(i) ? _expr1[i] : _val)
};

template <typename Expr1, typename Expr2, typename Func>
EXPR(FEWhere, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
     typename Expr1::Type, Expr1 C Expr2 C Func, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;

public:
    FEWhere(const Expr1 &expr1, const Expr2 &expr2, Func func) : _expr1(expr1), _expr2(expr2), _func(func) {}
    WRITABLE(return _func(_expr1[i]) ? _expr1[i] : _expr2[i])
};

template <typename Expr1, typename Expr2, typename Func>
EXPR(IEWhere, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
     typename Expr1::Type, Expr1 C Expr2 C Func, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;

public:
    IEWhere(const Expr1 &expr1,  const Expr2 &expr2, Func func) : _expr1(expr1), _expr2(expr2), _func(func) {}
    WRITABLE(return _func(i) ? _expr1[i] : _expr2[i])
};

template <typename Expr1>
EXPR(Polar, Readable, typename Expr1::Type, Expr1, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Polar(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator[](Size i) const { return std::polar(_expr1[i].real(), _expr1[i].imag()); }
};

template <typename T, typename Expr1>
EXPR(Convert, Readable, T, T C Expr1, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Convert(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator[](Size i) const { return Type(_expr1[i]); }
};

template <typename Expr1>
EXPR(Self, typename Expr1::Status, typename Expr1::Type, Expr1, _expr1.size())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Self(const Expr1 &expr1) : _expr1(expr1) {}
    WRITABLE(return _expr1[i])
};

template <typename T>
EXPR(Constant, Writable, T, T, _size)
    T _val;
    const Size _size;

public:
    Constant(Size size, T val)
        : _val(val), _size(size) {}
    WRITABLE(return _val)
};

template <typename T, typename Dist>
EXPR(Distribution, Readable, T, T C Dist, _size)
    Dist _dist;
    std::mt19937_64 _rand;
    const Size _size;
    Type operator[](Size i) { return Type(_dist(_rand)); }

public:
    Distribution(Size size, Dist &&dist) : _dist(std::forward<Dist>(dist)), _rand(rand()), _size(size) {}
    Type operator[](Size i) const { return const_cast<Distribution<Type, Dist> &>(*this)[i]; }
};

template <typename T>
EXPR(Sequence, Readable, T, T, _size)
    const Size _size;
    const Type _start, _step;

public:
    Sequence(Size size, Type start, Type step) : _size(size), _start(start), _step(step) {}
    Type operator[](Size i) const { return _start + i * _step; }
};

template <typename T>
EXPR(Strided, Writable, T, T, _size)
    Type *_data;
    const Size _size, _stride;

public:
    Strided(Size size, Type *data, Size stride)
        : _data(data), _size(size), _stride(stride) {}
    WRITABLE(return _data[i * _stride])
};

template <typename T, Size Count>
FLAGGED(Static, StaticFlag, Writable, T, T C Count, Count)
    T _data[Count];

public:
    Static(T value) { std::fill(_data, _data + Count, value); }
    Static(const std::vector<T> &list) { std::copy(list.begin(), list.end(), _data); }
    WRITABLE(return _data[i])
};

// ToVector

// Max and Min index

template<typename Expr1>
auto Turn(const Expr1 &expr1, Size _i)
{
    Size size = expr1.size();
    auto func = [size, _i](auto i, const auto &expr) { return expr[(i + _i) % size]; };
    return Access<Expr1, decltype(func)>(expr1, func);
}

template<typename Expr1>
auto MaxIdx(const Expr1 &expr1)
{
    auto func = [](auto val1, auto val2) { return val1 > val2; };
    return Redux<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto MinIdx(const Expr1 &expr1)
{
    auto func = [](auto val1, auto val2) { return val1 < val2; };
    return Redux<Expr1, decltype(func)>(expr1, func);
}

template <typename Expr1>
auto BFlip(const Expr1 &expr1, Size size)
{
    auto func = [size](auto i, const auto &expr)
    { return expr[i + size - 1 - 2 * (i % size)]; };
    return Access<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto Flip(const Expr1 &expr1)
{
    Size size = expr1.size() - 1;
    auto func = [size](auto i, const auto &expr) { return expr[size - i]; };
    return Access<Expr1, decltype(func)>(expr1, func);
}
// clang-format on
}  // namespace Tense::VectorImpl

#undef C
#undef EXPR
#undef FLAGGED
#undef OPERATOR
#undef WRITABLE
