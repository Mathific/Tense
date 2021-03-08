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

#include <tense/tensor/struct.h>

#include <complex>
#include <random>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#define C ,

#define EXPR(NAME, STATUS, TYPE, ARGS, SHAPE)   \
    struct NAME : public Base<TYPE, NAME<ARGS>> \
    {                                           \
        using Type = TYPE;                      \
        using Status = STATUS;                  \
                                                \
        Shape shape() const { return SHAPE; }   \
                                                \
    private:                                    \
        using This = NAME<ARGS>;

#define OPERATOR(OP)                                                                                         \
    auto &operator OP##=(Type expr2)                                                                         \
    {                                                                                                        \
        Eval::eval(*this, *this OP expr2);                                                                   \
        return *this;                                                                                        \
    }                                                                                                        \
    template <typename _Expr, typename = IsExpr<_Expr>, typename Temp = Status, typename = IsWritable<Temp>> \
    auto &operator OP##=(const _Expr &expr)                                                                  \
    {                                                                                                        \
        TENSE_TASSERT(shape(), ==, expr.shape(), "operator" << #OP, "Shapes of tensors must be equal")       \
        Eval::eval(*this, *this OP expr);                                                                    \
        return *this;                                                                                        \
    }

#define WRITABLE(EXPR)                                                                                       \
    template <typename Temp = Status, typename = IsWritable<Temp>>                                           \
    Type &operator[](Size idx)                                                                               \
    {                                                                                                        \
        EXPR;                                                                                                \
    }                                                                                                        \
    Type operator[](Size idx) const { EXPR; }                                                                \
                                                                                                             \
    auto &operator=(Type val)                                                                                \
    {                                                                                                        \
        Eval::assign(*this, val);                                                                            \
        return *this;                                                                                        \
    }                                                                                                        \
    template <typename _Expr, typename = IsExpr<_Expr>, typename Temp = Status, typename = IsWritable<Temp>> \
    auto &operator=(const _Expr &expr)                                                                       \
    {                                                                                                        \
        TENSE_TASSERT(shape(), ==, expr.shape(), "assign", "Shapes of tensors must be equal")                \
        Eval::eval(*this, expr);                                                                             \
        return *this;                                                                                        \
    }                                                                                                        \
    auto &operator=(const This &other) { return this->operator=<This>(other); }                              \
                                                                                                             \
    OPERATOR(+)                                                                                              \
    OPERATOR(-)                                                                                              \
    OPERATOR(*)                                                                                              \
    OPERATOR(/)                                                                                              \
    OPERATOR(%)                                                                                              \
    OPERATOR(&)                                                                                              \
    OPERATOR(|)                                                                                              \
    OPERATOR(^)                                                                                              \
    OPERATOR(<<)                                                                                             \
    OPERATOR(>>)

namespace Tense::TensorImpl
{
// clang-format off
template <typename T, typename Expr1, typename Func>
EXPR(Unary, Readable, T, T C Expr1 C Func, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;

public:
    Unary(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func) {}
    Type operator[](Size idx) const { return _func(_expr1[idx]); }
};

template <typename T, typename Expr1, typename Expr2, typename Func>
EXPR(Binary, Readable, T, T C Expr1 C Expr2 C Func, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;
    const bool _e1, _e2;
    const Size _size;

public:
    Binary(const Expr1 &expr1, const Expr2 &expr2, Func func, bool e1, bool e2, Size size)
        : _e1(e1), _e2(e2), _size(size), _expr1(expr1), _expr2(expr2), _func(func) { }
    Type operator[](Size idx) const
    {
        if (_e1 && !_e2)
            return _func(_expr1[idx], _expr2[idx % _size]);
        else if (!_e1 && _e2)
            return _func(_expr1[idx % _size], _expr2[idx]);
        else
            return _func(_expr1[idx], _expr2[idx]);
    }
};

template <typename T, typename Expr1, typename Func>
EXPR(Reduce, Readable, T, T C Expr1 C Func, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Shape _shape;
    const Size _size;
    const Type _init;

public:
    Reduce(const Expr1 &expr1, Func func, const Shape &shape, Size size, T init)
        : _expr1(expr1), _func(func), _shape(shape), _size(size), _init(init)
    {
    }
    Type operator[](Size idx) const
    {
        Type val = _init;
        const auto start = idx * _size;
        for (Size i = start; i < start + _size; ++i) val = _func(val, _expr1[i]);
        return val;
    }
};

template <typename Expr1>
EXPR(View, typename Expr1::Status, typename Expr1::Type, Expr1, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Shape _shape;
    const Size _idx;

public:
    View(const Expr1 &expr1, const Shape &shape, Size idx) : _expr1(expr1), _shape(shape), _idx(idx) {}
    WRITABLE(return _expr1[idx + _idx])
};

template <typename Expr1>
EXPR(Strided, typename Expr1::Status, typename Expr1::Type, Expr1, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Shape _shape;
    const Size _idx, _stride;

public:
    Strided(const Expr1 &expr1, const Shape &shape, Size idx, Size stride)
        : _expr1(expr1), _shape(shape), _idx(idx), _stride(stride)
    {
    }
    WRITABLE(return _expr1[_idx + idx * _stride])
};

template <typename Expr1>
EXPR(Repeat, Readable, typename Expr1::Type, Expr1, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Shape _shape;
    const Size _size;

public:
    Repeat(const Expr1 &expr1, const Shape &shape, Size size) : _expr1(expr1), _shape(shape), _size(size) {}
    Type operator[](Size idx) const { return _expr1[idx % _size]; }
};

template <typename Expr1, typename Expr2>
EXPR(Indirect, typename Expr1::Status, typename Expr1::Type, Expr1 C Expr2, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;

public:
    Indirect(const Expr1 &expr1, const Expr2 &expr2) : _expr1(expr1), _expr2(expr2) {}
    WRITABLE(return _expr1[_expr2[idx]])
};

template <typename Expr1>
EXPR(Flip, typename Expr1::Status, typename Expr1::Type, Expr1, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _size;

public:
    Flip(const Expr1 &expr1, Size size) : _expr1(expr1), _size(size) {}
    WRITABLE(return _expr1[idx + _size - 1 - 2 * (idx % _size)])
};

template <typename Expr1>
EXPR(All, Readable, bool, Expr1, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Shape _shape;
    const Size _size;

public:
    All(const Expr1 &expr1, const Shape &shape, Size size) : _expr1(expr1), _shape(shape), _size(size) {}
    Type operator[](Size idx) const
    {
        const auto start = idx * _size;
        for (Size i = start; i < start + _size; ++i)
            if (!_expr1[i]) return false;
        return true;
    }
};

template <typename Expr1>
EXPR(Any, Readable, bool, Expr1, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Shape _shape;
    const Size _size;

public:
    Any(const Expr1 &expr1, const Shape &shape, Size size)
        : _expr1(expr1), _shape(shape), _size(size) {}
    Type operator[](Size idx) const
    {
        const auto start = idx * _size;
        for (Size i = start; i < start + _size; ++i)
            if (_expr1[i]) return true;
        return false;
    }
};

template <typename Expr1, typename Func>
EXPR(Redux, Readable, Size, Expr1 C Func, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Shape _shape;
    const Size _size;

public:
    Redux(const Expr1 &expr1, Func func, const Shape &shape, Size size)
        : _expr1(expr1), _func(func), _shape(shape), _size(size) {}
    Type operator[](Size idx) const
    {
        Size index = 0, start = idx * _size;
        Type val = _expr1[start];
        for (Size i = start; i < start + _size; ++i)
            if (_func(_expr1[i], val)) index = i - start, val = _expr1[i];
        return index;
    }
};

template <typename T>
EXPR(Concat, Writable, T, T, _shape)
    std::vector<Tensor<T>> _list;
    const Shape _shape;
    const Size _size;

public:
    Concat(const std::vector<Tensor<T>> &list, const Shape &shape, Size size)
        : _list(list), _shape(shape), _size(size) {}
    WRITABLE(auto index = idx / _size / _list.size() * _size + idx % _size;
             return _list[(idx / _size) % _list.size()][index])
};

template <int P, typename Expr1>
EXPR(Power, Readable, typename Expr1::Type, P C Expr1, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Power(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator[](Size idx) const { return std::pow(_expr1[idx], P); }
};

template <typename Expr1, typename Func>
EXPR(FWhere, typename Expr1::Status, typename Expr1::Type, Expr1 C Func, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    Type _val;

public:
    FWhere(const Expr1 &expr1, Func func, Type val) : _expr1(expr1), _func(func), _val(val) {}
    WRITABLE(return _func(_expr1[idx]) ? _expr1[idx] : _val)
};

template <typename Expr1, typename Func>
EXPR(IWhere, typename Expr1::Status, typename Expr1::Type, Expr1 C Func, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    Type _val;

public:
    IWhere(const Expr1 &expr1, Func func, Type val) : _expr1(expr1), _func(func), _val(val) {}
    WRITABLE(return _func(idx) ? _expr1[idx] : _val)
};

template <typename Expr1, typename Expr2, typename Func>
EXPR(FEWhere, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Type, Expr1 C Expr2 C Func, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;

public:
    FEWhere(const Expr1 &expr1, const Expr2 & expr2, Func func)
        : _expr1(expr1), _expr2(expr2), _func(func) {}
    WRITABLE(return _func(_expr1[idx]) ? _expr1[idx] : _expr2[idx])
};

template <typename Expr1, typename Expr2, typename Func>
EXPR(IEWhere, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Type, Expr1 C Expr2 C Func, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;

public:
    IEWhere(const Expr1 &expr1, const Expr2 &expr2, Func func)
        : _expr1(expr1), _expr2(expr2), _func(func) {}
    WRITABLE(return _func(idx) ? _expr1[idx] : _expr2[idx])
};

template <typename Expr1>
EXPR(Polar, Readable, typename Expr1::Type, Expr1, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Polar(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator[](Size idx) const { return std::polar(_expr1[idx].real(), _expr1[idx].imag()); }
};

template <typename Expr1>
EXPR(Reshape, typename Expr1::Status, typename Expr1::Type, Expr1, _shape)
    typename Alias<Expr1, Status>::Type _expr1;
    const Shape _shape;

public:
    Reshape(const Expr1 &expr1, const Shape &shape) : _expr1(expr1), _shape(shape) {}
    WRITABLE(return _expr1[idx])
};

template <typename T, typename Expr1>
EXPR(Convert, Readable, T, T C Expr1, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Convert(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator[](Size idx) const { return Type(_expr1[idx]); }
};

template <typename Expr1>
EXPR(Self, typename Expr1::Status, typename Expr1::Type, Expr1, _expr1.shape())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Self(const Expr1 &expr1) : _expr1(expr1) {}
    WRITABLE(return _expr1[idx])
};

template <typename T>
EXPR(Initial, Writable, T, T, _shape)
    std::vector<T> _list;
    const Shape _shape;

public:
    Initial(const Shape &shape, const std::vector<T> &list) : _list(list), _shape(shape) {}
    WRITABLE(return _list[idx])
};

template <typename T>
EXPR(Constant, Writable, T, T, _shape)
    T _val;
    const Shape _shape;

public:
    Constant(const Shape &shape, T val) : _shape(shape), _val(val) {}
    Type operator[](Size) const { return _val; }
};

template <typename T, typename Dist>
EXPR(Distribution, Readable, T, T C Dist, _shape)
    Dist _dist;
    std::mt19937_64 _rand;
    const Shape _shape;
    Type operator[](Size) { return Type(_dist(_rand)); }

public:
    Distribution(const Shape &shape, Dist dist) : _dist(std::move(dist)), _rand(rand()), _shape(shape) {}
    Type operator[](Size idx) const { return const_cast<Distribution<Type, Dist> &>(*this)[idx]; }
};

template <typename T>
EXPR(Sequence, Readable, T, T, _shape)
    const Shape _shape;
    const Type _start, _step;

public:
    Sequence(const Shape &shape, Type start, Type step)
        : _shape(shape), _start(start), _step(step) {}
    Type operator[](Size idx) const { return _start + idx * _step; }
};

template <typename Expr1>
EXPR(ToTensor, typename Expr1::Status, typename Expr1::Type, Expr1, _shape)
    typename MatrixImpl::Alias<Expr1, Status>::Type _expr1;
    const Shape _shape;

public:
    using Major = typename Expr1::Major;
    ToTensor(const Expr1 &expr1, const Shape &shape) : _expr1(expr1), _shape(shape) {}
    WRITABLE(if constexpr (std::is_same<Major, Col>::value)
             return _expr1(idx % _shape[0], idx / _shape[0]);
             else return _expr1(idx / _shape[1], idx % _shape[1]))
};

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
// clang-format on
}

#undef C
#undef EXPR
#undef OPERATOR
#undef WRITABLE
