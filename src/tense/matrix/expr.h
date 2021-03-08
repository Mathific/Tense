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

#include <tense/matrix/struct.h>

#include <complex>
#include <random>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#define C ,

#define EXPR(NAME, STATUS, MAJOR, TYPE, ARGS, ROWS, COLS) \
    struct NAME : public Base<MAJOR, TYPE, NAME<ARGS>>    \
    {                                                     \
        using Type = TYPE;                                \
        using Major = MAJOR;                              \
        using Status = STATUS;                            \
        using Flag = void;                                \
                                                          \
        Size rows() const { return ROWS; }                \
        Size cols() const { return COLS; }                \
                                                          \
    private:                                              \
        using This = NAME<ARGS>;

#define FLAGGED(NAME, FLAG, STATUS, MAJOR, TYPE, ARGS, ROWS, COLS) \
    struct NAME : public Base<MAJOR, TYPE, NAME<ARGS>>             \
    {                                                              \
        using Type = TYPE;                                         \
        using Major = MAJOR;                                       \
        using Status = STATUS;                                     \
        using Flag = FLAG;                                         \
                                                                   \
        Size rows() const { return ROWS; }                         \
        Size cols() const { return COLS; }                         \
                                                                   \
    private:                                                       \
        using This = NAME<ARGS>;

#define SQUARE(NAME, FLAG, EXPR)                                     \
    template <typename Expr1>                                        \
    auto NAME(const Expr1 &expr1)                                    \
    {                                                                \
        auto func = [](auto i, auto j, const auto &expr1) { EXPR; }; \
        return Square<Expr1, decltype(func), FLAG>(expr1, func);     \
    }

#define OPERATOR(OP)                                                                                         \
    auto &operator OP##=(Type expr2)                                                                         \
    {                                                                                                        \
        Eval::eval(*this, *this OP expr2);                                                                   \
        return *this;                                                                                        \
    }                                                                                                        \
    template <typename _Expr, typename = IsExpr<_Expr>, typename Temp = Status, typename = IsWritable<Temp>> \
    auto &operator OP##=(const _Expr &expr)                                                                  \
    {                                                                                                        \
        TENSE_MASSERT(rows(), ==, expr.rows(), "operator" << #OP, "Rows of matrices must be equal")          \
        TENSE_MASSERT(cols(), ==, expr.cols(), "operator" << #OP, "Cols of matrices must be equal")          \
        Eval::eval(*this, *this OP expr);                                                                    \
        return *this;                                                                                        \
    }

#define WRITABLE(EXPR)                                                                                           \
    template <typename Temp = Status, typename = IsWritable<Temp>>                                               \
    Type &operator()(Size i, Size j)                                                                             \
    {                                                                                                            \
        EXPR;                                                                                                    \
    }                                                                                                            \
    Type operator()(Size i, Size j) const { EXPR; }                                                              \
                                                                                                                 \
    auto &operator=(Type val)                                                                                    \
    {                                                                                                            \
        Eval::assign(*this, val);                                                                                \
        return *this;                                                                                            \
    }                                                                                                            \
    template <typename _Expr, typename = IsExpr<_Expr>, typename Temp = Status, typename = IsWritable<Temp>>     \
    auto &operator=(const _Expr &expr)                                                                           \
    {                                                                                                            \
        TENSE_MASSERT(rows(), ==, expr.rows(), "assign", "Rows of matrices must be equal")                       \
        TENSE_MASSERT(cols(), ==, expr.cols(), "assign", "Cols of matrices must be equal")                       \
        Eval::eval(*this, expr);                                                                                 \
        return *this;                                                                                            \
    }                                                                                                            \
    auto &operator=(IL1D<Type> list)                                                                             \
    {                                                                                                            \
        TENSE_MASSERT(rows(), ==, list.size(), "operator=", "Rows of matrix and initializer list must be equal") \
        TENSE_MASSERT(cols(), ==, 1, "operator=", "Cols of matrix and initializer list must be equal")           \
        Eval::assign(*this, list);                                                                               \
        return *this;                                                                                            \
    }                                                                                                            \
    auto &operator=(IL2D<Type> list)                                                                             \
    {                                                                                                            \
        auto rows2 = list.size(), cols2 = rows2 ? list.begin()->size() : 0;                                      \
        TENSE_MASSERT(rows(), ==, rows2, "operator=", "Rows of matrix and initializer list must be equal")       \
        TENSE_MASSERT(cols(), ==, cols2, "operator=", "Cols of matrix and initializer list must be equal")       \
        Eval::assign(*this, list);                                                                               \
        return *this;                                                                                            \
    }                                                                                                            \
    auto &operator=(const This &other) { return this->operator=<This>(other); }                                  \
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

namespace Tense::MatrixImpl
{
// clang-format off
template <typename T, typename Expr1, typename Func>
EXPR(Unary, Readable, typename Expr1::Major, T, T C Expr1 C Func, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;

public:
    Unary(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func) {}
    Type operator()(Size i, Size j) const { return _func(_expr1(i, j)); }
};

template <typename T, typename Expr1, typename Expr2, typename Func>
EXPR(Binary, Readable, typename Expr1::Major, T, T C Expr1 C Expr2 C Func,
        std::max(_expr1.rows(), _expr2.rows()), std::max(_expr1.cols(), _expr2.cols()))
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const bool _i1, _i2, _j1, _j2;
    const Func _func;

public:
    Binary(const Expr1 &expr1, const Expr2 &expr2, Func func)
        : _expr1(expr1), _expr2(expr2),
          _i1(expr1.rows() >= expr2.rows()),
          _i2(expr1.rows() <= expr2.rows()),
          _j1(expr1.cols() >= expr2.cols()),
          _j2(expr1.cols() <= expr2.cols()),
          _func(func) {}
    Type operator()(Size i, Size j) const
    {
        return _func(_expr1(_i1 * i, _j1 * j), _expr2(_i2 * i, _j2 * j));
    }
};

template <typename Expr1, typename Func, typename FLAG>
FLAGGED(Square, FLAG, Readable, typename Expr1::Major, typename Expr1::Type,
        Expr1 C Func C FLAG, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;

public:
    using Expr = Expr1;
    const Expr1 &get() const { return _expr1; }

    Square(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func) {}
    Type operator()(Size i, Size j) const { return _func(i, j, _expr1); }
};

template<typename Expr1, typename Func>
EXPR(Access, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type,
        Expr1 C Func, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;

public:
    Access(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func) {}
    WRITABLE(return _func(i, j, _expr1))
};

template <typename Expr1>
EXPR(BRepeat, Readable, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows, _cols, _rows2, _cols2;

public:
    BRepeat(const Expr1 &expr1, Size rows, Size cols)
        : _expr1(expr1), _rows(rows), _cols(cols), _rows2(expr1.rows()), _cols2(expr1.cols()) {}
   Type operator()(Size i, Size j) const { return _expr1(i % _rows2, j % _cols2); }
};

template <typename Expr1>
EXPR(RRepeat, Readable, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows;

public:
    RRepeat(const Expr1 &expr1, Size rows) : _expr1(expr1), _rows(rows) {}
    Type operator()(Size i, Size j) const { return _expr1(0, j); }
};

template <typename Expr1>
EXPR(CRepeat, Readable, typename Expr1::Major, typename Expr1::Type, Expr1, _expr1.rows(), _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _cols;

public:
    CRepeat(const Expr1 &expr1, Size cols) : _expr1(expr1), _cols(cols) {}
    Type operator()(Size i, Size j) const { return _expr1(i, 0); }
};

template <typename Expr1>
EXPR(Repeat, Readable, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows, _cols;

public:
    Repeat(const Expr1 &expr1, Size rows, Size cols) : _expr1(expr1), _rows(rows), _cols(cols) {}
    Type operator()(Size i, Size j) const { return _expr1(0, 0); }
};

template <typename T, typename Expr1, typename Func>
FLAGGED(BReduce, HeavyFlag, Readable, typename Expr1::Major, T,
        T C Expr1 C Func, _expr1.rows() / _rows, _expr1.cols() / _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Type _init;
    const Size _rows, _cols;

public:
    BReduce(const Expr1 &expr1, Func func, Type init, Size rows, Size cols)
        : _expr1(expr1), _func(func), _init(init), _rows(rows), _cols(cols) {}
    Type operator()(Size i, Size j) const
    {
        Type val = _init;
        Size si = i * _rows, sj = j * _cols;

        if constexpr (std::is_same<Major, Col>::value)
            for (Size j = sj; j < sj + _cols; ++j)
                for (Size i = si; i < si + _rows; ++i) val = _func(val, _expr1(i, j));
        else
            for (Size i = si; i < si + _rows; ++i)
                for (Size j = sj; j < sj + _cols; ++j) val = _func(val, _expr1(i, j));

        return val;
    }
};

template <typename T, typename Expr1, typename Func>
FLAGGED(RReduce, HeavyFlag, Readable, typename Expr1::Major, T, T C Expr1 C Func, _expr1.rows(), 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Size _cols;
    const Type _init;

public:
    RReduce(const Expr1 &expr1, Func func, Type init)
        : _expr1(expr1), _func(func), _cols(expr1.cols()), _init(init) {}
    Type operator()(Size i, Size j) const
    {
        Type val = _init;
        for (Size j = 0; j < _cols; ++j) val = _func(val, _expr1(i, j));
        return val;
    }
};

template <typename T, typename Expr1, typename Func>
FLAGGED(CReduce, HeavyFlag, Readable, typename Expr1::Major, T, T C Expr1 C Func, 1, _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Size _rows;
    const Type _init;

public:
    CReduce(const Expr1 &expr1, Func func, Type init)
        : _expr1(expr1), _func(func), _rows(expr1.rows()), _init(init) {}
    Type operator()(Size i, Size j) const
    {
        Type val = _init;
        for (Size i = 0; i < _rows; ++i) val = _func(val, _expr1(i, j));
        return val;
    }
};

template <typename T, typename Expr1, typename Func>
FLAGGED(Reduce, HeavyFlag, Readable, typename Expr1::Major, T, T C Expr1 C Func, 1, 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Size _rows, _cols;
    const Type _init;

public:
    Reduce(const Expr1 &expr1, Func func, Type init)
        : _expr1(expr1), _func(func), _init(init), _rows(expr1.rows()), _cols(expr1.cols()) {}
    Type operator()(Size i, Size j) const
    {
        Type val = _init;

        if constexpr (std::is_same<Major, Col>::value)
            for (Size j = 0; j < _cols; ++j)
                for (Size i = 0; i < _rows; ++i) val = _func(val, _expr1(i, j));
        else
            for (Size i = 0; i < _rows; ++i)
                for (Size j = 0; j < _cols; ++j) val = _func(val, _expr1(i, j));

        return val;
    }
};

template <typename Expr1>
EXPR(SBlock, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _i, _j, _rows, _cols;

public:
    SBlock(const Expr1 &expr1, Size i, Size j, Size rows, Size cols)
        : _expr1(expr1), _i(i), _j(j), _rows(rows), _cols(cols) {}
    WRITABLE(return _expr1(_i + i, _j + j))
};

template <typename Expr1>
EXPR(SRow, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, 1, _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _i;

public:
    SRow(const Expr1 &expr1, Size i) : _expr1(expr1), _i(i) {}
    WRITABLE(return _expr1(_i, j))
};

template <typename Expr1>
EXPR(SCol, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _expr1.rows(), 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _j;

public:
    SCol(const Expr1 &expr1, Size j) : _expr1(expr1), _j(j) {}
    WRITABLE(return _expr1(i, _j))
};

template <typename Expr1>
EXPR(SElem, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, 1, 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _i, _j;

public:
    SElem(const Expr1 &expr1, Size i, Size j) : _expr1(expr1), _i(i), _j(j) {}
    WRITABLE(return _expr1(_i, _j))
};

template <typename Expr1>
EXPR(SDiag, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _size, 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _size;

public:
    SDiag(const Expr1 &expr1, Size size) : _expr1(expr1), _size(size) {}
    WRITABLE(return _expr1(i, i))
};

template <typename Expr1>
EXPR(RIndex, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    Size _rb, _rs, _cb, _cs, _rows, _cols;

public:
    RIndex(const Expr1 &expr1, Cut rows, Cut cols)
        : _expr1(expr1), _rb(rows.start), _rs(rows.step), _cb(cols.start), _cs(cols.step)
    {
        _rows = (rows.end - rows.start) / rows.step;
        _cols = (cols.end - cols.start) / cols.step;
    }
    WRITABLE(return _expr1(i * _rs + _rb, j *_cs + _cb))
};

template <typename Expr1>
EXPR(VIndex, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _rows.size(), _cols.size())
    typename Alias<Expr1, Status>::Type _expr1;
    const std::vector<Size> _rows, _cols;

public:
    VIndex(const Expr1 &expr1, const std::vector<Size> &rows, const std::vector<Size> &cols)
        : _expr1(expr1), _rows(rows), _cols(cols) {}
    WRITABLE(return _expr1(_rows[i], _cols[j]))
};

template <typename Expr1>
EXPR(RVIndex, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _cols.size())
    typename Alias<Expr1, Status>::Type _expr1;
    const std::vector<Size> _cols;
    Size _rb, _rs, _rows;

public:
    RVIndex(const Expr1 &expr1, Cut rows, const std::vector<Size> &cols)
        : _expr1(expr1), _cols(cols), _rb(rows.start), _rs(rows.step)
    {
        _rows = (rows.end - rows.start) / rows.step;
    }
    WRITABLE(return _expr1(i * _rs + _rb, _cols[j]))
};

template <typename Expr1>
EXPR(VRIndex, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _rows.size(), _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const std::vector<Size> _rows;
    Size _cb, _cs, _cols;

public:
    VRIndex(const Expr1 &expr1, const std::vector<Size> &rows, Cut cols)
        : _expr1(expr1), _rows(rows), _cb(cols.start), _cs(cols.step)
    {
        _cols = (cols.end - cols.start) / cols.step;
    }
    WRITABLE(return _expr1(_rows[i], j *_cs + _cb))
};

template <typename Expr1, typename Expr2>
EXPR(RIndirect, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type,
        Expr1 C Expr2, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;

public:
    RIndirect(const Expr1 &expr1, const Expr2 &expr2) : _expr1(expr1), _expr2(expr2) {}
    WRITABLE(return _expr1(_expr2(i, j), j))
};

template <typename Expr1, typename Expr2>
EXPR(CIndirect, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type,
        Expr1 C Expr2, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;

public:
    CIndirect(const Expr1 &expr1, const Expr2 &expr2) : _expr1(expr1), _expr2(expr2) {}
    WRITABLE(return _expr1(i, _expr2(i, j)))
};

template <typename Expr1, typename Expr2>
EXPR(RCat0, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Major, typename Expr1::Type, Expr1 C Expr2, _rows, _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Size _rows, _rows1;

public:
    RCat0(const Expr1 &expr1, const Expr2 &expr2, Size rows)
        : _expr1(expr1), _expr2(expr2), _rows(rows), _rows1(expr1.rows()) {}
    WRITABLE(return i < _rows1 ? _expr1(i, j) : _expr2(i - _rows1, j))
};

template <typename Expr1, typename Expr2>
EXPR(CCat0, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Major, typename Expr1::Type, Expr1 C Expr2, _expr1.rows(), _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Size _cols, _cols1;

public:
    CCat0(const Expr1 &expr1, const Expr2 &expr2, Size cols)
        : _expr1(expr1), _expr2(expr2), _cols(cols), _cols1(expr1.cols()) {}
    WRITABLE(return j < _cols1 ? _expr1(i, j) : _expr2(i, j - _cols1))
};

template <typename Expr1, typename Expr2>
EXPR(RCat1, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Major, typename Expr1::Type, Expr1 C Expr2, _expr1.rows() * 2, _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;

public:
    RCat1(const Expr1 &expr1, const Expr2 &expr2) : _expr1(expr1), _expr2(expr2) {}
    WRITABLE(return i & 1 ? _expr2(i / 2, j) : _expr1(i / 2, j))
};

template <typename Expr1, typename Expr2>
EXPR(CCat1, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Major, typename Expr1::Type, Expr1 C Expr2, _expr1.rows(), _expr1.cols() * 2)
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;

public:
    CCat1(const Expr1 &expr1, Expr2 expr2) : _expr1(expr1), _expr2(expr2) {}
    WRITABLE(return j & 1 ? _expr2(i, j / 2) : _expr1(i, j / 2))
};

template <typename Switch, typename Expr1>
FLAGGED(BAny, HeavyFlag, Readable, typename Expr1::Major, bool, Switch C Expr1, _expr1.rows() / _rows, _expr1.cols() / _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows, _cols;

public:
    BAny(const Expr1 &expr1, Size rows, Size cols) : _expr1(expr1), _rows(rows), _cols(cols) {}
    Type operator()(Size i, Size j) const
    {
        Size si = i * _rows, sj = j * _cols;
        if constexpr (std::is_same<Major, Col>::value)
        {
            for (Size j = sj; j < sj + _cols; ++j)
                for (Size i = si; i < si + _rows; ++i)
                    if (!Switch::value ^ static_cast<bool>(_expr1(i, j))) return Switch::value;
        }
        else
            for (Size i = si; i < si + _rows; ++i)
                for (Size j = sj; j < sj + _cols; ++j)
                    if (!Switch::value ^ static_cast<bool>(_expr1(i, j))) return Switch::value;

        return !Switch::value;
    }
};

template <typename Switch, typename Expr1>
FLAGGED(RAny, HeavyFlag, Readable, typename Expr1::Major, bool, Switch C Expr1, _expr1.rows(), 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _cols;

public:
    RAny(const Expr1 &expr1) : _expr1(expr1), _cols(expr1.cols()) {}
    Type operator()(Size i, Size j) const
    {
        for (Size j = 0; j < _cols; ++j)
            if (!Switch::value ^ static_cast<bool>(_expr1(i, j))) return Switch::value;
        return !Switch::value;
    }
};

template <typename Switch, typename Expr1>
FLAGGED(CAny, HeavyFlag, Readable, typename Expr1::Major, bool, Switch C Expr1, 1, _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows;

public:
    CAny(const Expr1 &expr1) : _expr1(expr1), _rows(expr1.rows()) {}
    Type operator()(Size i, Size j) const
    {
        for (Size i = 0; i < _rows; ++i)
            if (!Switch::value ^ static_cast<bool>(_expr1(i, j))) return Switch::value;
        return !Switch::value;
    }
};

template <typename Switch, typename Expr1>
FLAGGED(Any, HeavyFlag, Readable, typename Expr1::Major, bool, Switch C Expr1, 1, 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows, _cols;

public:
    Any(const Expr1 &expr1) : _expr1(expr1), _rows(expr1.rows()), _cols(expr1.cols()) {}
    Type operator()(Size i, Size j) const
    {
        if constexpr (std::is_same<Major, Col>::value)
        {
            for (Size j = 0; j < _cols; ++j)
                for (Size i = 0; i < _rows; ++i)
                    if (!Switch::value ^ static_cast<bool>(_expr1(i, j))) return Switch::value;
        }
        else
            for (Size i = 0; i < _rows; ++i)
                for (Size j = 0; j < _cols; ++j)
                    if (!Switch::value ^ static_cast<bool>(_expr1(i, j))) return Switch::value;

        return !Switch::value;
    }
};

template <typename Expr1, typename Func>
FLAGGED(RRedux, HeavyFlag, Readable, typename Expr1::Major, Size, Expr1 C Func, _expr1.rows(), 1)
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Size _cols;

public:
    RRedux(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func), _cols(expr1.cols()) {}
    Type operator()(Size i, Size j) const
    {
        Size index = 0;
        typename Expr1::Type val = _expr1(i, 0);
        for (Size j = 0; j < _cols; ++j)
            if (_func(_expr1(i, j), val)) index = j, val = _expr1(i, j);
        return index;
    }
};

template <typename Expr1, typename Func>
FLAGGED(CRedux, HeavyFlag, Readable, typename Expr1::Major, Size, Expr1 C Func, 1, _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    const Size _rows;

public:
    CRedux(const Expr1 &expr1, Func func) : _expr1(expr1), _func(func), _rows(expr1.rows()) {}
    Type operator()(Size i, Size j) const
    {
        Size index = 0;
        typename Expr1::Type val = _expr1(0, j);
        for (Size i = 0; i < _rows; ++i)
            if (_func(_expr1(i, j), val)) index = i, val = _expr1(i, j);
        return index;
    }
};

template <typename Expr1>
EXPR(Diagonal, Readable, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows, _cols;

public:
    Diagonal(const Expr1 &expr1, Size rows, Size cols) : _expr1(expr1), _rows(rows), _cols(cols) {}
    Type operator()(Size i, Size j) const { return i == j ? _expr1(i, 0) : 0; }
};

template <int P, typename Expr1>
EXPR(Power, Readable, typename Expr1::Major, typename Expr1::Type, P C Expr1, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Power(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator()(Size i, Size j) const { return std::pow(_expr1(i, j), P); }
};

template <typename Expr1, typename Func>
EXPR(FWhere, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type,
        Expr1 C Func, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    Type _val;

public:
    FWhere(const Expr1 &expr1, Func func, Type val) : _expr1(expr1), _func(func), _val(val) {}
    WRITABLE(return _func(_expr1(i, j)) ? _expr1(i, j) : _val)
};

template <typename Expr1, typename Func>
EXPR(IWhere, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type,
        Expr1 C Func, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    const Func _func;
    Type _val;

public:
    IWhere(const Expr1 &expr1, Func func, Type val) : _expr1(expr1), _func(func), _val(val) {}
    WRITABLE(return _func(i, j) ? _expr1(i, j) : _val)
};

template <typename Expr1, typename Expr2, typename Func>
EXPR(FEWhere, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Major, typename Expr1::Type, Expr1 C Expr2 C Func, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;

public:
    FEWhere(const Expr1 &expr1, const Expr2 &expr2, Func func) : _expr1(expr1), _expr2(expr2), _func(func) {}
    WRITABLE(return _func(_expr1(i, j)) ? _expr1(i, j) : _expr2(i, j))
};

template <typename Expr1, typename Expr2, typename Func>
EXPR(IEWhere, typename SumWritable<typename Expr1::Status C typename Expr2::Status>::Type,
        typename Expr1::Major, typename Expr1::Type, Expr1 C Expr2 C Func, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;
    typename Alias<Expr2, Status>::Type _expr2;
    const Func _func;

public:
    IEWhere(const Expr1 &expr1,  const Expr1 &expr2, Func func) : _expr1(expr1), _expr2(expr2), _func(func) {}
    WRITABLE(return _func(i, j) ? _expr1(i, j) : _expr2(i, j))
};

template <typename Expr1>
EXPR(Polar, Readable, typename Expr1::Major, typename Expr1::Type, Expr1, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Polar(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator()(Size i, Size j) const { return std::polar(_expr1(i, j).real(), _expr1(i, j).imag()); }
};

template <typename Expr1>
FLAGGED(Reshape, HeavyFlag, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _rows, _cols)
    typename Alias<Expr1, Status>::Type _expr1;
    const Size _rows, _cols, _cols2;

public:
    Reshape(const Expr1 &expr1, Size rows, Size cols)
        : _expr1(expr1), _rows(rows), _cols(cols), _cols2(_expr1.cols()) {}
    WRITABLE(auto idx = i * _cols + j; return _expr1(idx / _cols2, idx % _cols2))
};

template <typename T, typename Expr1>
EXPR(Convert, Readable, typename Expr1::Major, T, T C Expr1, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Convert(const Expr1 &expr1) : _expr1(expr1) {}
    Type operator()(Size i, Size j) const { return Type(_expr1(i, j)); }
};

template <typename Expr1>
EXPR(Self, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type, Expr1, _expr1.rows(), _expr1.cols())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Self(const Expr1 &expr1) : _expr1(expr1) {}
    WRITABLE(return _expr1(i, j))
};

template <typename M, typename T>
EXPR(Constant, Writable, M, T, M C T, _rows, _cols)
    T _val;
    const Size _rows, _cols;

public:
    Constant(Size rows, Size cols, T val)
        : _val(val), _rows(rows), _cols(cols) {}
    WRITABLE(return _val)
};

template <typename M, typename T>
EXPR(Eye, Readable, M, T, M C T, _rows, _cols)
    const Size _rows, _cols;

public:
    Eye(Size rows, Size cols) : _rows(rows), _cols(cols) {}
    Type operator()(Size i, Size j) const { return (i == j) ? 1 : 0; }
};

template <typename M, typename T, typename Dist>
EXPR(Distribution, Readable, M, T, M C T C Dist, _rows, _cols)
    Dist _dist;
    std::mt19937_64 _rand;
    const Size _rows, _cols;
    Type operator()(Size i, Size j) { return Type(_dist(_rand)); }

public:
    Distribution(Size rows, Size cols, Dist dist) : _dist(std::move(dist)), _rand(rand()), _rows(rows), _cols(cols) {}
    Type operator()(Size i, Size j) const { return const_cast<Distribution<M, Type, Dist> &>(*this)(i, j); }
};

template <typename M, typename T>
EXPR(Sequence, Readable, M, T, M C T, _rows, _cols)
    const Size _rows, _cols;
    const Type _start, _step;

public:
    Sequence(Size rows, Size cols, Type start, Type step) : _rows(rows), _cols(cols), _start(start), _step(step) {}
    Type operator()(Size i, Size j) const { return _start + (i * _cols + j) * _step; }
};

template <typename M, typename T>
EXPR(Strided, Writable, M, T, M C T, _rows, _cols)
    Type *_data;
    const Size _rows, _cols, _rstr, _cstr;

public:
    Strided(Size rows, Size cols, Type *data, Size rstride, Size cstride)
        : _data(data), _rows(rows), _cols(cols), _rstr(rstride), _cstr(cstride) {}
    WRITABLE(return _data[i * _rstr + j * _cstr])
};

template <typename M, typename T, Size Rows, Size Cols>
FLAGGED(Static, StaticFlag, Writable, M, T, M C T C Rows C Cols, Rows, Cols)
    T _data[Rows * Cols];

public:
    Static(T value) { std::fill(_data, _data + Rows * Cols, value); }
    Static(const std::vector<T> &list) { std::copy(list.begin(), list.end(), _data); }
    WRITABLE(if constexpr (std::is_same<Major, Col>::value) return _data[j * Rows + i];
             else return _data[i * Cols + j];)
};

template <typename Expr1, typename M>
EXPR(ToMatrix, typename Expr1::Status, M, typename Expr1::Type, Expr1 C M, _rows, _cols)
    typename TensorImpl::Alias<Expr1, Status>::Type _expr1;
    const Size _rows, _cols;
    // TODO maybe we can use operator() of tensor

public:
    ToMatrix(const Expr1 &expr1, Size rows, Size cols) : _expr1(expr1), _rows(rows), _cols(cols) {}
    WRITABLE(if constexpr (std::is_same<Major, Col>::value)
             return _expr1[j * _rows + i]; else return _expr1[i * _cols + j])
};

template<typename Expr1>
FLAGGED(Transpose, TransFlag, typename Expr1::Status, typename Expr1::Major, typename Expr1::Type,
        Expr1, _expr1.cols(), _expr1.rows())
    typename Alias<Expr1, Status>::Type _expr1;

public:
    Transpose(const Expr1 &expr1) : _expr1(expr1) {}
    const Expr1 &get() const { return _expr1; }
    WRITABLE(return _expr1(j, i))
};

template<typename T>
auto conjugate(const T &item)
{
    return item;
}
template<typename T>
auto conjugate(const std::complex<T> &item)
{
    return std::conj(item);
}

SQUARE(SDiagonal, void, return i == j ? expr1(i, i) : 0)
SQUARE(ZDiagonal, void, return i == j ? 0 : expr1(i, j))
SQUARE(ODiagonal, void, return i == j ? 1 : expr1(i, j))
SQUARE(Upper, UpperFlag, return i <= j ? expr1(i, j) : 0)
SQUARE(Lower, LowerFlag, return i >= j ? expr1(i, j) : 0)
SQUARE(OUpper, OUpperFlag, return i < j ? expr1(i, j) : (i == j ? 1 : 0))
SQUARE(OLower, OLowerFlag, return i > j ? expr1(i, j) : (i == j ? 1 : 0))
SQUARE(ZUpper, void, return i < j ? expr1(i, j) : 0)
SQUARE(ZLower, void, return i > j ? expr1(i, j) : 0)
SQUARE(USymmetric, USymmFlag, return i <= j ? expr1(i, j) : expr1(j, i))
SQUARE(LSymmetric, LSymmFlag, return i >= j ? expr1(i, j) : expr1(j, i))
SQUARE(UHermitian, UHermFlag, return i <= j ? expr1(i, j) : conjugate(expr1(j, i)))
SQUARE(LHermitian, LHermFlag, return i >= j ? expr1(i, j) : conjugate(expr1(j, i)))
SQUARE(Conjugate, ConjFlag, return conjugate(expr1(i, j)))

template<typename Expr1>
auto RMaxIdx(const Expr1 &expr1)
{
    auto func = [](auto val1, auto val2) { return val1 > val2; };
    return RRedux<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto RMinIdx(const Expr1 &expr1)
{
    auto func = [](auto val1, auto val2) { return val1 < val2; };
    return RRedux<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto CMaxIdx(const Expr1 &expr1)
{
    auto func = [](auto val1, auto val2) { return val1 > val2; };
    return CRedux<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto CMinIdx(const Expr1 &expr1)
{
    auto func = [](auto val1, auto val2) { return val1 < val2; };
    return CRedux<Expr1, decltype(func)>(expr1, func);
}

template<typename Expr1>
auto RTurn(const Expr1 &expr1, Size _i)
{
    Size rows = expr1.rows();
    auto func = [rows, _i](auto i, auto j, const auto &expr) { return expr((i + _i) % rows, j); };
    return Access<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto CTurn(const Expr1 &expr1, Size _j)
{
    Size cols = expr1.cols();
    auto func = [cols, _j](auto i, auto j, const auto &expr) { return expr(i, (j + _j) % cols); };
    return Access<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto Turn(const Expr1 &expr1, Size _i, Size _j)
{
    Size rows = expr1.rows(), cols = expr1.cols();
    auto func = [rows, cols, _i, _j](auto i, auto j, const auto &expr)
                { return expr((i + _i) % rows, (j + _j) % cols); };
    return Access<Expr1, decltype(func)>(expr1, func);
}

template <typename Expr1>
auto BFlip(const Expr1 &expr1, Size rows, Size cols)
{
    auto func = [rows, cols](auto i, auto j, const auto &expr)
                { return expr(i + rows - 1 - 2 * (i % rows), j + cols - 1 - 2 * (j % cols)); };
    return Access<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto RFlip(const Expr1 &expr1)
{
    Size cols = expr1.cols() - 1;
    auto func = [cols](auto i, auto j, const auto &expr) { return expr(i, cols - j); };
    return Access<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto CFlip(const Expr1 &expr1)
{
    Size rows = expr1.rows() - 1;
    auto func = [rows](auto i, auto j, const auto &expr) { return expr(rows - i, j); };
    return Access<Expr1, decltype(func)>(expr1, func);
}
template<typename Expr1>
auto Flip(const Expr1 &expr1)
{
    Size rows = expr1.rows() - 1, cols = expr1.cols() - 1;
    auto func = [rows, cols](auto i, auto j, const auto &expr) { return expr(rows - i, cols - j); };
    return Access<Expr1, decltype(func)>(expr1, func);
}
// clang-format on
}  // namespace Tense::MatrixImpl

#undef C
#undef EXPR
#undef FLAGGED
#undef SQUARE
#undef OPERATOR
#undef WRITABLE
