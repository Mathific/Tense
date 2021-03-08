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

#include <tense/matrix/base.h>
#include <tense/matrix/extr.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <new>

#define OPERATOR(OP)                                                                                       \
    Derived &operator OP##=(const Type &expr2)                                                             \
    {                                                                                                      \
        Eval::eval(*this, *this OP expr2);                                                                 \
        return *this;                                                                                      \
    }                                                                                                      \
    template <typename Expr2, typename = IsExpr<Expr2>>                                                    \
    Derived &operator OP##=(const Expr2 &expr2)                                                            \
    {                                                                                                      \
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "operator" << #OP, "Rows of matrices must be equal") \
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "operator" << #OP, "Cols of matrices must be equal") \
        Eval::eval(*this, *this OP expr2);                                                                 \
        return *this;                                                                                      \
    }

namespace Tense::MatrixImpl
{
template <typename M, typename T>
class Matrix : public Base<M, T, Matrix<M, T>>
{
    struct Data
    {
        bool owner = true;
        Size rows, cols;
        T *data = nullptr;

        ~Data()
        {
            if (data != nullptr && owner) delete[] data;
        }
        Data(Size rows, Size cols, const T &val) : rows(rows), cols(cols)
        {
            TENSE_MASSERT(rows, >, 0, "constructor", "Input rows can't be zero")
            TENSE_MASSERT(cols, >, 0, "constructor", "Input cols can't be zero")
            data = new (std::align_val_t(TENSE_ALIGNMENT)) T[rows * cols];
            std::fill(data, data + rows * cols, val);
        }
        Data(Size rows, Size cols, T *data, Mode mode) : owner(mode != Mode::Hold), rows(rows), cols(cols), data(data)
        {
            TENSE_MASSERT(rows, >, 0, "constructor", "Input rows can't be zero")
            TENSE_MASSERT(cols, >, 0, "constructor", "Input cols can't be zero")
            if (mode != Mode::Copy) return;
            this->data = new (std::align_val_t(TENSE_ALIGNMENT)) T[rows * cols];
            std::copy(data, data + rows * cols, this->data);
        }
    };

    using Derived = Matrix<M, T>;
    using DataPtr = std::shared_ptr<Data>;

    DataPtr _shared;

public:
    using Type = T;
    using Major = M;
    using Status = Writable;
    using Flag = void;

    Matrix() {}
    Matrix(const Derived &other) = default;
    Derived &operator=(const Derived &other) = default;
    Matrix(Size rows, Size cols, const T &val = T(0)) : _shared(new Data(rows, cols, val)) {}
    Matrix(Size rows, Size cols, T *data, Mode mode = Mode::Copy) : _shared(new Data(rows, cols, data, mode)) {}

    Matrix(IL1D<T> list)
    {
        _shared = DataPtr(new Data(list.size(), 1, T(0)));
        std::copy(list.begin(), list.end(), _shared->data);
    }
    Matrix(IL2D<T> list)
    {
        auto rows2 = list.size(), cols2 = rows2 ? list.begin()->size() : 0;
        _shared = DataPtr(new Data(rows2, cols2, T(0)));
        Eval::assign(*this, list);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    Matrix(const Expr2 &expr) : _shared(new Data(expr.rows(), expr.cols(), T(0)))
    {
        Eval::eval(*this, expr);
    }
    Derived &_reshape(Size rows, Size cols)
    {
        TENSE_MASSERT(_shared->rows * _shared->cols, ==, rows * cols, "_reshape", "Size of shapes must be equal")
        _shared->rows = rows, _shared->cols = cols;
    }
    Size memory() const
    {
        if (!this->valid()) return sizeof(DataPtr);
        return this->size() * sizeof(Type) + sizeof(DataPtr) + sizeof(Data);
    }
    Derived copy() const
    {
        if (!this->valid()) return Derived();
        return Derived(this->rows(), this->cols(), const_cast<Type *>(this->data()), Mode::Copy);
    }
    void resize(Size rows, Size cols)
    {
        TENSE_MASSERT(rows, >, 0, "resize", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "resize", "Input cols can't be zero")
        if (rows == this->rows() && cols == this->cols()) return;

        Derived target(rows, cols);
        auto R = std::min(this->rows(), rows), C = std::min(this->cols(), cols);
        target.block(0, 0, R, C) = this->block(0, 0, R, C);
        *this = target;
    }
    Type *release()
    {
        auto data = _shared->data;
        _shared->owner = false;
        _shared->data = nullptr;
        _shared->rows = 0;
        _shared->cols = 0;
        return data;
    }
    Tensor<Type> wrap(Mode mode = Mode::Hold)
    {
        auto data = mode == Mode::Own ? this->release() : this->data();
        return Tensor<Type>({this->rows(), this->cols()}, data, mode);
    }
    const Derived &eval() const { return *this; }

    void reset() { _shared.reset(); }
    bool valid() const { return bool(_shared); }
    Size rows() const { return _shared->rows; }
    Size cols() const { return _shared->cols; }
    Size size() const { return _shared->rows * _shared->cols; }

    T *data() { return _shared->data; }
    const T *data() const { return _shared->data; }
    T *begin() { return _shared->data; }
    const T *begin() const { return _shared->data; }
    T *end() { return _shared->data + size(); }
    const T *end() const { return _shared->data + size(); }
    T &operator[](Size idx) { return _shared->data[idx]; }
    const T &operator[](Size idx) const { return _shared->data[idx]; }

    T &operator()(Size i, Size j)
    {
        if constexpr (std::is_same<Major, Col>::value)
            return _shared->data[j * _shared->rows + i];
        else
            return _shared->data[i * _shared->cols + j];
    }
    const T &operator()(Size i, Size j) const
    {
        if constexpr (std::is_same<Major, Col>::value)
            return _shared->data[j * _shared->rows + i];
        else
            return _shared->data[i * _shared->cols + j];
    }

    Derived &operator=(IL1D<Type> list)
    {
        if (!this->valid() || this->rows() != list.size() || this->cols() != 1)
            _shared = new DataPtr(Data(list.size(), 1, T(0)));
        std::copy(list.begin(), list.end(), _shared->data);
    }
    Derived &operator=(IL2D<Type> list)
    {
        auto rows2 = list.size(), cols2 = rows2 ? list.begin()->size() : 0;
        if (!this->valid() || this->rows() != rows2 || this->cols() != cols2)
            _shared = new DataPtr(Data(rows2, cols2, T(0)));
        Eval::assign(*this, list);
        return *this;
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    Derived &operator=(const Expr2 &expr)
    {
        if (!this->valid() || this->rows() != expr.rows() || this->cols() != expr.cols())
            _shared = DataPtr(new Data(expr.rows(), expr.cols(), T(0)));
        Eval::eval(*this, expr);
        return *this;
    }
    Derived &operator=(const Type &expr2)
    {
        std::fill(this->data(), this->data() + this->size(), expr2);
        return *this;
    }

    OPERATOR(+)
    OPERATOR(*)
    OPERATOR(-)
    OPERATOR(/)
    OPERATOR(%)
    OPERATOR(&)
    OPERATOR(|)
    OPERATOR(^)
    OPERATOR(<<)
    OPERATOR(>>)
};

template <typename Expr1>
void print(std::ostream &os, const Expr1 &expr)
{
    const Size rows = expr.rows(), cols = expr.cols();
    os << "Matrix<" << TypeName<typename Expr1::Type>() << "," << rows << "," << cols << "> [" << std::endl;
    std::vector<Size> width(cols, 0);

    std::ostringstream stream;
    for (Size i = 0; i < rows; ++i)
        for (Size j = 0; j < cols; ++j)
        {
            stream.str("");
            stream.clear();
            stream << expr(i, j);
            width[j] = std::max<Size>(width[j], stream.str().size());
        }

    for (Size i = 0; i < rows; ++i)
    {
        os << "    ";
        for (Size j = 0; j < cols; ++j) os << std::setw(width[j]) << expr(i, j) << (j == cols - 1 ? "" : ", ");
        os << std::endl;
    }
    os << "]";
}

template <typename... Ts>
std::ostream &operator<<(std::ostream &os, const Matrix<Ts...> &matrix)
{
    if (!matrix.valid()) return os << "Matrix<>[]";
    print(os, matrix);
    return os;
}
template <typename Expr, typename = IsExpr<Expr>>
std::ostream &operator<<(std::ostream &os, const Expr &matrix)
{
    print(os, matrix);
    return os;
}
}  // namespace Tense::MatrixImpl

#undef OPERATOR
