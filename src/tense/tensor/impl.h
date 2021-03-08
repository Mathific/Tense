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

#include <tense/tensor/base.h>

#include <memory>

#define OPERATOR(OP)                                                                                    \
    Derived &operator OP##=(Type expr2)                                                                 \
    {                                                                                                   \
        Eval::eval(*this, *this OP expr2);                                                              \
        return *this;                                                                                   \
    }                                                                                                   \
    template <typename Expr2, typename = IsExpr<Expr2>>                                                 \
    Derived &operator OP##=(const Expr2 &expr2)                                                         \
    {                                                                                                   \
        TENSE_TASSERT(shape(), ==, expr2.shape(), "operator" << #OP, "Shapes of tensors must be equal") \
        Eval::eval(*this, *this OP expr2);                                                              \
        return *this;                                                                                   \
    }

namespace Tense::TensorImpl
{
template <typename T>
class Tensor : public Base<T, Tensor<T>>
{
    struct Data
    {
        bool owner = true;
        Shape shape, stride;
        T *data = nullptr;

        ~Data()
        {
            if (data != nullptr && owner) delete[] data;
        }
        Data(const Shape &shape, const T &val) : Data(shape)
        {
            Helper::check(shape);
            auto size = Helper::elems(shape);
            data = new (std::align_val_t(TENSE_ALIGNMENT)) T[size];
            std::fill(data, data + size, val);
        }
        Data(const Shape &shape, const std::vector<T> &list) : Data(shape)
        {
            Helper::check(shape);
            auto size = Helper::elems(shape);
            TENSE_TASSERT(size, ==, list.size(), "constructor", "List size must be equal to input size")
            data = new (std::align_val_t(TENSE_ALIGNMENT)) T[size];
            std::copy(list.begin(), list.end(), data);
        }
        Data(const Shape &shape, T *data, Mode mode) : Data(shape)
        {
            this->owner = mode != Mode::Hold;
            this->data = data;

            Helper::check(shape);
            if (mode != Mode::Copy) return;
            auto size = Helper::elems(shape);
            this->data = new (std::align_val_t(TENSE_ALIGNMENT)) T[size];
            std::copy(data, data + size, this->data);
        }
        Data(const Shape &shape) : shape(shape), stride(Helper::stride(shape)) {}
    };

    using Derived = Tensor<T>;
    using DataPtr = std::shared_ptr<Data>;

    DataPtr _shared;

public:
    using Type = T;
    using Status = Writable;

    Tensor() {}
    Tensor(const Derived &other) = default;
    Derived &operator=(const Derived &other) = default;
    Tensor(const Shape &shape, const T &val = T(0)) : _shared(new Data(shape, val)) {}
    Tensor(const Shape &shape, std::vector<T> list) : _shared(new Data(shape, list)) {}
    Tensor(const Shape &shape, T *data, Mode mode = Mode::Copy) : _shared(new Data(shape, data, mode)) {}

    template <typename Expr2, typename = IsExpr<Expr2>>
    Tensor(const Expr2 &expr) : _shared(new Data(expr.shape(), T(0)))
    {
        Eval::eval(*this, expr);
    }
    Derived &_reshape(const Shape &shape)
    {
        TENSE_TASSERT(Helper::elems(_shared->shape), ==, Helper::elems(shape), "_reshape",
                      "Size of shapes must be equal")
        Helper::check(shape), _shared->shape = shape, _shared->stride = Helper::stride(shape);
    }
    Size memory() const
    {
        if (!valid()) return sizeof(DataPtr);
        return size() * sizeof(Type) + dims() * sizeof(Size) * 2 + sizeof(DataPtr) + sizeof(Data);
    }
    Type *release()
    {
        _shared->owner = false;
        auto data = _shared->data;
        _shared->data = nullptr;
        return data;
    }
    template <typename Major>
    Matrix<Major, Type> wrap(Mode mode = Mode::Hold)
    {
        auto data = mode == Mode::Own ? this->release() : this->data();
        TENSE_TASSERT(dims(), ==, 2, "wrap", "Dimension must be 2")
        return Matrix<Major, Type>(size(0), size(1), data, mode);
    }
    void reset() { _shared.reset(); }
    const Derived &eval() const { return *this; }
    Tensor<T> copy() const { return Tensor<T>(_shared->shape, const_cast<T *>(_shared->data), Mode::Copy); }

    bool valid() const { return bool(_shared); }
    Size dims() const { return _shared->shape.size(); }
    const Shape &shape() const { return _shared->shape; }
    Size size() const { return Helper::elems(_shared->shape); }
    Size size(Size index) const { return _shared->shape[index]; }

    T *data() { return _shared->data; }
    const T *data() const { return _shared->data; }
    T *begin() { return _shared->data; }
    const T *begin() const { return _shared->data; }
    T *end() { return _shared->data + Helper::elems(_shared->shape); }
    const T *end() const { return _shared->data + Helper::elems(_shared->shape); }
    T &operator[](Size index) { return _shared->data[index]; }
    const T &operator[](Size index) const { return _shared->data[index]; }

    template <typename... Args>
    T &operator()(Args &&...args)
    {
        return _shared->data[Access::item<0, sizeof...(Args)>(_shared->stride, std::forward<Args>(args)...)];
    }
    template <typename... Args>
    const T &operator()(Args &&...args) const
    {
        return _shared->data[Access::item<0, sizeof...(Args)>(_shared->stride, std::forward<Args>(args)...)];
    }

    template <typename Expr2, typename = IsExpr<Expr2>>
    Derived &operator=(const Expr2 &expr)
    {
        if (!valid() || Helper::elems(shape()) != Helper::elems(expr.shape()))
            _shared = DataPtr(new Data(expr.shape(), T(0)));
        else
            _shared->shape = expr.shape();

        Eval::eval(*this, expr);
        return *this;
    }
    Derived &operator=(const Type &expr2)
    {
        std::fill(_shared->data, _shared->data + size(), expr2);
        return *this;
    }

    OPERATOR(+)
    OPERATOR(-)
    OPERATOR(*)
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
    const auto &shape = expr.shape();
    os << "Tensor<" << TypeName<typename Expr1::Type>() << ",";
    for (Size i = 0; i < shape.size(); ++i) os << shape[i] << (i + 1 == shape.size() ? "" : ",");
    os << "> [" << std::endl << "    ";
    auto size = Helper::elems(shape);
    for (Size i = 0; i < size; ++i) os << expr[i] << (i + 1 == size ? "" : ", ");
    os << "\n]";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor)
{
    if (!tensor.valid()) return os << "Tensor<>[]";
    print(os, tensor);
    return os;
}

template <typename Expr, typename = IsExpr<Expr>>
std::ostream &operator<<(std::ostream &os, const Expr &tensor)
{
    print(os, tensor);
    return os;
}
}  // namespace Tense::TensorImpl

#undef OPERATOR
