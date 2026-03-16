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

#include <tense/vector/base.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <new>

#define OPERATOR(OP)                                                                                     \
    Derived& operator OP## = (const Type& expr2)                                                         \
    {                                                                                                    \
        Eval::eval(*this, *this OP expr2);                                                               \
        return *this;                                                                                    \
    }                                                                                                    \
    template <typename Expr2, typename = IsExpr<Expr2>>                                                  \
    Derived& operator OP## = (const Expr2& expr2)                                                        \
    {                                                                                                    \
        TENSE_VASSERT(this->size(), ==, expr2.size(), "operator" + #OP, "size of vectors must be equal") \
        Eval::eval(*this, *this OP expr2);                                                               \
        return *this;                                                                                    \
    }

namespace Tense::VectorImpl
{
template <typename T>
class Vector : public Base<T, Vector<T>>
{
    struct Data
    {
        bool owner = true;
        Size size;
        T* data = nullptr;

        ~Data()
        {
            if (data != nullptr && owner) ::operator delete[](data, std::align_val_t(TENSE_ALIGNMENT));
        }
        Data(Size size) : size(size)
        {
            TENSE_VASSERT(size, >, 0, "constructor", "Input size can't be zero")
            data = new (std::align_val_t(TENSE_ALIGNMENT)) T[size];
        }
        Data(Size size, const T& val) : Data(size)  //
        {
            std::fill(data, data + size, val);
        }
        Data(Size size, T* data, Mode mode) : owner(mode != Mode::Hold), size(size), data(data)
        {
            TENSE_VASSERT(size, >, 0, "constructor", "Input size can't be zero")
            if (mode != Mode::Copy) return;
            this->data = new (std::align_val_t(TENSE_ALIGNMENT)) T[size];
            std::copy(data, data + size, this->data);
        }
    };

    using Derived = Vector<T>;
    using DataPtr = std::shared_ptr<Data>;

    DataPtr _shared;

public:
    using Type = T;
    using Status = Writable;
    using Flag = void;

    Vector() = default;
    Vector(const Derived& other) = default;
    Vector& operator=(const Derived& other) = default;
    explicit Vector(Size size, const T& val = T(0)) : _shared(std::make_shared<Data>(size, val)) {}
    Vector(Size size, T* data, Mode mode = Mode::Copy) : _shared(std::make_shared<Data>(size, data, mode)) {}

    Vector(const std::initializer_list<T>& list)
    {
        _shared = std::make_shared<Data>(list.size());
        std::copy(list.begin(), list.end(), _shared->data);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    Vector(const Expr2& expr) : _shared(std::make_shared<Data>(expr.size()))
    {
        Eval::eval(*this, expr);
    }
    Size memory() const
    {
        if (!this->valid()) return sizeof(DataPtr);
        return this->size() * sizeof(Type) + sizeof(DataPtr) + sizeof(Data);
    }
    Derived copy() const
    {
        if (!this->valid()) return Derived();
        return Derived(this->size(), const_cast<Type*>(this->data()), Mode::Copy);
    }
    void resize(Size size)
    {
        TENSE_VASSERT(size, >, 0, "resize", "Input size can't be zero")
        if (size == this->size()) return;

        Derived target(size);
        auto S = std::min(this->size(), size);
        target.block(0, S) = this->block(0, S);
        *this = target;
    }
    Type* release()
    {
        auto data = _shared->data;
        _shared->owner = false;
        _shared->data = nullptr;
        _shared->size = 0;
        _shared = DataPtr();
        return data;
    }
    Tensor<Type> wrap(Mode mode = Mode::Hold)
    {
        Shape shape{this->size()};
        auto data = mode == Mode::Own ? this->release() : this->data();
        return Tensor<Type>(shape, data, mode);
    }
    const Derived& eval() const { return *this; }

    void reset() { _shared.reset(); }
    bool valid() const { return bool(_shared); }
    Size size() const { return _shared->size; }

    T* data() { return _shared->data; }
    const T* data() const { return _shared->data; }
    T* begin() { return _shared->data; }
    const T* begin() const { return _shared->data; }
    T* end() { return _shared->data + size(); }
    const T* end() const { return _shared->data + size(); }
    T& operator[](Size idx) { return _shared->data[idx]; }
    const T& operator[](Size idx) const { return _shared->data[idx]; }
    T& operator()(Size idx) { return _shared->data[idx]; }
    const T& operator()(Size idx) const { return _shared->data[idx]; }

    Derived& operator=(const std::initializer_list<Type>& list)
    {
        if (!this->valid() || this->size() != list.size()) _shared = std::make_shared<Data>(list.size());
        std::copy(list.begin(), list.end(), _shared->data);
        return *this;
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    Derived& operator=(const Expr2& expr)
    {
        if (!this->valid() || this->size() != expr.size()) _shared = std::make_shared<Data>(expr.size());
        Eval::eval(*this, expr);
        return *this;
    }
    Derived& operator=(const Type& expr2)
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
void print(std::ostream& os, const Expr1& expr)
{
    const Size size = expr.size();
    os << "Vector<" << Impl::TypeName<typename Expr1::Type>() << ", " << size << "> [";
    for (Size i = 0; i < size; ++i) os << expr[i] << (i == size - 1 ? "" : ", ");
    os << "]";
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const Vector<Ts...>& vector)
{
    if (!vector.valid()) return os << "Vector<>[]";
    print(os, vector);
    return os;
}
template <typename Expr, typename = IsExpr<Expr>>
std::ostream& operator<<(std::ostream& os, const Expr& vector)
{
    print(os, vector);
    return os;
}
}  // namespace Tense::VectorImpl

#undef OPERATOR
