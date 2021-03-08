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

#include <tense/matrix/expr.h>
#include <tense/matrix/extr.h>

#include <limits>
#include <random>

#define _USE_MATH_DEFINES
#include <cmath>

#define _UNARY0(NAME, TYPE, FUNC)                           \
    auto NAME() const                                       \
    {                                                       \
        return unary<TYPE>([](auto val1) { return FUNC; }); \
    }

#define _UNARY1(NAME, TYPE, VTYPE, FUNC)                        \
    auto NAME(VTYPE val2) const                                 \
    {                                                           \
        return unary<TYPE>([val2](auto val1) { return FUNC; }); \
    }

#define _BINARY(NAME, TYPE, FUNC)                                                     \
    template <typename Expr2, typename = IsExpr<Expr2>>                               \
    auto NAME(const Expr2 &expr2) const                                               \
    {                                                                                 \
        return binary<TYPE, Expr2>(expr2, [](auto val1, auto val2) { return FUNC; }); \
    }

#define _REDUCE0(NAME, TYPE, INIT, FUNC)                                                   \
    auto b##NAME(Size rows, Size cols) const                                               \
    {                                                                                      \
        return breduce<TYPE>(rows, cols, INIT, [](auto val1, auto val2) { return FUNC; }); \
    }                                                                                      \
    auto r##NAME() const                                                                   \
    {                                                                                      \
        return rreduce<TYPE>(INIT, [](auto val1, auto val2) { return FUNC; });             \
    }                                                                                      \
    auto c##NAME() const                                                                   \
    {                                                                                      \
        return creduce<TYPE>(INIT, [](auto val1, auto val2) { return FUNC; });             \
    }                                                                                      \
    auto NAME() const                                                                      \
    {                                                                                      \
        return reduce<TYPE>(INIT, [](auto val1, auto val2) { return FUNC; });              \
    }

#define _REDUCE1(NAME, TYPE, VTYPE, INIT, FUNC)                                                \
    auto b##NAME(Size rows, Size cols, VTYPE val3) const                                       \
    {                                                                                          \
        return breduce<TYPE>(rows, cols, INIT, [val3](auto val1, auto val2) { return FUNC; }); \
    }                                                                                          \
    auto r##NAME(VTYPE val3) const                                                             \
    {                                                                                          \
        return rreduce<TYPE>(INIT, [val3](auto val1, auto val2) { return FUNC; });             \
    }                                                                                          \
    auto c##NAME(VTYPE val3) const                                                             \
    {                                                                                          \
        return creduce<TYPE>(INIT, [val3](auto val1, auto val2) { return FUNC; });             \
    }                                                                                          \
    auto NAME(VTYPE val3) const                                                                \
    {                                                                                          \
        return reduce<TYPE>(INIT, [val3](auto val1, auto val2) { return FUNC; });              \
    }

#define SQUARE(NAME, EXPR) \
    auto NAME() const { return EXPR(derived()); }

#define OPERATOR0(NAME, OP) \
    auto operator OP() const { return derived().NAME(); }

#define OPERATOR1(NAME, RNAME, OP)                                       \
    template <typename Expr2, typename = IsExpr<Expr2>>                  \
    auto operator OP(const Expr2 &expr2) const                           \
    {                                                                    \
        return derived().NAME(expr2);                                    \
    }                                                                    \
    auto operator OP(Type expr2) const { return derived().NAME(expr2); } \
    friend auto operator OP(Type expr2, const Derived &expr1) { return expr1.RNAME(expr2); }

#define UNARY0(NAME, FUNC) _UNARY0(NAME, typename Derived::Type, FUNC)
#define UNARY1(NAME, TYPE, FUNC) _UNARY1(NAME, typename Derived::Type, TYPE, FUNC)
#define BINARY(NAME, FUNC) _BINARY(NAME, typename Derived::Type, FUNC)
#define REDUCE0(NAME, INIT, FUNC) _REDUCE0(NAME, typename Derived::Type, INIT, FUNC)
#define REDUCE1(NAME, TYPE, INIT, FUNC) _REDUCE1(NAME, TYPE, typename Derived::Type, INIT, FUNC)

namespace Tense::MatrixImpl
{
template <typename Major, typename Type, typename Derived>
class Base : Expr
{
    const Derived &derived() const { return *static_cast<const Derived *>(this); }
    Size rows() const { return static_cast<const Derived *>(this)->rows(); }
    Size cols() const { return static_cast<const Derived *>(this)->cols(); }

    auto _brepeat(Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, >, 0, "brepeat", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "brepeat", "Input cols can't be zero")
        return BRepeat<Derived>(derived(), rows * this->rows(), cols * this->cols());
    }
    auto _rrepeat(Size rows) const
    {
        TENSE_MASSERT(rows, >, 0, "rrepeat", "Input rows can't be zero")
        TENSE_MASSERT(this->rows(), ==, 1, "rrepeat", "Matrix rows must be 1")
        return RRepeat<Derived>(derived(), rows);
    }
    auto _crepeat(Size cols) const
    {
        TENSE_MASSERT(cols, >, 0, "crepeat", "Input cols can't be zero")
        TENSE_MASSERT(this->cols(), ==, 1, "crepeat", "Matrix cols must be 1")
        return CRepeat<Derived>(derived(), cols);
    }
    auto _repeat(Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, >, 0, "repeat", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "repeat", "Input cols can't be zero")
        TENSE_MASSERT(this->rows(), ==, 1, "repeat", "Matrix rows must be 1")
        TENSE_MASSERT(this->cols(), ==, 1, "repeat", "Matrix cols must be 1")
        return Repeat<Derived>(derived(), rows, cols);
    }

public:
    template <typename T, typename Func>
    auto unary(Func func) const
    {
        return Unary<T, Derived, Func>(derived(), func);
    }
    template <typename T, typename Expr2, typename Func, typename = IsExpr<Expr2>>
    auto binary(const Expr2 &expr2, Func func) const
    {
        if (this->rows() != 1 && expr2.rows() != 1)
            TENSE_MASSERT(this->rows(), ==, expr2.rows(), "binary", "Rows of matrices must be equal")
        if (this->cols() != 1 && expr2.cols() != 1)
            TENSE_MASSERT(this->cols(), ==, expr2.cols(), "binary", "Cols of matrices must be equal")

        if constexpr (IsHeavy<Derived>::value && IsHeavy<Expr2>::value)
            return eval().template binary<T>(expr2.eval(), func);
        else if constexpr (IsHeavy<Derived>::value)
            return eval().template binary<T>(expr2, func);
        else if constexpr (IsHeavy<Expr2>::value)
            return binary<T>(expr2.eval(), func);
        else
            return Binary<T, Derived, Expr2, Func>(derived(), expr2, func);
    }

    auto brepeat(Size rows, Size cols) const
    {
        if constexpr (IsHeavy<Derived>::value)
            return derived().eval()._brepeat(rows, cols);
        else
            return derived()._brepeat(rows, cols);
    }
    auto rrepeat(Size rows) const
    {
        if constexpr (IsHeavy<Derived>::value)
            return derived().eval()._rrepeat(rows);
        else
            return derived()._rrepeat(rows);
    }
    auto crepeat(Size cols) const
    {
        if constexpr (IsHeavy<Derived>::value)
            return derived().eval()._crepeat(cols);
        else
            return derived()._crepeat(cols);
    }
    auto repeat(Size rows, Size cols) const
    {
        if constexpr (IsHeavy<Derived>::value)
            return derived().eval()._repeat(rows, cols);
        else
            return derived()._repeat(rows, cols);
    }

    template <typename T, typename Func>
    auto breduce(Size rows, Size cols, T init, Func func) const
    {
        TENSE_MASSERT(rows, >, 0, "breduce", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "breduce", "Input cols can't be zero")
        TENSE_MASSERT(this->rows() % rows, ==, 0, "breduce", "Matrix rows must be divisable by input rows")
        TENSE_MASSERT(this->cols() % cols, ==, 0, "breduce", "Matrix cols must be divisable by input cols")
        return BReduce<T, Derived, Func>(derived(), func, init, rows, cols);
    }
    template <typename T, typename Func>
    auto rreduce(T init, Func func) const
    {
        return RReduce<T, Derived, Func>(derived(), func, init);
    }
    template <typename T, typename Func>
    auto creduce(T init, Func func) const
    {
        return CReduce<T, Derived, Func>(derived(), func, init);
    }
    template <typename T, typename Func>
    auto reduce(T init, Func func) const
    {
        return Reduce<T, Derived, Func>(derived(), func, init).item();
    }

    auto block(Size i, Size j, Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, >, 0, "block", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "block", "Input cols can't be zero")
        TENSE_MASSERT(i + rows, <=, this->rows(), "block", "Block bounds can't be out of range")
        TENSE_MASSERT(j + cols, <=, this->cols(), "block", "Block bounds can't be out of range")
        return SBlock<Derived>(derived(), i, j, rows, cols);
    }
    auto row(Size row) const
    {
        TENSE_MASSERT(row, <, this->rows(), "row", "Input row can't be out of range")
        return SRow<Derived>(derived(), row);
    }
    auto col(Size col) const
    {
        TENSE_MASSERT(col, <, this->cols(), "col", "Input col can't be out of range")
        return SCol<Derived>(derived(), col);
    }
    auto elem(Size row, Size col) const
    {
        TENSE_MASSERT(row, <, this->rows(), "elem", "Input row can't be out of range")
        TENSE_MASSERT(col, <, this->cols(), "elem", "Input col can't be out of range")
        return SElem<Derived>(derived(), row, col);
    }
    auto diag() const { return SDiag<Derived>(derived(), std::min(this->rows(), this->cols())); }

    auto index(Cut rows, Cut cols) const
    {
        if (rows.step == 0) rows = {this->rows()};
        if (cols.step == 0) cols = {this->cols()};
        TENSE_MASSERT(rows.start, <=, this->rows(), "index", "Input row start can't be out of range")
        TENSE_MASSERT(rows.end, <=, this->rows(), "index", "Input row end can't be out of range")
        TENSE_MASSERT(cols.start, <=, this->cols(), "index", "Input col start can't be out of range")
        TENSE_MASSERT(cols.end, <=, this->cols(), "index", "Input col end can't be out of range")
        return RIndex<Derived>(derived(), rows, cols);
    }
    auto index(Cut rows, const std::vector<Size> &cols) const
    {
        if (rows.step == 0) rows = {this->rows()};
        TENSE_MASSERT(rows.start, <=, this->rows(), "index", "Input row start can't be out of range")
        TENSE_MASSERT(rows.end, <=, this->rows(), "index", "Input row end can't be out of range")
        return RVIndex<Derived>(derived(), rows, cols);
    }
    auto index(const std::vector<Size> &rows, Cut cols) const
    {
        if (cols.step == 0) cols = {this->cols()};
        TENSE_MASSERT(cols.start, <=, this->cols(), "index", "Input col start can't be out of range")
        TENSE_MASSERT(cols.end, <=, this->cols(), "index", "Input col end can't be out of range")
        return VRIndex<Derived>(derived(), rows, cols);
    }
    auto index(const std::vector<Size> &rows, const std::vector<Size> &cols) const
    {
        return VIndex<Derived>(derived(), rows, cols);
    }
    auto index(const std::initializer_list<Size> &rows, const std::initializer_list<Size> &cols) const
    {
        return index(std::vector<Size>(rows), std::vector<Size>(cols));
    }
    auto index(const Cut &rows, const std::initializer_list<Size> &cols) const
    {
        return index(rows, std::vector<Size>(cols));
    }
    auto index(const std::initializer_list<Size> &rows, const Cut &cols) const
    {
        return index(std::vector<Size>(rows), cols);
    }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto rindirect(const Expr2 &expr2)
    {
        TENSE_MASSERT(rows(), ==, expr2.rows(), "rindirect", "Rows of matrices must be equal")
        TENSE_MASSERT(cols(), ==, expr2.cols(), "rindirect", "Cols of matrices must be equal")
        return RIndirect<Derived, Expr2>(derived(), expr2);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto cindirect(const Expr2 &expr2)
    {
        TENSE_MASSERT(rows(), ==, expr2.rows(), "cindirect", "Rows of matrices must be equal")
        TENSE_MASSERT(cols(), ==, expr2.cols(), "cindirect", "Cols of matrices must be equal")
        return CIndirect<Derived, Expr2>(derived(), expr2);
    }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto rcat0(const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "rcat", "Cols of matrices must be equal")
        return RCat0<Derived, Expr2>(derived(), expr2, this->rows() + expr2.rows());
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto ccat0(const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "ccat", "Rows of matrices must be equal")
        return CCat0<Derived, Expr2>(derived(), expr2, this->cols() + expr2.cols());
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto rcat1(const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "ircat", "Rows of matrices must be equal")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "ircat", "Cols of matrices must be equal")
        return RCat1<Derived, Expr2>(derived(), expr2);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto ccat1(const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "iccat", "Rows of matrices must be equal")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "iccat", "Cols of matrices must be equal")
        return CCat1<Derived, Expr2>(derived(), expr2);
    }

    auto rturn(Size rows) const
    {
        TENSE_MASSERT(rows, <=, this->rows(), "rturn", "Input rows can't be higher than matrix rows")
        return RTurn<Derived>(derived(), this->rows() - rows);
    }
    auto cturn(Size cols) const
    {
        TENSE_MASSERT(cols, <=, this->cols(), "cturn", "Input cols can't be higher than matrix cols")
        return CTurn<Derived>(derived(), this->cols() - cols);
    }
    auto turn(Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, <=, this->rows(), "turn", "Input rows can't be higher than matrix rows")
        TENSE_MASSERT(cols, <=, this->cols(), "turn", "Input cols can't be higher than matrix cols")
        return Turn<Derived>(derived(), this->rows() - rows, this->cols() - cols);
    }

    auto rminidx() const { return RMinIdx<Derived>(derived()); }
    auto cminidx() const { return CMinIdx<Derived>(derived()); }
    auto rmaxidx() const { return RMaxIdx<Derived>(derived()); }
    auto cmaxidx() const { return CMaxIdx<Derived>(derived()); }

    auto ball(Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, >, 0, "ball", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "ball", "Input cols can't be zero")
        TENSE_MASSERT(this->rows() % rows, ==, 0, "ball", "Matrix rows must be divisable by input rows")
        TENSE_MASSERT(this->cols() % cols, ==, 0, "ball", "Matrix cols must be divisable bo input cols")
        return BAny<std::false_type, Derived>(derived(), rows, cols);
    }
    auto rall() const { return RAny<std::false_type, Derived>(derived()); }
    auto call() const { return CAny<std::false_type, Derived>(derived()); }
    auto all() const { return Any<std::false_type, Derived>(derived()).item(); }

    auto bany(Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, >, 0, "bany", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "bany", "Input cols can't be zero")
        TENSE_MASSERT(this->rows() % rows, ==, 0, "bany", "Matrix rows must be divisable by input rows")
        TENSE_MASSERT(this->cols() % cols, ==, 0, "bany", "Matrix cols must be divisable bo input cols")
        return BAny<std::true_type, Derived>(derived(), rows, cols);
    }
    auto rany() const { return RAny<std::true_type, Derived>(derived()); }
    auto cany() const { return CAny<std::true_type, Derived>(derived()); }
    auto any() const { return Any<std::true_type, Derived>(derived()).item(); }

    auto bflip(Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, >, 0, "bflip", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "bflip", "Input cols can't be zero")
        TENSE_MASSERT(this->rows() % rows, ==, 0, "bflip", "Matrix rows must be divisable by input rows")
        TENSE_MASSERT(this->cols() % cols, ==, 0, "bflip", "Matrix cols must be divisable bo input cols")
        return BFlip<Derived>(derived(), rows, cols);
    }
    auto rflip() const { return RFlip<Derived>(derived()); }
    auto cflip() const { return CFlip<Derived>(derived()); }
    auto flip() const { return Flip<Derived>(derived()); }

    template <int P>
    auto pow() const
    {
        return Power<P, Derived>(derived());
    }
    auto asdiag() const
    {
        TENSE_MASSERT(this->cols(), ==, 1, "diagmat", "Cols must be 1")
        return Diagonal<Derived>(derived(), this->rows(), this->rows());
    }
    auto asdiag(Size rows, Size cols) const
    {
        TENSE_MASSERT(rows, >, 0, "diagmat", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "diagmat", "Input cols can't be zero")
        TENSE_MASSERT(this->cols(), ==, 1, "diagmat", "Cols must be 1")
        return Diagonal<Derived>(derived(), rows, cols);
    }
    auto reshape(Size rows, Size cols) const
    {
        auto size = this->rows() * this->cols();
        if (rows == 0) rows = size / cols;
        if (cols == 0) cols = size / rows;
        TENSE_MASSERT(size, ==, rows * cols, "reshape", "New size must be equal to old size")
        return Reshape<Derived>(derived(), rows, cols);
    }
    template <typename T>
    auto type() const
    {
        if constexpr (IsComplex<Type>::value && !IsComplex<T>::value)
            return derived().real().template type<T>();
        else
            return Convert<T, Derived>(derived());
    }

    template <typename Func>
    auto where(Func func, Type val = Type(0)) const
    {
        return FWhere<Derived, Func>(derived(), func, val);
    }
    template <typename Func>
    auto iwhere(Func func, Type val = Type(0)) const
    {
        return IWhere<Derived, Func>(derived(), func, val);
    }
    template <typename Func, typename Expr2, typename = IsExpr<Expr2>>
    auto where(Func func, const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "where", "Rows of matrices must be equal")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "where", "Rows of matrices must be equal")
        return FEWhere<Derived, Expr2, Func>(derived(), expr2, func);
    }
    template <typename Func, typename Expr2, typename = IsExpr<Expr2>>
    auto iwhere(Func func, const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "iwhere", "Rows of matrices must be equal")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "iwhere", "Rows of matrices must be equal")
        return IEWhere<Derived, Expr2, Func>(derived(), expr2, func);
    }

    auto uherm() const
    {
        static_assert(IsComplex<Type>::value, "Type must be complex in matrix::uherm.");
        TENSE_MASSERT(this->rows(), ==, this->cols(), "uherm", "Matrix must be square")
        return UHermitian<Derived>(derived());
    }
    auto lherm() const
    {
        static_assert(IsComplex<Type>::value, "Type must be complex in matrix::lherm.");
        TENSE_MASSERT(this->rows(), ==, this->cols(), "lherm", "Matrix must be square")
        return LHermitian<Derived>(derived());
    }
    auto polar() const { return Polar<Derived>(derived()); }
    auto conj() const { return Conjugate<Derived>(derived()); }
    auto trans() const { return Transpose<Derived>(derived()); }

    auto expr() const { return Self<Derived>(derived()); }
    auto eval() const { return Matrix<Major, Type>(derived()); }
    auto copy() const { return eval(); }
    Type item() const { return derived()(0, 0); }
    Shape shape() const { return {rows(), cols()}; }
    auto tensor() const { return TensorImpl::ToTensor<Derived>(derived(), {derived().rows(), derived().cols()}); }

    ////////////////////////////////////////////////////////////////////////////

    auto topleft(Size rows, Size cols) { return block(0, 0, rows, cols); }
    auto topright(Size rows, Size cols) { return block(0, this->cols() - cols, rows, cols); }
    auto bottomleft(Size rows, Size cols) { return block(this->rows() - rows, 0, rows, cols); }
    auto bottomright(Size rows, Size cols) { return block(this->rows() - rows, this->cols() - cols, rows, cols); }

    template <Size P>
    auto bnorm(Size rows, Size cols) const
    {
        static_assert(P > 0, "P can't be zero in Matrix::bnorm");

        if constexpr (P == Inf)
            return derived().abs().bmax(rows, cols);
        else if constexpr (P == 1)
            return derived().abs().bsum(rows, cols);
        else if constexpr (P == 2)
            return derived().abs().square().bsum(rows, cols).sqrt();
        else
            return derived().abs().template pow<P>().bsum(rows, cols).pow(1.f / P);
    }
    template <Size P>
    auto rnorm() const
    {
        static_assert(P > 0, "P can't be zero in Matrix::rnorm");

        if constexpr (P == Inf)
            return derived().abs().rmax();
        else if constexpr (P == 1)
            return derived().abs().rsum();
        else if constexpr (P == 2)
            return derived().abs().square().rsum().sqrt();
        else
            return derived().abs().template pow<P>().rsum().pow(1.f / P);
    }
    template <Size P>
    auto cnorm() const
    {
        static_assert(P > 0, "P can't be zero in Matrix::cnorm");

        if constexpr (P == Inf)
            return derived().abs().cmax();
        else if constexpr (P == 1)
            return derived().abs().csum();
        else if constexpr (P == 2)
            return derived().abs().square().csum().sqrt();
        else
            return derived().abs().template pow<P>().csum().pow(1.f / P);
    }
    template <Size P>
    auto norm() const
    {
        static_assert(P > 0, "P can't be zero in Matrix::norm");

        if constexpr (P == Inf)
            return derived().abs().max();
        else if constexpr (P == 1)
            return derived().abs().sum();
        else if constexpr (P == 2)
            return std::sqrt(derived().abs().square().sum());
        else
            return std::pow(derived().abs().template pow<P>().sum(), 1.f / P);
    }

    template <Size P, typename Expr2, typename = IsExpr<Expr2>>
    auto bdistance(Size rows, Size cols, const Expr2 &expr2) const
    {
        return derived().sub(expr2).template bnorm<P>(rows, cols);
    }
    template <Size P, typename Expr2, typename = IsExpr<Expr2>>
    auto rdistance(const Expr2 &expr2) const
    {
        return derived().sub(expr2).template rnorm<P>();
    }
    template <Size P, typename Expr2, typename = IsExpr<Expr2>>
    auto cdistance(const Expr2 &expr2) const
    {
        return derived().sub(expr2).template cnorm<P>();
    }
    template <Size P, typename Expr2, typename = IsExpr<Expr2>>
    auto distance(const Expr2 &expr2) const
    {
        return derived().sub(expr2).template norm<P>();
    }

    template <Size P>
    auto bdistance(Size rows, Size cols, Type expr2) const
    {
        return derived().sub(expr2).template bnorm<P>(rows, cols);
    }
    template <Size P>
    auto rdistance(Type expr2) const
    {
        return derived().sub(expr2).template rnorm<P>();
    }
    template <Size P>
    auto cdistance(Type expr2) const
    {
        return derived().sub(expr2).template cnorm<P>();
    }
    template <Size P>
    auto distance(Type expr2) const
    {
        return derived().sub(expr2).template norm<P>();
    }

    auto bvar(Size rows, Size cols, Size dof = 0) const
    {
        Size size = rows * cols;
        auto expr1 = derived().square().bsum(rows, cols);
        auto expr2 = derived().bsum(rows, cols).square() / size;
        return (expr1 - expr2) / (size - dof);
    }
    auto rvar(Size dof = 0) const
    {
        Size size = this->cols();
        auto expr1 = derived().square().rsum();
        auto expr2 = derived().rsum().square() / size;
        return (expr1 - expr2) / (size - dof);
    }
    auto cvar(Size dof = 0) const
    {
        Size size = this->rows();
        auto expr1 = derived().square().csum();
        auto expr2 = derived().csum().square() / size;
        return (expr1 - expr2) / (size - dof);
    }
    auto var(Size dof = 0) const
    {
        Size size = this->rows() * this->cols();
        auto expr1 = derived().square().sum();
        auto expr2 = std::pow(derived().sum(), 2) / size;
        return (expr1 - expr2) / (size - dof);
    }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto bcov(const Expr2 &expr2, Size rows, Size cols, Size dof = 0) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "bcov", "Rows of matrices must be the same")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "bcov", "Cols of matrices must be the same")

        Size size = rows * cols;
        auto _expr1 = derived().mul(expr2).bsum(rows, cols);
        auto _expr2 = derived().bsum(rows, cols).mul(expr2.bsum(rows, cols)) / size;
        return (_expr1 - _expr2) / (size - dof);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto rcov(const Expr2 &expr2, Size dof = 0) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "rcov", "Rows of matrices must be the same")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "rcov", "Cols of matrices must be the same")

        Size size = this->cols();
        auto _expr1 = derived().mul(expr2).rsum();
        auto _expr2 = derived().rsum().mul(expr2.rsum()) / size;
        return (_expr1 - _expr2) / (size - dof);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto ccov(const Expr2 &expr2, Size dof = 0) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "ccov", "Rows of matrices must be the same")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "ccov", "Cols of matrices must be the same")

        Size size = this->rows();
        auto _expr1 = derived().mul(expr2).csum();
        auto _expr2 = derived().csum().mul(expr2.csum()) / size;
        return (_expr1 - _expr2) / (size - dof);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto cov(const Expr2 &expr2, Size dof = 0) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "cov", "Rows of matrices must be the same")
        TENSE_MASSERT(this->cols(), ==, expr2.cols(), "cov", "Cols of matrices must be the same")

        Size size = this->rows() * this->cols();
        auto _expr1 = derived().mul(expr2).sum();
        auto _expr2 = derived().sum() * expr2.sum() / size;
        return (_expr1 - _expr2) / (size - dof);
    }
    auto cov(Size dof = 0) const
    {
        auto mat = (derived() - derived().cmean()).eval();
        return mat.adjoint() * mat / (mat.cols() - dof);
    }

    auto bmean(Size rows, Size cols) const { return derived().bsum(rows, cols) / (rows * cols); }
    auto rmean() const { return derived().rsum() / this->cols(); }
    auto cmean() const { return derived().csum() / this->rows(); }
    auto mean() const { return derived().sum() / (this->rows() * this->cols()); }

    auto bstd(Size rows, Size cols, Size dof = 0) const { return derived().bvar(rows, cols, dof).sqrt(); }
    auto rstd(Size dof = 0) const { return derived().rvar(dof).sqrt(); }
    auto cstd(Size dof = 0) const { return derived().cvar(dof).sqrt(); }
    auto std(Size dof = 0) const { return std::sqrt(derived().var(dof)); }

    auto bcontains(Size rows, Size cols, Type expr2) const { return derived().eq(expr2).bany(rows, cols); }
    auto rcontains(Type expr2) const { return derived().eq(expr2).rany(); }
    auto ccontains(Type expr2) const { return derived().eq(expr2).cany(); }
    auto contains(Type expr2) const { return derived().eq(expr2).any(); }

    auto bequal(Size rows, Size cols, Type expr2) const { return derived().eq(expr2).ball(rows, cols); }
    auto requal(Type expr2) const { return derived().eq(expr2).rall(); }
    auto cequal(Type expr2) const { return derived().eq(expr2).call(); }
    auto equal(Type expr2) const { return derived().eq(expr2).all(); }

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto bequal(Size rows, Size cols, const Expr2 &expr2) const
    {
        return derived().eq(expr2).ball(rows, cols);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto requal(const Expr2 &expr2) const
    {
        return derived().eq(expr2).rall();
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto cequal(const Expr2 &expr2) const
    {
        return derived().eq(expr2).call();
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto equal(const Expr2 &expr2) const
    {
        return derived().eq(expr2).all();
    }

    auto bnormalize(Size rows, Size cols) const
    {
        auto min = derived().bmin(rows, cols).eval();
        auto dist = derived().bmax(rows, cols).sub(min).eval();
        return (derived() - min) / dist;
    }
    auto rnormalize() const
    {
        auto min = derived().rmin().eval();
        auto dist = derived().rmax().sub(min).eval();
        return (derived() - min) / dist;
    }
    auto cnormalize() const
    {
        auto min = derived().cmin().eval();
        auto dist = derived().cmax().sub(min).eval();
        return (derived() - min) / dist;
    }
    auto normalize() const
    {
        auto min = derived().min();
        auto dist = (derived().max() - min);
        return (derived() - min) / dist;
    }

    auto bstandize(Size rows, Size cols, Size dof = 0) const
    {
        auto mean = derived().bmean(rows, cols).eval();
        auto std = derived().bstd(rows, cols, dof).eval();
        return (derived() - mean) / std;
    }
    auto rstandize(Size dof = 0) const
    {
        auto mean = derived().rmean().eval();
        auto std = derived().rstd(dof).eval();
        return (derived() - mean) / std;
    }
    auto cstandize(Size dof = 0) const
    {
        auto mean = derived().cmean().eval();
        auto std = derived().cstd(dof).eval();
        return (derived() - mean) / std;
    }
    auto standize(Size dof = 0) const
    {
        auto mean = derived().mean();
        auto std = derived().std(dof);
        return (derived() - mean) / std;
    }

    auto close(Type expr2, typename FromComplex<Type>::Type expr3 = 1e-5) const
    {
        return derived().sub(expr2).abs().le(expr3).all();
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto close(const Expr2 &expr2, typename FromComplex<Type>::Type expr3 = 1e-5) const
    {
        return derived().sub(expr2).abs().le(expr3).all();
    }
    auto trace() const { return derived().diag().sum(); }
    auto adjoint() const { return derived().conj().trans(); }

    auto abs() const
    {
        return unary<typename FromComplex<Type>::Type>([](auto val1) { return std::abs(val1); });
    }
    auto clip(Type val2, Type val3) const
    {
        if (val2 > val3) std::swap(val2, val3);
        return unary<Type>([val2, val3](auto val1) { return std::min(std::max(val1, val2), val3); });
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto dot(const Expr2 &expr2) const
    {
        const auto &expr1 = derived();
        TENSE_MASSERT(std::min(expr1.rows(), expr1.cols()), ==, 1, "dot", "Fist matrix must be 1D")
        TENSE_MASSERT(std::min(expr2.rows(), expr2.cols()), ==, 1, "dot", "Fist matrix must be 1D")

        if (expr1.rows() == 1 && expr2.cols() == 1 && expr1.cols() == expr2.rows())
            return expr1.mul(expr2.trans()).sum();
        if (expr1.cols() == 1 && expr2.rows() == 1 && expr1.rows() == expr2.cols())
            return expr1.mul(expr2.trans()).sum();
        else if (expr1.rows() == 1 && expr2.rows() == 1 && expr1.cols() == expr2.cols())
            return expr1.mul(expr2).sum();
        else if (expr1.cols() == 1 && expr2.cols() == 1 && expr1.rows() == expr2.rows())
            return expr1.mul(expr2).sum();
    }

    ////////////////////////////////////////////////////////////////////////////

    static auto init(Size rows, Size cols, Type val)
    {
        TENSE_MASSERT(rows, >, 0, "init", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "init", "Input cols can't be zero")
        return Constant<Major, Type>(rows, cols, val);
    }
    static auto eye(Size size)
    {
        TENSE_MASSERT(size, >, 0, "eye", "Input size can't be zero")
        return Eye<Major, Type>(size, size);
    }
    static auto eye(Size rows, Size cols)
    {
        TENSE_MASSERT(rows, >, 0, "eye", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "eye", "Input cols can't be zero")
        return Eye<Major, Type>(rows, cols);
    }
    template <typename Dist>
    static auto dist(Size rows, Size cols, Dist dist)
    {
        TENSE_MASSERT(rows, >, 0, "dist", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "dist", "Input cols can't be zero")
        return Distribution<Major, Type, Dist>(rows, cols, dist);
    }
    static auto seq(Size rows, Size cols, Type start, Type end)
    {
        TENSE_MASSERT(rows, >, 0, "seq", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "seq", "Input cols can't be zero")
        auto step = (end - start) / (rows * cols);
        return Sequence<Major, Type>(rows, cols, start, step);
    }
    static auto strided(Size rows, Size cols, Type *data, Size rstride, Size cstride)
    {
        TENSE_MASSERT(rows, >, 0, "seq", "Input rows can't be zero")
        TENSE_MASSERT(cols, >, 0, "seq", "Input cols can't be zero")
        return Strided<Major, Type>(rows, cols, data, rstride, cstride);
    }
    template <Size Rows, Size Cols>
    static auto stat(Type value = Type(0))
    {
        static_assert(Rows > 0, "Rows can't be zero in Matrix::static");
        static_assert(Cols > 0, "Cols can't be zero in Matrix::static");
        return Static<Major, Type, Rows, Cols>(value);
    }
    template <Size Rows, Size Cols>
    static auto stat(const std::vector<Type> &list)
    {
        static_assert(Rows > 0, "Rows can't be zero in Matrix::static");
        static_assert(Cols > 0, "Cols can't be zero in Matrix::static");
        return Static<Major, Type, Rows, Cols>(list);
    }
    template <Size Rows, Size Cols>
    static auto stat(IL1D<Type> list)
    {
        TENSE_MASSERT(list.size(), ==, Rows, "stat", "Matrix rows must match initializer list")
        TENSE_MASSERT(Cols, ==, 1, "stat", "Matrix cols must match initializer list")
        auto target = stat<Rows, Cols>();
        Eval::assign(target, list);
        return target;
    }
    template <Size Rows, Size Cols>
    static auto stat(IL2D<Type> list)
    {
        TENSE_MASSERT(list.size(), ==, Rows, "stat", "Matrix rows must match initializer list")
        if (list.size() != 0)
            TENSE_MASSERT(list.begin()->size(), ==, Cols, "stat", "Matrix cols must match initializer list")
        auto target = stat<Rows, Cols>();
        Eval::assign(target, list);
        return target;
    }

    ////////////////////////////////////////////////////////////////////////////

    static auto numin(Size rows, Size cols) { return init(rows, cols, std::numeric_limits<Type>::min()); }
    static auto numax(Size rows, Size cols) { return init(rows, cols, std::numeric_limits<Type>::max()); }
    static auto lowest(Size rows, Size cols) { return init(rows, cols, std::numeric_limits<Type>::lowest()); }
    static auto epsilon(Size rows, Size cols) { return init(rows, cols, std::numeric_limits<Type>::epsilon()); }
    static auto inf(Size rows, Size cols) { return init(rows, cols, std::numeric_limits<Type>::infinity()); }
    static auto nan(Size rows, Size cols) { return init(rows, cols, std::numeric_limits<Type>::quiet_NaN()); }
    static auto zeros(Size rows, Size cols) { return init(rows, cols, 0); }
    static auto ones(Size rows, Size cols) { return init(rows, cols, 1); }
    static auto e(Size rows, Size cols) { return init(rows, cols, M_E); }
    static auto pi(Size rows, Size cols) { return init(rows, cols, M_PI); }
    static auto sqrt2(Size rows, Size cols) { return init(rows, cols, M_SQRT2); }

    ////////////////////////////////////////////////////////////////////////////

    static auto uniform(Size rows, Size cols, Type a = 0, Type b = 1)
    {
        if constexpr (std::is_integral<Type>::value)
            return dist(rows, cols, std::uniform_int_distribution<Type>(a, b));
        else if constexpr (std::is_floating_point<Type>::value)
            return dist(rows, cols, std::uniform_real_distribution<Type>(a, b));
        else
            throw std::runtime_error("Data type not supported in matrix::uniform.");
    }
    static auto bernoulli(Size rows, Size cols, double p = 0.5)
    {
        return dist(rows, cols, std::bernoulli_distribution(p));
    }
    static auto binomial(Size rows, Size cols, int t, double p = 0.5)
    {
        return dist(rows, cols, std::binomial_distribution(t, p));
    }
    static auto geometric(Size rows, Size cols, double p = 0.5)
    {
        return dist(rows, cols, std::geometric_distribution(p));
    }
    static auto pascal(Size rows, Size cols, int k, double p = 0.5)
    {
        return dist(rows, cols, std::negative_binomial_distribution(k, p));
    }
    static auto poisson(Size rows, Size cols, double mean = 1)
    {
        return dist(rows, cols, std::poisson_distribution(mean));
    }
    static auto exponential(Size rows, Size cols, double lambda = 1)
    {
        return dist(rows, cols, std::exponential_distribution(lambda));
    }
    static auto gamma(Size rows, Size cols, double alpha = 1, double beta = 1)
    {
        return dist(rows, cols, std::gamma_distribution(alpha, beta));
    }
    static auto weibull(Size rows, Size cols, double a = 1, double b = 1)
    {
        return dist(rows, cols, std::weibull_distribution(a, b));
    }
    static auto extremevalue(Size rows, Size cols, double a = 0, double b = 1)
    {
        return dist(rows, cols, std::extreme_value_distribution(a, b));
    }
    static auto normal(Size rows, Size cols, double mean = 0, double std = 1)
    {
        return dist(rows, cols, std::normal_distribution(mean, std));
    }
    static auto lognormal(Size rows, Size cols, double m = 0, double s = 1)
    {
        return dist(rows, cols, std::lognormal_distribution(m, s));
    }
    static auto chisquared(Size rows, Size cols, double n = 1)
    {
        return dist(rows, cols, std::chi_squared_distribution(n));
    }
    static auto cauchy(Size rows, Size cols, double a = 0, double b = 1)
    {
        return dist(rows, cols, std::cauchy_distribution(a, b));
    }
    static auto fisherf(Size rows, Size cols, double m = 1, double n = 1)
    {
        return dist(rows, cols, std::fisher_f_distribution(m, n));
    }
    static auto studentt(Size rows, Size cols, double n = 1)
    {
        return dist(rows, cols, std::student_t_distribution(n));
    }

    ////////////////////////////////////////////////////////////////////////////

    template <typename Func>
    auto rsort(Func func) const
    {
        auto target = derived().copy();
        Backend::rsort(target, func);
        return target;
    }
    template <typename Func>
    auto csort(Func func) const
    {
        auto target = derived().copy();
        Backend::csort(target, func);
        return target;
    }
    auto rsort() const
    {
        return rsort([](auto i, auto j) { return i < j; });
    }
    auto csort() const
    {
        return csort([](auto i, auto j) { return i < j; });
    }

    template <typename Func>
    auto rsortidx(Func func) const
    {
        return Backend::rsortidx(derived(), func);
    }
    template <typename Func>
    auto csortidx(Func func) const
    {
        return Backend::csortidx(derived(), func);
    }
    auto rsortidx() const
    {
        return rsortidx([](auto i, auto j) { return i < j; });
    }
    auto csortidx() const
    {
        return csortidx([](auto i, auto j) { return i < j; });
    }

    auto rshuffle() const
    {
        auto target = derived().copy();
        Backend::rshuffle(target);
        return target;
    }
    auto cshuffle() const
    {
        auto target = derived().copy();
        Backend::cshuffle(target);
        return target;
    }
    auto shuffle() const
    {
        auto target = derived().copy();
        Backend::shuffle(target);
        return target;
    }
    auto rshuffleidx() const { return Backend::rshuffleidx(derived()); }
    auto cshuffleidx() const { return Backend::cshuffleidx(derived()); }

    ////////////////////////////////////////////////////////////////////////////

    template <typename Expr2, typename Expr3, typename = IsExpr<Expr2>, typename = IsExpr<Expr3>>
    auto _mm(const Expr2 &expr2, Expr3 &expr3) const
    {
        TENSE_MASSERT(this->cols(), ==, expr2.rows(), "mm",
                      "Cols of first matrix must be equal to rows of second matrix")
        TENSE_MASSERT(this->rows(), ==, expr3.rows(), "mm",
                      "Rows of first matrix must be equal to rows of output matrix")
        TENSE_MASSERT(expr2.cols(), ==, expr3.cols(), "mm",
                      "Cols of second matrix must be equal to rows of output matrix")

        Backend::multiply(derived(), expr2, expr3);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto mm(const Expr2 &expr2, Matrix<Major, Type> &expr3) const
    {
        static_assert(std::is_same<Type, typename Expr2::Type>::value,
                      "Data type of matrices must be the same in matrix::mm.");
        static_assert(std::is_same<Major, typename Expr2::Major>::value,
                      "Major of matrices must be the same in matrix::mm.");

        TENSE_MASSERT(this->cols(), ==, expr2.rows(), "mm",
                      "Cols of first matrix must be equal to rows of second matrix")
        TENSE_MASSERT(this->rows(), ==, expr3.rows(), "mm",
                      "Rows of first matrix must be equal to rows of output matrix")
        TENSE_MASSERT(expr2.cols(), ==, expr3.cols(), "mm",
                      "Cols of second matrix must be equal to rows of output matrix")

        if constexpr (!IsReal<Type>::value)
            Backend::multiply(derived(), expr2, expr3);
        else
            External::multiply(derived(), expr2, expr3);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto _mm(const Expr2 &expr2) const
    {
        Matrix<Major, Type> expr3(this->rows(), expr2.cols());
        _mm(expr2, expr3);
        return expr3;
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto mm(const Expr2 &expr2) const
    {
        Matrix<Major, Type> expr3(this->rows(), expr2.cols());
        mm(expr2, expr3);
        return expr3;
    }

    auto inverse() const
    {
        TENSE_MASSERT(this->rows(), ==, this->cols(), "inverse", "Matrix must be square")
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::inverse.");
        return External::inverse(derived());
    }
    auto det() const
    {
        TENSE_MASSERT(this->rows(), ==, this->cols(), "det", "Matrix must be square")
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::inverse.");
        return External::determinant(derived());
    }
    auto plu() const
    {
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::plu.");
        return External::lufact(derived());
    }
    auto cholesky() const
    {
        TENSE_MASSERT(this->rows(), ==, this->cols(), "cholesky", "Matrix must be square")
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::cholesky.");
        return External::cholesky(derived());
    }
    auto qr() const
    {
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::qr.");
        return External::qrfact(derived());
    }
    auto schur(bool vectors = true) const
    {
        TENSE_MASSERT(this->rows(), ==, this->cols(), "schur", "Matrix must be square")
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::schur.");
        return External::schur(derived(), vectors);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto solve(const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->rows(), ==, this->cols(), "solve", "Matrix must be square")
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "solve",
                      "Cols of matrix must be equal to rows of right hand side")

        static_assert(IsReal<Type>::value, "Data type not supported in matrix::solve.");
        static_assert(std::is_same<Type, typename Expr2::Type>::value,
                      "Data type of matrices must be the same in matrix::solve.");
        static_assert(std::is_same<Major, typename Expr2::Major>::value,
                      "Data type of matrices must be the same in matrix::solve.");

        return External::solve(derived(), expr2);
    }
    template <typename Expr2, typename = IsExpr<Expr2>>
    auto ls(const Expr2 &expr2) const
    {
        TENSE_MASSERT(this->rows(), ==, expr2.rows(), "ls",
                      "Second dimention of matrix must be equal to rows of right hand size")

        static_assert(IsReal<Type>::value, "Data type not supported in matrix::ls.");
        static_assert(std::is_same<Type, typename Expr2::Type>::value,
                      "Data type of matrices must be the same in matrix::ls.");
        static_assert(std::is_same<Major, typename Expr2::Major>::value,
                      "Data type of matrices must be the same in matrix::ls.");

        return External::lsquares(derived(), expr2);
    }
    auto eigen(bool left = true, bool right = true) const
    {
        TENSE_MASSERT(this->rows(), ==, this->cols(), "eigen", "Matrix must be square")
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::eigen.");
        return External::eigen(derived(), left, right);
    }
    auto svd(bool left = true, bool right = true) const
    {
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::svd.");
        return External::svd(derived(), left, right);
    }
    auto rank(typename FromComplex<Type>::Type epsilon) const
    {
        static_assert(IsReal<Type>::value, "Data type not supported in matrix::rank.");
        return External::rank(derived(), epsilon);
    }

    ////////////////////////////////////////////////////////////////////////////

    UNARY0(cos, std::cos(val1))
    UNARY0(sin, std::sin(val1))
    UNARY0(tan, std::tan(val1))
    UNARY0(acos, std::acos(val1))
    UNARY0(asin, std::asin(val1))
    UNARY0(atan, std::atan(val1))
    UNARY0(cosh, std::cosh(val1))
    UNARY0(sinh, std::sinh(val1))
    UNARY0(tanh, std::tanh(val1))
    UNARY0(acosh, std::acosh(val1))
    UNARY0(asinh, std::asinh(val1))
    UNARY0(exp, std::exp(val1))
    UNARY0(log, std::log(val1))
    UNARY0(log2, std::log2(val1))
    UNARY0(log10, std::log10(val1))
    UNARY0(exp2, std::exp2(val1))
    UNARY0(expm1, std::expm1(val1))
    UNARY0(ilogb, std::ilogb(val1))
    UNARY0(log1p, std::log1p(val1))
    UNARY0(sqrt, std::sqrt(val1))
    UNARY0(cbrt, std::cbrt(val1))
    UNARY0(erf, std::erf(val1))
    UNARY0(erfc, std::erfc(val1))
    UNARY0(tgamma, std::tgamma(val1))
    UNARY0(lgamma, std::lgamma(val1))
    UNARY0(ceil, std::ceil(val1))
    UNARY0(floor, std::floor(val1))
    UNARY0(trunc, std::trunc(val1))
    UNARY0(round, std::round(val1))
    UNARY0(lround, std::lround(val1))
    UNARY0(llround, std::llround(val1))
    UNARY0(rint, std::rint(val1))
    UNARY0(lrint, std::lrint(val1))
    UNARY0(llrint, std::llrint(val1))
    UNARY0(nearbyint, std::nearbyint(val1))
    UNARY0(proj, std::proj(val1))
    UNARY0(neg, -val1)
    UNARY0(pos, +val1)
    UNARY0(_not, !val1)
    UNARY0(square, val1 *val1)
    UNARY0(cube, val1 *val1 *val1)
    UNARY0(frac, val1 - std::floor(val1))
    UNARY0(ln, std::log(val1))
    UNARY0(rev, 1 / val1)
    UNARY0(rsqrt, 1 / std::sqrt(val1))
    UNARY0(relu, std::fmax(0, val1))
    UNARY0(sigmoid, 1 / (1 + std::exp(-val1)))
    UNARY0(deg2rad, val1 *M_PI / 180)
    UNARY0(rad2deg, val1 * 180 / M_PI)

    ////////////////////////////////////////////////////////////////////////////

    UNARY1(atan2, Type, std::atan2(val1, val2))
    UNARY1(fdim, Type, std::fdim(val1, val2))
    UNARY1(ldexp, int, std::ldexp(val1, val2))
    UNARY1(scalbn, int, std::scalbn(val1, val2))
    UNARY1(scalbln, long, std::scalbln(val1, val2))
    UNARY1(pow, Type, std::pow(val1, val2))
    UNARY1(hypot, Type, std::hypot(val1, val2))
    UNARY1(remainder, Type, std::remainder(val1, val2))
    UNARY1(copysign, Type, std::copysign(val1, val2))
    UNARY1(nextafter, Type, std::nextafter(val1, val2))
    UNARY1(nexttoward, Type, std::nexttoward(val1, val2))
    UNARY1(fmin, Type, std::fmax(val1, val2))
    UNARY1(fmax, Type, std::fmin(val1, val2))
    UNARY1(add, Type, val1 + val2)
    UNARY1(sub, Type, val1 - val2)
    UNARY1(mul, Type, val1 *val2)
    UNARY1(div, Type, val1 / val2)
    UNARY1(mod, Type, val1 % val2)
    UNARY1(_and, Type, val1 &val2)
    UNARY1(_or, Type, val1 | val2)
    UNARY1(_xor, Type, val1 ^ val2)
    UNARY1(lshift, Size, val1 << val2)
    UNARY1(rshift, Size, val1 >> val2)
    UNARY1(revsub, Type, val2 - val1)
    UNARY1(revdiv, Type, val2 / val1)
    UNARY1(revmod, Type, val2 % val1)
    UNARY1(revlshift, Type, val2 << val1)
    UNARY1(revlrshift, Type, val2 >> val1)
    UNARY1(heaviside, Type, val1 < 0 ? 0 : (val1 > 0 ? 1 : val2))

    ////////////////////////////////////////////////////////////////////////////

    _UNARY0(arg, typename Type::value_type, std::arg(val1))
    _UNARY0(norm, typename Type::value_type, std::norm(val1))
    _UNARY0(real, typename Type::value_type, val1.real())
    _UNARY0(imag, typename Type::value_type, val1.imag())
    _UNARY0(isnan, bool, std::isnan(val1))
    _UNARY0(isinf, bool, std::isinf(val1))
    _UNARY0(sign, bool, std::signbit(val1))
    _UNARY0(zero, bool, val1 == 0)
    _UNARY0(nonzero, bool, val1 != 0)
    _UNARY1(gt, bool, Type, val1 > val2)
    _UNARY1(ge, bool, Type, val1 >= val2)
    _UNARY1(lt, bool, Type, val1 < val2)
    _UNARY1(le, bool, Type, val1 <= val2)
    _UNARY1(eq, bool, Type, val1 == val2)
    _UNARY1(ne, bool, Type, val1 != val2)
    _BINARY(gt, bool, val1 > val2)
    _BINARY(ge, bool, val1 >= val2)
    _BINARY(lt, bool, val1 < val2)
    _BINARY(le, bool, val1 <= val2)
    _BINARY(eq, bool, val1 == val2)
    _BINARY(ne, bool, val1 != val2)
    _BINARY(complex, std::complex<Type>, std::complex(val1, val2))
    _BINARY(lshift, Size, val1 << val2)
    _BINARY(rshift, Size, val1 >> val2)

    ////////////////////////////////////////////////////////////////////////////

    BINARY(add, val1 + val2)
    BINARY(sub, val1 - val2)
    BINARY(mul, val1 *val2)
    BINARY(div, val1 / val2)
    BINARY(mod, val1 % val2)
    BINARY(_and, val1 &val2)
    BINARY(_or, val1 | val2)
    BINARY(_xor, val1 ^ val2)
    BINARY(atan2, std::atan2(val1, val2))
    BINARY(pow, std::pow(val1, val2))
    BINARY(remainder, std::remainder(val1, val2))
    BINARY(fmin, std::fmin(val1, val2))
    BINARY(fmax, std::fmax(val1, val2))
    BINARY(mask, val2 ? val1 : 0)
    BINARY(heaviside, val1 < 0 ? 0 : (val1 > 0 ? 1 : val2))

    ////////////////////////////////////////////////////////////////////////////

    REDUCE0(sum, 0, val1 + val2)
    REDUCE0(prod, 1, val1 *val2)
    REDUCE0(max, std::numeric_limits<Type>::min(), std::max(val1, val2))
    REDUCE0(min, std::numeric_limits<Type>::max(), std::min(val1, val2))
    _REDUCE1(count, Size, Type, 0, val1 + (val2 == val3))
    SQUARE(diagonal, SDiagonal)
    SQUARE(zdiagonal, ZDiagonal)
    SQUARE(odiagonal, ODiagonal)
    SQUARE(usymm, USymmetric)
    SQUARE(lsymm, LSymmetric)
    SQUARE(upper, Upper)
    SQUARE(lower, Lower)
    SQUARE(oupper, OUpper)
    SQUARE(olower, OLower)
    SQUARE(zupper, ZUpper)
    SQUARE(zlower, ZLower)

    ////////////////////////////////////////////////////////////////////////////

    OPERATOR0(neg, -)
    OPERATOR0(pos, +)
    OPERATOR0(_not, !)
    OPERATOR0(_not, ~)
    OPERATOR1(lt, gt, <)
    OPERATOR1(le, ge, <=)
    OPERATOR1(gt, lt, >)
    OPERATOR1(ge, le, >=)
    OPERATOR1(eq, eq, ==)
    OPERATOR1(ne, ne, !=)
    OPERATOR1(add, add, +)
    OPERATOR1(sub, revsub, -)
    OPERATOR1(div, revdiv, /)
    OPERATOR1(mod, revmod, %)
    OPERATOR1(_and, _and, &)
    OPERATOR1(_or, _or, |)
    OPERATOR1(_xor, _xor, ^)
    OPERATOR1(lshift, revlshift, <<)
    OPERATOR1(rshift, revrshift, >>)

    template <typename Expr2, typename = IsExpr<Expr2>>
    auto operator*(const Expr2 &expr2) const
    {
        return derived().mm(expr2);
    }
    auto operator*(Type expr2) const { return derived().mul(expr2); }
    friend auto operator*(Type expr2, const Derived &expr1) { return expr1.mul(expr2); }
};
}  // namespace Tense::MatrixImpl

#undef _UNARY0
#undef _UNARY1
#undef _BINARY
#undef _REDUCE0
#undef _REDUCE1
#undef _SQUARE
#undef SQUARE
#undef UNARY0
#undef UNARY1
#undef BINARY
#undef REDUCE0
#undef REDUCE1
#undef _OPERATOR0
#undef _OPERATOR1
#undef OPERATOR1
