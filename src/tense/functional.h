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

#include <complex>

#define _USE_MATH_DEFINES
#include <cmath>

#define FUNC(NAME)                                     \
    template <typename Expr, typename... Args>         \
    auto NAME(const Expr &expr, Args &&...args)        \
    {                                                  \
        return expr.NAME(std::forward<Args>(args)...); \
    }
#define TEMP(NAME, TYPE)                                           \
    template <TYPE T, typename Expr, typename... Args>             \
    auto NAME(const Expr &expr, Args &&...args)                    \
    {                                                              \
        return expr.template NAME<T>(std::forward<Args>(args)...); \
    }
#define MATH(NAME)   \
    using std::NAME; \
    FUNC(NAME);

namespace Tense::Functional
{
FUNC(memory)
FUNC(resize)
FUNC(release)
FUNC(reset)
FUNC(valid)
FUNC(rows)
FUNC(cols)
FUNC(size)

MATH(cos)
MATH(sin)
MATH(tan)
MATH(acos)
MATH(asin)
MATH(atan)
MATH(cosh)
MATH(sinh)
MATH(tanh)
MATH(acosh)
MATH(asinh)
MATH(exp)
MATH(log)
MATH(log2)
MATH(log10)
MATH(exp2)
MATH(expm1)
MATH(ilogb)
MATH(log1p)
MATH(sqrt)
MATH(cbrt)
MATH(erf)
MATH(erfc)
MATH(tgamma)
MATH(lgamma)
MATH(ceil)
MATH(floor)
MATH(trunc)
MATH(round)
MATH(lround)
MATH(llround)
MATH(rint)
MATH(lrint)
MATH(llrint)
MATH(nearbyint)
MATH(proj)
FUNC(neg)
FUNC(pos)
FUNC(_not)
FUNC(square)
FUNC(cube)
FUNC(frac)
FUNC(ln)
FUNC(rev)
FUNC(rsqrt)
FUNC(relu)
FUNC(sigmoid)
FUNC(deg2rad)
FUNC(rad2deg)
MATH(atan2)
MATH(fdim)
MATH(ldexp)
MATH(scalbn)
MATH(scalbln)
MATH(pow)
MATH(hypot)
MATH(remainder)
MATH(copysign)
MATH(nextafter)
MATH(nexttoward)
MATH(fmin)
MATH(fmax)
FUNC(add)
FUNC(sub)
FUNC(mul)
FUNC(div)
FUNC(mod)
FUNC(_and)
FUNC(_or)
FUNC(_xor)
FUNC(lshift)
FUNC(rshift)
FUNC(revsub)
FUNC(revdiv)
FUNC(revmod)
FUNC(revlshift)
FUNC(revlrshift)
FUNC(heaviside)
MATH(arg)
MATH(norm)
FUNC(real)
FUNC(imag)
FUNC(isnan)
FUNC(isinf)
FUNC(sign)
FUNC(zero)
FUNC(nonzero)
FUNC(gt)
FUNC(ge)
FUNC(lt)
FUNC(le)
FUNC(eq)
FUNC(ne)
FUNC(complex)
FUNC(mask)
FUNC(sum)
FUNC(prod)
FUNC(max)
FUNC(min)
FUNC(count)
FUNC(diagonal)
FUNC(zdiagonal)
FUNC(odiagonal)
FUNC(usymm)
FUNC(lsymm)
FUNC(upper)
FUNC(lower)
FUNC(oupper)
FUNC(olower)
FUNC(zupper)
FUNC(zlower)

FUNC(brepeat)
FUNC(rrepeat)
FUNC(crepeat)
FUNC(repeat)
FUNC(block)
FUNC(row)
FUNC(col)
FUNC(elem)
FUNC(diag)
FUNC(index)
FUNC(rcat0)
FUNC(ccat0)
FUNC(rcat1)
FUNC(ccat1)
FUNC(rturn)
FUNC(cturn)
FUNC(turn)
FUNC(rminidx)
FUNC(cminidx)
FUNC(rmaxidx)
FUNC(cmaxidx)
FUNC(ball)
FUNC(rall)
FUNC(call)
FUNC(all)
FUNC(bany)
FUNC(rany)
FUNC(cany)
FUNC(any)
FUNC(bflip)
FUNC(rflip)
FUNC(cflip)
FUNC(flip)
FUNC(asdiag)
FUNC(reshape)
FUNC(where)
FUNC(iwhere)
FUNC(uherm)
FUNC(lherm)
MATH(polar)
MATH(conj)
FUNC(trans)
FUNC(expr)
FUNC(eval)
FUNC(copy)
FUNC(item)
FUNC(shape)
FUNC(tensor)
FUNC(topleft)
FUNC(topright)
FUNC(bottomleft)
FUNC(bottomright)
FUNC(bmean)
FUNC(rmean)
FUNC(cmean)
FUNC(mean)
FUNC(bstd)
FUNC(rstd)
FUNC(cstd)
FUNC(std)
FUNC(bvar)
FUNC(rvar)
FUNC(cvar)
FUNC(var)
FUNC(bcov)
FUNC(rcov)
FUNC(ccov)
FUNC(cov)
FUNC(bcontains)
FUNC(rcontains)
FUNC(ccontains)
FUNC(contains)
FUNC(bequal)
FUNC(requal)
FUNC(cequal)
FUNC(equal)
FUNC(bnormalize)
FUNC(rnormalize)
FUNC(cnormalize)
FUNC(normalize)
FUNC(bstandize)
FUNC(rstandize)
FUNC(cstandize)
FUNC(standize)
FUNC(close)
FUNC(trace)
FUNC(adjoint)
MATH(abs)
FUNC(clip)
FUNC(dot)
FUNC(rsort)
FUNC(csort)
FUNC(sort)
FUNC(rshuffle)
FUNC(cshuffle)
FUNC(shuffle)
FUNC(_mm)
FUNC(mm)
FUNC(inverse)
FUNC(det)
FUNC(plu)
FUNC(cholesky)
FUNC(qr)
FUNC(schur)
FUNC(solve)
FUNC(ls)
FUNC(eigen)
FUNC(svd)
FUNC(rank)
FUNC(view)
FUNC(strided)
FUNC(indirect)
FUNC(matrix)

TEMP(pow, int)
TEMP(type, typename)
TEMP(bnorm, int)
TEMP(rnorm, int)
TEMP(cnorm, int)
TEMP(norm, int)
TEMP(bdistance, int)
TEMP(rdistance, int)
TEMP(cdistance, int)
TEMP(distance, int)
}  // namespace Tense::Functional

#undef FUNC
#undef TEMP
#undef MATH
