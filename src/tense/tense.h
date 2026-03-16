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

#include <tense/functional.h>
#include <tense/matrix/impl.h>
#include <tense/struct.h>
#include <tense/tensor/impl.h>
#include <tense/util.h>
#include <tense/vector/impl.h>

#undef TENSE_VASSERT
#undef TENSE_MASSERT
#undef TENSE_TASSERT
#undef TENSE_ALIGNMENT
#undef TENSE_PARALLEL_FOR

// TODO % operator to fmod in floats?
// TODO conditio openmp pragmas on size
// TODO fwhere and constant readable or writable?
// TODO detect overlapping read/writes and eval first
// TODO add block reduce to vector
// TODO sort based on expr not ptr
// TODO replace var/cov with two-pass algorithm
// TODO shape should be small vector
// TODO replace reduction expression in vector with actual computation
// TODO maybe remove semi-colon from asserts
// TODO use std::array in static
// TODO maybe implement sortidx and shuffleidx
// TODO sort/shuffle idx functions should be heavy aware
// TODO distribution is not thread-safe (vector, matrix, tensor)
// TODO Alias might cause temp Static to be held as reference (vector, matrix, tensor)
// TODO cov with self in matrix?
// TODO % is not performant in repeat (vector, matrix, tensor)
// TODO custom deleter and better valid check to prevent illigal access
