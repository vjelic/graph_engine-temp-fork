// Copyright 2023 Xilinx, Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cassert>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <immintrin.h>

#include "graph-engine/utils/CpuSupport.hpp"

namespace {

template<std::size_t PAYLOAD_BYTES_PER_VEC, std::size_t VECTOR_SIZE, std::size_t M, std::size_t N, std::size_t I>
constexpr uint8_t maskValue()
{
    if (I >= VECTOR_SIZE - PAYLOAD_BYTES_PER_VEC) {
        constexpr std::size_t INDEX = (VECTOR_SIZE - (I + 1));
        return static_cast<uint8_t>((INDEX / N) * M + (INDEX % N));
    }
    else {
        return -1;
    }
}

template<std::size_t PAYLOAD_BYTES_PER_VEC, std::size_t VECTOR_SIZE, std::size_t M, std::size_t N, typename T, std::size_t... I>
T buildMaskImpl(std::index_sequence<I...>);

template<std::size_t PAYLOAD_BYTES_PER_VEC, std::size_t VECTOR_SIZE, std::size_t M, std::size_t N, std::size_t... I>
__m256i buildMaskImpl(std::index_sequence<I...>)
{
    return _mm256_set_epi8(maskValue<PAYLOAD_BYTES_PER_VEC, VECTOR_SIZE, M, N, I>()...);
}

template<std::size_t PAYLOAD_BYTES_PER_VEC, std::size_t M, std::size_t N, typename T>
T buildMask()
{
    return buildMaskImpl<PAYLOAD_BYTES_PER_VEC, sizeof(T), M, N>(std::make_index_sequence<sizeof(T)>{});
}

inline void depad_avx512(const std::uint8_t* readPtr, unsigned int zeromask, const __m256i& mask, std::uint8_t* writePtr){
    __m256i inVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(readPtr));
    inVec = _mm256_maskz_permutexvar_epi8(zeromask, mask, inVec);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(writePtr), inVec);
}

template<std::size_t M, std::size_t N>
inline void fastDepad_4d_MtoN(const std::vector<std::int32_t>& srcShapes, const std::vector<std::int32_t>& srcStrides, std::uint8_t* data)
{
    static_assert(N < M, "N >= M does not make sense for M->N depadding!");
    static_assert(M <= 32, "M > 32 not supported!");

    constexpr std::size_t VEC_SIZE_BYTES = 256 / 8;
    constexpr std::size_t M_PER_VEC = VEC_SIZE_BYTES / M;
    constexpr std::size_t PAYLOAD_BYTES_PER_VEC = M_PER_VEC * N;
    constexpr std::size_t IN_STEP = M_PER_VEC * M;

    const std::uint8_t* __restrict readPtr = data;

    const __m256i mask = buildMask<PAYLOAD_BYTES_PER_VEC, M, N, __m256i>();
    const unsigned int zeromask = (1 << PAYLOAD_BYTES_PER_VEC) - 1;

    const std::uint8_t *inBuffMax = data + srcStrides[0];

    const std::size_t aMax = srcShapes[0];
    const std::size_t bMax = srcShapes[1];
    const std::size_t cMax = srcShapes[2] / M_PER_VEC;

    const auto stride_0 = srcStrides[0];
    const auto stride_1 = srcStrides[1];

    // This unroll factor is experimentally defined. If you change it, you need
    // to change the number of function calls in the inner for loop below.
    const auto unrollFactor = 3;
    const auto cRemainder = cMax % unrollFactor;
    const auto cNewMax = cMax - cRemainder;

    for (std::size_t a = 0; a < aMax; ++a) {
        const auto posA = a * stride_0;

        for (std::size_t b = 0; b < bMax; ++b) {
            const auto posB = b * stride_1;
            const auto* __restrict inAddr = &readPtr[posB + posA];
            std::uint8_t* __restrict writePtr = data + (b * PAYLOAD_BYTES_PER_VEC * cMax) + (a * bMax * PAYLOAD_BYTES_PER_VEC * cMax);

            // If vector size is not divisible by M, use non-vectorized approach as long as input and output vectors would overlap
            // (which would cause data corruption) or input read would be partially out of bounds (which could cause a crash).
            if (IN_STEP != VEC_SIZE_BYTES && ((writePtr - inAddr) < VEC_SIZE_BYTES || (inAddr + VEC_SIZE_BYTES) >= inBuffMax)) {
                for (std::size_t c = 0; c < cMax; ++c) {
                    auto* writePtrLocal = writePtr + c * PAYLOAD_BYTES_PER_VEC;
                    const auto* readPtrLocal = inAddr + c * IN_STEP;
                    for (std::size_t i = 0; i < M_PER_VEC; ++i) {
                        auto* writePtrLocal2 = writePtrLocal + i * N;
                        const auto* readPtrLocal2 = readPtrLocal + i * M;
                        std::memcpy(writePtrLocal2, readPtrLocal2, N);
                    }
                }
            }
            else {
                for (std::size_t c = 0; c < cNewMax; c+=unrollFactor) {
                    auto* writePtrLocal = writePtr + c * PAYLOAD_BYTES_PER_VEC;
                    const auto* readPtrLocal = inAddr + c * IN_STEP;
                    depad_avx512(readPtrLocal, zeromask, mask, writePtrLocal);
                    depad_avx512(readPtrLocal + IN_STEP, zeromask, mask, writePtrLocal + PAYLOAD_BYTES_PER_VEC);
                    depad_avx512(readPtrLocal + (2*IN_STEP), zeromask, mask, writePtrLocal + (2*PAYLOAD_BYTES_PER_VEC));
                }
                for (std::size_t c = 0; c < cRemainder; ++c) {
                    auto* writePtrLocal = writePtr + (cNewMax + c) * PAYLOAD_BYTES_PER_VEC;
                    const auto* readPtrLocal = inAddr + (cNewMax + c) * IN_STEP;
                    depad_avx512(readPtrLocal, zeromask, mask, writePtrLocal);
                }
            }
        }
    }
}

inline void fastDepad_4d_innerStride2(const std::vector<std::int32_t>& srcShapes, const std::vector<std::int32_t>& srcStrides, std::uint8_t* data)
{
    std::size_t destIndex = 0;

    uint8_t* __restrict writePtr = data;
    const uint8_t* __restrict readPtr = data;

    uint64_t posA = 0;
    for (std::size_t a = 0; a < srcShapes[0]; ++a) {
        uint64_t posB = 0;

        for (std::size_t b = 0; b < srcShapes[1]; ++b) {
            std::memcpy(writePtr + destIndex, readPtr + posB + posA, srcShapes[2] * srcShapes[3]);
            destIndex += srcShapes[2] * srcShapes[3];
            posB += srcStrides[1];
        }

        posA += srcStrides[0];
    }
}

inline void fastDepad_4d_innerStride1(const std::vector<std::int32_t>& srcShapes, const std::vector<std::int32_t>& srcStrides, std::uint8_t* data)
{
    std::size_t destIndex = 0;

    uint8_t* __restrict writePtr = data;
    const uint8_t* __restrict readPtr = data;

    uint64_t posA = 0;
    for (std::size_t a = 0; a < srcShapes[0]; ++a) {
        uint64_t posB = 0;

        for (std::size_t b = 0; b < srcShapes[1]; ++b) {
            uint64_t posC = 0;

            for (std::size_t c = 0; c < srcShapes[2]; ++c) {
                std::memcpy(writePtr + destIndex, readPtr + posC + posB + posA, srcShapes[3]);
                destIndex += srcShapes[3];

                posC += srcStrides[2];
            }

            posB += srcStrides[1];
        }

        posA += srcStrides[0];
    }
}

inline void fastDepad_4d_noInner(const std::vector<std::int32_t>& srcShapes, const std::vector<std::int32_t>& srcStrides, std::uint8_t* data)
{
    std::size_t destIndex = 0;
    const std::uint8_t* __restrict readPtr = data;

    uint64_t posA = 0;
    for (std::size_t a = 0; a < srcShapes[0]; ++a) {
        uint64_t posB = 0;

        for (std::size_t b = 0; b < srcShapes[1]; ++b) {
            uint64_t posC = 0;

            for (std::size_t c = 0; c < srcShapes[2]; ++c) {
                data[destIndex++] = readPtr[
                    posC
                        + posB
                        + posA
                ];

                posC += srcStrides[2];
            }

            posB += srcStrides[1];
        }

        posA += srcStrides[0];
    }
}

template<std::size_t M, std::size_t N>
inline bool fastDepadPossible_4d_MtoN(const std::vector<std::int32_t>& srcShapes, const std::vector<std::int32_t>& srcStrides)
{
    return (
                cpu_support::cpuSupport.avx2
            && cpu_support::cpuSupport.avx512vbmi2
            && srcStrides[3] == 1
            && srcShapes[3] == N
            && srcStrides[2] == M
            && (srcShapes[2] % (256 / 8 / M)) == 0
            && srcShapes[2] >= (256 / 8 / M)
        );
}

} // anon ns

namespace {

inline void fastDepad_4d(const std::vector<std::int32_t>& srcShapes, const std::vector<std::int32_t>& srcStrides, std::uint8_t* data)
{
    // 4->1 case
    if (fastDepadPossible_4d_MtoN<4, 1>(srcShapes, srcStrides)) {
        return fastDepad_4d_MtoN<4, 1>(srcShapes, srcStrides, data);
    }

    // 8->1 case
    else if (fastDepadPossible_4d_MtoN<8, 1>(srcShapes, srcStrides)) {
        return fastDepad_4d_MtoN<8, 1>(srcShapes, srcStrides, data);
    }

    // 16->1 case
    else if (fastDepadPossible_4d_MtoN<16, 1>(srcShapes, srcStrides)) {
        return fastDepad_4d_MtoN<16, 1>(srcShapes, srcStrides, data);
    }

    // 4->3 case
    else if (fastDepadPossible_4d_MtoN<4, 3>(srcShapes, srcStrides)) {
        return fastDepad_4d_MtoN<4, 3>(srcShapes, srcStrides, data);
    }

    // 8->3 case
    else if (fastDepadPossible_4d_MtoN<8, 3>(srcShapes, srcStrides)) {
        return fastDepad_4d_MtoN<8, 3>(srcShapes, srcStrides, data);
    }

    // 12->10 case
    else if (fastDepadPossible_4d_MtoN<12, 10>(srcShapes, srcStrides)) {
        return fastDepad_4d_MtoN<12, 10>(srcShapes, srcStrides, data);
    }

    // 16->3 case
    else if (fastDepadPossible_4d_MtoN<16, 3>(srcShapes, srcStrides)) {
        return fastDepad_4d_MtoN<16, 3>(srcShapes, srcStrides, data);
    }

    // memcpy in w loop
    else if (srcStrides[3] == 1 && srcShapes[3] > 1 && srcStrides[2] == srcShapes[3]) {
        return fastDepad_4d_innerStride2(srcShapes, srcStrides, data);
    }

    // Generic srcStrides[3] == 1 case (memcpy inner loop)
    else if (srcStrides[3] == 1 && srcShapes[3] > 1) {
        return fastDepad_4d_innerStride1(srcShapes, srcStrides, data);
    }

    // Case with no inner loop
    else if (srcShapes[3] == 1) {
        return fastDepad_4d_noInner(srcShapes, srcStrides, data);
    }

    // Generic 4d case
    else
    {
        std::size_t destIndex = 0;

        uint8_t* __restrict writePtr = data;
        const uint8_t* __restrict readPtr = data;

        uint64_t posA = 0;
        for (std::size_t a = 0; a < srcShapes[0]; ++a) {
            uint64_t posB = 0;

            for (std::size_t b = 0; b < srcShapes[1]; ++b) {
                uint64_t posC = 0;

                for (std::size_t c = 0; c < srcShapes[2]; ++c) {
                    for (std::size_t d = 0; d < srcShapes[3]; ++d) {
                        writePtr[destIndex++] = readPtr[

                            d * srcStrides[3]
                                + posC
                                + posB
                                + posA

                        ];
                    }

                    posC += srcStrides[2];
                }

                posB += srcStrides[1];
            }

            posA += srcStrides[0];
        }
    }
}

} // anon ns

void fastDepad(const std::vector<std::int32_t>& srcShapes, const std::vector<std::int32_t>& srcStrides, std::uint8_t* data)
{
    if ((srcShapes[0] == 1) && (srcStrides.size() == 4) && (srcShapes.size() == 4)) {
        if ((srcStrides[3] == 1) && (srcStrides[2] == srcShapes[3]) && (srcStrides[1] == (srcShapes[2] * srcShapes[3]))) {
            // No depadding needed
            return;
        }
    }

    if (srcShapes.size() == 4 && srcStrides.size() == 4) {
        return fastDepad_4d(srcShapes, srcStrides, data);
    }
    else {
        std::vector<std::int32_t> srcIndexes(srcShapes.size(), 0);
        uint32_t dstIdx = 0;
        uint32_t srcOffset = 0;

        while (1) {
            data[dstIdx++] = data[srcOffset];

            int j;
            for (j = static_cast<int>(srcShapes.size()) - 1; j >= 0; j--) {
                srcIndexes[j]++;
                srcOffset += srcStrides.at(j);

                if (srcIndexes[j] < srcShapes[j])
                    break; // from for

                srcOffset -= srcIndexes[j] * srcStrides.at(j);
                srcIndexes[j] = 0;
            }
            if (j < 0)
                break; // from while
        }
    }
}
