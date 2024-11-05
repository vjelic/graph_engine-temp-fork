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

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <immintrin.h>

#include "graph-engine/utils/CpuSupport.hpp"

namespace {

inline void fastPad_6to8_avx512(const uint8_t* input, uint8_t* output,
  std::size_t numIters, std::int32_t stride_n, std::int32_t stride_h,
  const std::array<std::pair<std::int32_t, std::int32_t>, 4>& dstSlices) {


  assert(dstSlices[2].first == 0);
  assert(dstSlices[3].first == 0);

  numIters *= dstSlices[1].second;
  const std::size_t numLoop = numIters / 8;
  const std::size_t remaining = numIters % 8;

    __m512i index_table = _mm512_set_epi8(
        0, 0, 47, 46, 45, 44, 43, 42,
        0, 0, 41, 40, 39, 38, 37, 36,
        0, 0, 35, 34, 33, 32, 31, 30,
        0, 0, 29, 28, 27, 26, 25, 24,
        0, 0, 23, 22, 21, 20, 19, 18,
        0, 0, 17, 16, 15, 14, 13, 12,
        0, 0, 11, 10, 9, 8, 7, 6,
        0, 0, 5, 4, 3, 2, 1, 0
    );

  for (auto i = 0u; i < numLoop; i++) {
    __m512i input_vec = _mm512_loadu_si512(input);
    __m512i result_vec = _mm512_maskz_permutexvar_epi8(0x3F3F3F3F3F3F3F3F, index_table, input_vec);
    _mm512_storeu_si512(output, input_vec);

    output+=64;
    input+=48;
  }

  for (auto i = 0u; i < remaining; i++) {
    *output++ = *input++;
    *output++ = *input++;
    *output++ = *input++;
    *output++ = *input++;
    *output++ = *input++;
    *output++ = *input++;
    *output++ = 0;
    *output++ = 0;
  }
}

inline void fastPad_3to4_avx512(const std::uint8_t* input, std::uint8_t* output,
  std::size_t numIters, std::int32_t stride_n, std::int32_t stride_h,
  const std::array<std::pair<std::int32_t, std::int32_t>, 4>& dstSlices)
{
  // NOTE This method works on 256-bit vectors but still requires AVX 512
  //      because it uses the _mm256_permutexvar_epi8 instruction to allow
  //      inter-lane shuffling.

  assert(dstSlices[2].first == 0);
  assert(dstSlices[3].first == 0);
  assert(dstSlices[3].second == 3);

  const std::size_t numLoop = numIters / 8;
  const std::size_t remainder = numIters % 8;

  const unsigned int numOuterIters = (dstSlices[0].second - dstSlices[0].first) * (dstSlices[1].second - dstSlices[1].first);
  unsigned int i = 0;

  for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
    for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
      const bool useMaskLoad = (++i == numOuterIters);

      uint8_t* outPtr = &output[on * stride_n + oh * stride_h];

      std::size_t unrolledIterations = (numLoop + 3) / 4;

      __m256i in;

      // NOTE If it is OK to align with 'garbage' data, the _mm256_and_si256 instruction
      //      is not strictly needed. It makes the input data cleaner, though, and might
      //      prevent confusion in debugging scenarios.

#define LOAD_FULL   in = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input))
#define LOAD_MASKED in = _mm256_maskload_epi32(reinterpret_cast<const int*>(input),   \
                            _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1))

#define PAD_STEP(LOADER)  do {                                                        \
            LOADER;                                                                   \
            in = _mm256_maskz_permutexvar_epi8(0x77777777, _mm256_set_epi8(           \
                 0, 23, 22, 21, 0, 20, 19, 18, 0, 17, 16, 15, 0, 14, 13, 12,          \
                 0, 11, 10,  9, 0,  8,  7,  6, 0,  5,  4,  3, 0,  2,  1,  0), in);    \
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(outPtr), in);             \
            input += 8 * 3;                                                           \
            outPtr += 8 * 4;                                                          \
          } while (false)

      // TODO This way of unrolling via Duff's device might not be needed. Double-check
      //      performance with and without unrolling.
      if (useMaskLoad) {
        switch (numLoop % 4) {
        case 0: do {  PAD_STEP(LOAD_MASKED);
        case 3:       PAD_STEP(LOAD_MASKED);
        case 2:       PAD_STEP(LOAD_MASKED);
        case 1:       PAD_STEP(LOAD_MASKED);
              } while (--unrolledIterations > 0);
        }
      } else {
        switch (numLoop % 4) {
        case 0: do {  PAD_STEP(LOAD_FULL);
        case 3:       PAD_STEP(LOAD_FULL);
        case 2:       PAD_STEP(LOAD_FULL);
        case 1:       PAD_STEP(LOAD_FULL);
              } while (--unrolledIterations > 0);
        }
      }

#undef PAD_STEP
#undef LOAD_MASKED
#undef LOAD_FULL

      // Process remainder
      for (std::size_t i = 0; i < remainder; ++i) {
        *outPtr++ = *input++;
        *outPtr++ = *input++;
        *outPtr++ = *input++;
        *outPtr++ = 0;
      }
    }
  }
}

inline void fastPad_3to4_avx(const std::uint8_t *input, std::uint8_t *output,
  std::size_t outputSize, std::int32_t stride_n, std::int32_t stride_h,
  const std::array<std::pair<std::int32_t, std::int32_t>, 4> &dstSlices)
{
  assert(dstSlices[2].first == 0);
  assert(dstSlices[3].first == 0);
  assert(dstSlices[3].second == 3);

  if (dstSlices[0].first != 0 || dstSlices[1].first != 0
    || (4 * dstSlices[2].second)  != stride_h
    || (4 * dstSlices[2].second * dstSlices[1].second) != stride_n) {
    std::memset(output, 0, outputSize);
  }

  const std::size_t numIters = (dstSlices[3].second * dstSlices[2].second) / 3;

  if (numIters >= 32 && ((numIters % 32) < 16 || numIters >= 64)) {
    if (cpu_support::cpuSupport.avx512vbmi2 && cpu_support::cpuSupport.avx2) {
      fastPad_3to4_avx512(input, output, numIters, stride_n, stride_h, dstSlices);
      return;
    }
  }

  const std::size_t numLoop = numIters / 4;
  const std::size_t remainder = numIters % 4;

  const unsigned int numOuterIters = (dstSlices[0].second - dstSlices[0].first) * (dstSlices[1].second - dstSlices[1].first);
  unsigned int i = 0;

  for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
    for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
      const bool useMaskLoad = (++i == numOuterIters);

      std::uint8_t *outPtr = &output[on * stride_n + oh * stride_h];

      std::size_t unrolledIterations = (numLoop + 3) / 4;

      __m128i vec;

#define LOAD_FULL     vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input))
#define LOAD_MASKED   vec = _mm_maskload_epi32(reinterpret_cast<const int *>(input),  \
                              _mm_set_epi32(0, -1, -1, -1))

#define PAD_STEP(LOADER)  do {                                                        \
            LOADER;                                                                   \
            vec = _mm_shuffle_epi8(vec, _mm_set_epi8(                                 \
                   -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0));            \
            _mm_storeu_si128(reinterpret_cast<__m128i *>(outPtr), vec);               \
            input += 4 * 3;                                                           \
            outPtr += 4 * 4;                                                          \
          } while (false)

      // TODO This way of unrolling via Duff's device might not be needed. Double-check
      //      performance with and without unrolling.
      if (useMaskLoad) {
        switch (numLoop % 4) {
        case 0: do {  PAD_STEP(LOAD_MASKED);
        case 3:       PAD_STEP(LOAD_MASKED);
        case 2:       PAD_STEP(LOAD_MASKED);
        case 1:       PAD_STEP(LOAD_MASKED);
              } while (--unrolledIterations > 0);
        }
      } else {
        switch (numLoop % 4) {
        case 0: do {  PAD_STEP(LOAD_FULL);
        case 3:       PAD_STEP(LOAD_FULL);
        case 2:       PAD_STEP(LOAD_FULL);
        case 1:       PAD_STEP(LOAD_FULL);
              } while (--unrolledIterations > 0);
        }
      }

#undef PAD_STEP
#undef LOAD_MASKED
#undef LOAD_FULL

      // Process remainder
      for (std::size_t i = 0; i < remainder; ++i) {
        *outPtr++ = *input++;
        *outPtr++ = *input++;
        *outPtr++ = *input++;
        *outPtr++ = 0;
      }
    }
  }
}

inline void fastPad_2to4_avx(const std::uint8_t* input, std::uint8_t* output,
    std::size_t outputSize, std::int32_t stride_n, std::int32_t stride_h,
    const std::array<std::pair<std::int32_t, std::int32_t>, 4>& dstSlices)
{
  assert(dstSlices[2].first == 0);
  assert(dstSlices[3].first == 0);
  assert(dstSlices[3].second == 2);

  if (dstSlices[0].first != 0 || dstSlices[1].first != 0
    || (4 * dstSlices[2].second) != stride_h
    || (4 * dstSlices[2].second * dstSlices[1].second) != stride_n) {
    std::memset(output, 0, outputSize);
  }

  const std::size_t numIters = (dstSlices[3].second * dstSlices[2].second);

  const std::size_t numLoop = numIters / 16;
  const std::size_t remainder = numIters % 16;

  for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
    for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
      std::uint8_t* outPtr = &output[on * stride_n + oh * stride_h];

      std::size_t unrolledIterations = (numLoop + 3) / 4;

      __m128i in_vec;
      __m128i vec0;
      __m128i vec1;

#define PAD_STEP()  do { \
            in_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));       \
            vec0 = _mm_shuffle_epi8(in_vec, _mm_set_epi8(                             \
                    -1, -1,  7,  6, -1, -1,  5,  4, -1, -1,  3,  2, -1, -1,  1,  0)); \
            _mm_storeu_si128(reinterpret_cast<__m128i *>(outPtr), vec0);              \
            outPtr += 4 * 4;                                                          \
            vec1 = _mm_shuffle_epi8(in_vec, _mm_set_epi8(                             \
                    -1, -1, 15, 14, -1, -1, 13, 12, -1, -1, 11, 10, -1, -1,  9,  8)); \
            _mm_storeu_si128(reinterpret_cast<__m128i *>(outPtr), vec1);              \
            outPtr += 4 * 4;                                                          \
            input += 16;                                                              \
          } while (false)

      // TODO This way of unrolling via Duff's device might not be needed. Double-check
      //      performance with and without unrolling.
      switch (numLoop % 4) {
      case 0: do {
          PAD_STEP();
      case 3:       PAD_STEP();
      case 2:       PAD_STEP();
      case 1:       PAD_STEP();
            } while (--unrolledIterations > 0);
      }

#undef PAD_STEP

      // Process remainder
      for (std::size_t i = 0; i < remainder; ++i) {
          *outPtr++ = *input++;
          *outPtr++ = 0;
          *outPtr++ = 0;
          *outPtr++ = 0;
      }
    }
  }
}

inline void fastPad_2to8_avx(const std::uint8_t* input, std::uint8_t* output,
    std::size_t outputSize, std::int32_t stride_n, std::int32_t stride_h,
    const std::array<std::pair<std::int32_t, std::int32_t>, 4>& dstSlices)
{
    assert(dstSlices[2].first == 0);
    assert(dstSlices[3].first == 0);
    assert(dstSlices[3].second == 2);

    if (dstSlices[0].first != 0 || dstSlices[1].first != 0
        || (8 * dstSlices[2].second) != stride_h
        || (8 * dstSlices[2].second * dstSlices[1].second) != stride_n) {
        std::memset(output, 0, outputSize);
    }

    const std::size_t numIters = (dstSlices[3].second * dstSlices[2].second);

    const std::size_t numLoop = numIters / 32;
    const std::size_t remainder = (numIters % 32) / 2;

    for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
        for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
            std::uint8_t* outPtr = &output[on * stride_n + oh * stride_h];

            std::size_t unrolledIterations = (numLoop + 3) / 4;

            __m256i in_vec;
            __m256i vec0;
            __m256i vec1;
            __m256i vec2;
            __m256i vec3;

#define PAD_STEP()  do { \
        in_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));              \
        vec0 = _mm256_maskz_permutexvar_epi8(0x03030303, _mm256_set_epi8(                   \
                -1, -1, -1, -1, -1, -1, 7, 6, -1, -1, -1, -1, -1, -1, 5, 4,                 \
                -1, -1, -1, -1, -1, -1, 3, 2, -1, -1, -1, -1, -1, -1, 1, 0), in_vec);       \
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(outPtr), vec0);                     \
        outPtr += 4 * 8;                                                                    \
        vec1 = _mm256_maskz_permutexvar_epi8(0x03030303, _mm256_set_epi8(                   \
                -1, -1, -1, -1, -1, -1, 15, 14, -1, -1, -1, -1, -1, -1, 13, 12,             \
                -1, -1, -1, -1, -1, -1, 11, 10, -1, -1, -1, -1, -1, -1, 9, 8), in_vec);     \
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(outPtr), vec1);                     \
        outPtr += 4 * 8;                                                                    \
        vec2 = _mm256_maskz_permutexvar_epi8(0x03030303, _mm256_set_epi8(                   \
                -1, -1, -1, -1, -1, -1, 23, 22, -1, -1, -1, -1, -1, -1, 21, 20,             \
                -1, -1, -1, -1, -1, -1, 19, 18, -1, -1, -1, -1, -1, -1, 17, 16), in_vec);   \
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(outPtr), vec2);                     \
        outPtr += 4 * 8;                                                                    \
        vec3 = _mm256_maskz_permutexvar_epi8(0x03030303, _mm256_set_epi8(                   \
                -1, -1, -1, -1, -1, -1, 31, 30, -1, -1, -1, -1, -1, -1, 29, 28,             \
                -1, -1, -1, -1, -1, -1, 27, 26, -1, -1, -1, -1, -1, -1, 25, 24), in_vec);   \
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(outPtr), vec3);                     \
        outPtr += 4 * 8;                                                                    \
        input += 4 * 8;                                                                     \
      } while (false)

            // TODO This way of unrolling via Duff's device might not be needed. Double-check
            //      performance with and without unrolling.
            switch (numLoop % 4) {
            case 0: do {
                PAD_STEP();
            case 3:       PAD_STEP();
            case 2:       PAD_STEP();
            case 1:       PAD_STEP();
                } while (--unrolledIterations > 0);
            }

#undef PAD_STEP

            // Process remainder
            for (std::size_t i = 0; i < remainder; ++i) {
                *outPtr++ = *input++;
                *outPtr++ = *input++;
                *outPtr++ = 0;
                *outPtr++ = 0;
                *outPtr++ = 0;
                *outPtr++ = 0;
                *outPtr++ = 0;
                *outPtr++ = 0;
            }
        }
    }
}

inline void fastPad_1to4_avx(const std::uint8_t *input, std::uint8_t *output,
    std::size_t outputSize, std::int32_t stride_n, std::int32_t stride_h,
    const std::array<std::pair<std::int32_t, std::int32_t>, 4> &dstSlices)
{
  assert(dstSlices[2].first == 0);
  assert(dstSlices[3].first == 0);
  assert(dstSlices[3].second == 1);

  if (dstSlices[0].first != 0 || dstSlices[1].first != 0
    || (4 * dstSlices[2].second) != stride_h
    || (4 * dstSlices[2].second * dstSlices[1].second) != stride_n) {
    std::memset(output, 0, outputSize);
  }

  const std::size_t numIters = (dstSlices[3].second * dstSlices[2].second);

  const std::size_t numLoop = numIters / 16;
  const std::size_t remainder = numIters % 16;

  for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
    for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
      std::uint8_t* outPtr = &output[on * stride_n + oh * stride_h];

      std::size_t unrolledIterations = (numLoop + 3) / 4;

      __m128i in_vec;
      __m128i vec0;
      __m128i vec1;

#define PAD_STEP()  do { \
            in_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));       \
            vec0 = _mm_shuffle_epi8(in_vec, _mm_set_epi8(                             \
                    -1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));     \
            _mm_storeu_si128(reinterpret_cast<__m128i *>(outPtr), vec0);              \
            outPtr += 4 * 4;                                                          \
            vec1 = _mm_shuffle_epi8(in_vec, _mm_set_epi8(                             \
                    -1, -1, -1, 7, -1, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1, 4));     \
            _mm_storeu_si128(reinterpret_cast<__m128i *>(outPtr), vec1);              \
            outPtr += 4 * 4;                                                          \
            vec0 = _mm_shuffle_epi8(in_vec, _mm_set_epi8(                             \
                    -1, -1, -1, 11, -1, -1, -1, 10, -1, -1, -1, 9, -1, -1, -1, 8));   \
            _mm_storeu_si128(reinterpret_cast<__m128i *>(outPtr), vec0);              \
            outPtr += 4 * 4;                                                          \
            vec1 = _mm_shuffle_epi8(in_vec, _mm_set_epi8(                             \
                    -1, -1, -1, 15, -1, -1, -1, 14, -1, -1, -1, 13, -1, -1, -1, 12)); \
            _mm_storeu_si128(reinterpret_cast<__m128i *>(outPtr), vec1);              \
            outPtr += 4 * 4;                                                          \
            input += 16;                                                              \
          } while (false)

      // TODO This way of unrolling via Duff's device might not be needed. Double-check
      //      performance with and without unrolling.
      switch (numLoop % 4) {
      case 0: do {  PAD_STEP();
      case 3:       PAD_STEP();
      case 2:       PAD_STEP();
      case 1:       PAD_STEP();
          } while (--unrolledIterations > 0);
      }

#undef PAD_STEP

      // Process remainder
      for (std::size_t i = 0; i < remainder; ++i) {
        *outPtr++ = *input++;
        *outPtr++ = 0;
        *outPtr++ = 0;
        *outPtr++ = 0;
      }
    }
  }
}

} // anon ns

void fastPad(const std::uint8_t *input, std::uint8_t *output,
  const std::array<std::int32_t, 4> &paddedShape,
  const std::array<std::pair<std::int32_t, std::int32_t>, 4> &dstSlices)
{
  if (paddedShape[0] != 1) {
    throw std::invalid_argument("paddedShape[0] != 1 is not supported!");
  }

  const std::int32_t stride_w = paddedShape[3];
  const std::int32_t stride_h = paddedShape[2] * stride_w;
  const std::int32_t stride_n = paddedShape[1] * stride_h;

  const std::size_t inputSize = (dstSlices[0].second - dstSlices[0].first)
    * (dstSlices[1].second - dstSlices[1].first)
    * (dstSlices[2].second - dstSlices[2].first)
    * (dstSlices[3].second - dstSlices[3].first);
  const std::size_t outputSize = sizeof(std::uint8_t) * stride_n;

  // Trivial case - no padding needed at all
  if (outputSize == inputSize) {
    std::memcpy(output, input, outputSize);
    return;
  }

  const std::size_t numIterationsInner = dstSlices[3].second - dstSlices[3].first;

  // AIE network implementations often require pixels to be 32-bit aligned, while input data often is RGB/BGR.
  // Detect the situation where padding from RGB/BGR to RGBX/BGRX (3 -> 4 bytes in inner loop) is needed and
  // run the optimized implementation.
  // Optimized versions assume there's at least one AVX pad step, hence check if we have at least the required
  // number of iterations (12 for 3->4, 16 for 1->4).
  if (cpu_support::cpuSupport.avx && numIterationsInner == 3 && stride_w == 4 && dstSlices[2].first == 0 && dstSlices[3].first == 0
      && (dstSlices[3].second * dstSlices[2].second) >= 12) {
    fastPad_3to4_avx(input, output, outputSize, stride_n, stride_h, dstSlices);
  }

  // 1-to-4 padding (e.g. MEP C1)
  else if (cpu_support::cpuSupport.avx && numIterationsInner == 1 && stride_w == 4 && dstSlices[2].first == 0 && dstSlices[3].first == 0
      && (dstSlices[3].second * dstSlices[2].second) >= 16) {
    fastPad_1to4_avx(input, output, outputSize, stride_n, stride_h, dstSlices);
  }

  // 2-to-4 padding (e.g. MEP F1)
  else if (cpu_support::cpuSupport.avx && numIterationsInner == 2 && stride_w == 4 && dstSlices[2].first == 0 && dstSlices[3].first == 0
      && (dstSlices[3].second * dstSlices[2].second) >= 16) {
    fastPad_2to4_avx(input, output, outputSize, stride_n, stride_h, dstSlices);
  }

  // 2-to-8 padding (e.g. MEP F2 2x4)
  else if (cpu_support::cpuSupport.avx && cpu_support::cpuSupport.avx512vbmi2 && cpu_support::cpuSupport.avx2
      && numIterationsInner == 2 && stride_w == 8 && dstSlices[2].first == 0 && dstSlices[3].first == 0
      && (dstSlices[3].second * dstSlices[2].second) >= 32) {
    fastPad_2to8_avx(input, output, outputSize, stride_n, stride_h, dstSlices);
  }

  // Otherwise, run the generic, but less efficient implementation.
  else {
    if (dstSlices[0].first == 0 && dstSlices[1].first == 0 && dstSlices[2].first == 0 && dstSlices[3].first == 0
        && stride_w == dstSlices[3].second
        && stride_h == dstSlices[2].second * stride_w) { // Tail padding only
      std::size_t tailPaddingBytes = stride_n - static_cast<std::size_t>(dstSlices[1].second) * stride_h;
      std::memset(output + outputSize - tailPaddingBytes, 0, tailPaddingBytes);
    }
    else {
      std::memset(output, 0, outputSize);
    }

    const std::size_t innerSize = static_cast<std::size_t>(dstSlices[3].second) - dstSlices[1].first;

    if (stride_w == dstSlices[3].second && stride_h == dstSlices[2].second * stride_w && dstSlices[3].first == 0 && dstSlices[2].first == 0) { // on loop only
      for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
        std::uint8_t* outputPtr = &output[on * stride_n + dstSlices[1].first * stride_h];
        std::size_t numBytes = innerSize * dstSlices[2].second * (static_cast<std::size_t>(dstSlices[1].second) - dstSlices[1].first);

        std::memcpy(outputPtr, input, numBytes);
        outputPtr += numBytes;
        input += numBytes;
      }
    }
    else if (stride_w == dstSlices[3].second && dstSlices[3].first == 0) { // on + oh loops only
      for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
        for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
          std::uint8_t* outputPtr = &output[on * stride_n + oh * stride_h + dstSlices[2].first * stride_w];
          std::size_t numBytes = innerSize * (static_cast<std::size_t>(dstSlices[2].second) - dstSlices[2].first);

          std::memcpy(outputPtr, input, numBytes);
          outputPtr += numBytes;
          input += numBytes;
        }
      }
    }
    else { // generic case
      if (innerSize > 4) { // Experimentally determined break-even point for memcpy-based implementation
        for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
          for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
            for (int ow = dstSlices[2].first; ow < dstSlices[2].second; ++ow) {
              std::uint8_t* outputPtr = &output[on * stride_n + oh * stride_h + ow * stride_w + dstSlices[3].first];

              std::memcpy(outputPtr, input, innerSize);
              outputPtr += innerSize;
              input += innerSize;
            }
          }
        }
      }
      else {
      for (int on = dstSlices[0].first; on < dstSlices[0].second; ++on) {
          for (int oh = dstSlices[1].first; oh < dstSlices[1].second; ++oh) {
            for (int ow = dstSlices[2].first; ow < dstSlices[2].second; ++ow) {
              std::uint8_t* outputPtr = &output[on * stride_n + oh * stride_h + ow * stride_w + dstSlices[3].first];

              for (int oc = dstSlices[3].first; oc < dstSlices[3].second; ++oc) {
                *outputPtr++ = *input++;
              }
            }
          }
        }
      }
    }
  }
}
