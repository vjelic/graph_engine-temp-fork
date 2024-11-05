// Copyright 2024 Advanced Micro Devices Inc.
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

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <immintrin.h>

#include "graph-engine/utils/CpuSupport.hpp"

namespace {

bool isAligned(void* data, int alignment) {
    return ((uintptr_t)data & (alignment - 1)) == 0;
}  

// Convert float to int8 with rounding
template <typename T> inline T rounder(float data) {
    static const int data_max = std::numeric_limits<T>::max();
    static const int data_min = std::numeric_limits<T>::min();
    T rlt = 0;
    if (data > data_max) {
        rlt = data_max;
    }
    else if (data < data_min) {
        rlt = data_min;
    }
    else if ((data - floor(data)) == 0.5) {
        rlt = std::round(data * 0.5f) * 2.0f;
    }
    else {
        rlt = static_cast<T>(round(data));
    }
    return rlt;
}

#ifdef _WIN32 
inline void float2fix_avx512(const float *src, std::int8_t *dst, std::size_t num_elements, const float scale)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR = VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  //const __m512 round_vector = _mm512_set1_ps(0.5f);

  for (std::size_t i = 0; i < num_iter; ++i) {
    __m512 in = _mm512_loadu_ps(src);
    in = _mm512_roundscale_ps(_mm512_mul_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT);
    __m128i int8s = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(in));
    _mm_storeu_epi8(dst, int8s);

    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const float& src) { return rounder<std::int8_t>(scale * src); });
  }
}
#endif
inline void fix2float_avx512(const std::int8_t *src, float *dst, std::size_t num_elements, const float scale)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m128);
  constexpr std::size_t INT8_SIZE_BYTES = sizeof(std::int8_t);
  constexpr std::size_t INT8S_PER_VECTOR = VECTOR_SIZE_BYTES / INT8_SIZE_BYTES;

  const std::size_t num_iter = num_elements / INT8S_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT8S_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);

  for (std::size_t i = 0; i < num_iter; ++i) {
    __m128i int8s = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
    __m512 floats = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int8s));
    floats = _mm512_mul_ps(floats, scale_vector);
    _mm512_storeu_ps(dst, floats);

    src += INT8S_PER_VECTOR;
    dst += INT8S_PER_VECTOR;
  }

  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const std::int8_t& src) { return scale * src; });
  }
}

} // anon ns

void float2fix(const float *src, std::int8_t *dst, std::size_t num_elements, const float scale)
{
#ifdef _WIN32  //TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f
    && cpu_support::cpuSupport.avx512bw
    && cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    float2fix_avx512(src, dst, num_elements, scale);
  } else {
    std::transform(src, src + num_elements, dst, [&] (const float &src) { return rounder<std::int8_t>(scale * src); });
  }
#else
  std::transform(src, src + num_elements, dst, [&] (const float &src) { return rounder<std::int8_t>(scale * src); });
#endif
}

#ifdef _WIN32 
inline void float2fix_avx512_stream(const float *src, __m128i *dst, std::size_t num_elements, const float scale, const int8_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR = VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  //const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m128i zero_point_vector =  _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);      
    __m512 in = _mm512_loadu_ps(src);
    in = _mm512_roundscale_ps(_mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT);
    __m128i int8s = _mm_add_epi8(_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(in)), zero_point_vector);
    // _mm_storeu_epi8(dst, int8s);
    _mm_stream_si128((__m128i*)dst, int8s);    
    src += FLOATS_PER_VECTOR;
    dst += 1;
  }
  _mm_sfence();
  
  if (remainder > 0) {
    std::transform(src, src + remainder, (std::int8_t*)dst, [&] (const float &src) { return rounder<std::int8_t>(src/scale)+zero_point; });
  }
}
#endif

#ifdef _WIN32 
inline void float2fix_avx512(const float *src, std::int8_t *dst, std::size_t num_elements, const float scale, const int8_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR = VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  //const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m128i zero_point_vector =  _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);      
    __m512 in = _mm512_loadu_ps(src);
    in = _mm512_roundscale_ps(_mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT);
    __m128i int8s = _mm_add_epi8(_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(in)), zero_point_vector);
    _mm_storeu_epi8(dst, int8s);
    // _mm_stream_si128((__m128i*)dst, int8s);    
    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }
  
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&] (const float &src) { return rounder<std::int8_t>(src/scale)+zero_point; });
  }
}
#endif

//To-do: MNDBG: confirm if const needed
void float2fix(const float *src, std::int8_t *dst, std::size_t num_elements, const float scale, const int zero_point)
{
#ifdef _WIN32  //TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f
    && cpu_support::cpuSupport.avx512bw
    && cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if(isAligned((void*)dst, sizeof(__m128i))) {
      float2fix_avx512_stream(src, (__m128i*)dst, num_elements, scale, zero_point);
    } else {
      float2fix_avx512(src, dst, num_elements, scale, zero_point);   
    }
  } else {
    std::transform(src, src + num_elements, dst, [&] (const float &src) { return rounder<std::int8_t>(src/scale)+zero_point; });
  }
#else
  std::transform(src, src + num_elements, dst, [&] (const float &src) { return rounder<std::int8_t>(src/scale)+zero_point; });
#endif
}

#ifdef _WIN32 
inline void float2int16_avx512_stream(const float *src, __m256i *dst, std::size_t num_elements, const float scale, const int16_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR = VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m256i zero_point_vector =  _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);  
    __m512 in = _mm512_loadu_ps(src);
    __m256i int16s = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT)));
    int16s = _mm256_add_epi16(int16s, zero_point_vector);
    _mm256_stream_si256(dst, int16s);   
    src += FLOATS_PER_VECTOR;
    dst += 1;
  }
  _mm_sfence();
  if (remainder > 0) {
    std::transform(src, src + remainder, (std::int16_t*)dst, [&](const float& src) { return rounder<std::int16_t>( ((src/scale)+zero_point)); });
  }
}
#endif

#ifdef _WIN32 
inline void float2int16_avx512(const float *src, std::int16_t *dst, std::size_t num_elements, const float scale, const int16_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m512);
  constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);
  constexpr std::size_t FLOATS_PER_VECTOR = VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

  static_assert(FLOAT_SIZE_BYTES == 4, "Unexpected float size!");

  const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * FLOATS_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  //const __m512 round_vector = _mm512_set1_ps(0.5f);
  const __m256i zero_point_vector =  _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 64, _MM_HINT_T0);  
    __m512 in = _mm512_loadu_ps(src);
    __m256i int16s = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_div_ps(in, scale_vector), _MM_FROUND_TO_NEAREST_INT)));
    int16s = _mm256_add_epi16(int16s, zero_point_vector);
    _mm256_storeu_si256((__m256i*)dst, int16s);    
    src += FLOATS_PER_VECTOR;
    dst += FLOATS_PER_VECTOR;
  }
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&](const float& src) { return rounder<std::int16_t>( ((src/scale)+zero_point)); });
  }
}
#endif

//To-do: MNDBG: confirm if const needed
void float2int16(const float *src, std::int16_t *dst, std::size_t num_elements, const float scale, const int zero_point ) 
{
#ifdef _WIN32  //TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f
    && cpu_support::cpuSupport.avx512bw
    && cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if(isAligned((void*)dst, sizeof(__m256i))) {
      float2int16_avx512_stream(src, (__m256i*)dst, num_elements, scale, zero_point);
    } else {
      float2int16_avx512(src, dst, num_elements, scale, zero_point);   
    }
  } else {
    std::transform(src, src + num_elements, dst, [&] (const float &src) { return rounder<std::int16_t>( ((src/scale)+zero_point)); });
  }
#else
  std::transform(src, src + num_elements, dst, [&] (const float &src) { return rounder<std::int16_t>( ((src/scale)+zero_point)); });
#endif
}

void fix2float(const std::int8_t *src, float *dst, std::size_t num_elements, const float scale)
{
  if (cpu_support::cpuSupport.avx512f) {
    fix2float_avx512(src, dst, num_elements, scale);
  } else {
    std::transform(src, src + num_elements, dst, [&] (const std::int8_t &src) { return scale*src; });
  }
}

#ifdef _WIN32 
inline void fix2float_avx512_stream256(const std::int8_t *src, float *dst, std::size_t num_elements, const float scale, const int8_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m128);
  constexpr std::size_t INT8_SIZE_BYTES = sizeof(int8_t);
  constexpr std::size_t INT8_PER_VECTOR = VECTOR_SIZE_BYTES / INT8_SIZE_BYTES;

  static_assert(INT8_SIZE_BYTES == 1, "Unexpected int8_t size!");
  const std::size_t num_iter = num_elements / INT8_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT8_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m128i zero_point_vector =  _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 16, _MM_HINT_T0);      
    __m128i in = _mm_loadu_epi8(src);    
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_sub_epi8(in, zero_point_vector))), scale_vector);  
    __m256 temp0 = _mm512_extractf32x8_ps(mul, 0);
    __m256 temp1 = _mm512_extractf32x8_ps(mul, 1);
    _mm256_stream_ps(dst, temp0);
    _mm256_stream_ps(dst + 8, temp1);     
    // _mm512_storeu_ps(dst, mul);
    src += INT8_PER_VECTOR;
    dst += INT8_PER_VECTOR;
  }
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&] (const std::int8_t &src) { return (std::int8_t((src-zero_point))*scale); });
  }
}
#endif

#ifdef _WIN32 
inline void fix2float_avx512(const std::int8_t *src, float *dst, std::size_t num_elements, const float scale, const int8_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m128);
  constexpr std::size_t INT8_SIZE_BYTES = sizeof(int8_t);
  constexpr std::size_t INT8_PER_VECTOR = VECTOR_SIZE_BYTES / INT8_SIZE_BYTES;

  static_assert(INT8_SIZE_BYTES == 1, "Unexpected int8_t size!");
  const std::size_t num_iter = num_elements / INT8_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT8_PER_VECTOR);

  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m128i zero_point_vector =  _mm_set1_epi8(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)src + 16, _MM_HINT_T0);      
    __m128i in = _mm_loadu_epi8(src);    
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_sub_epi8(in, zero_point_vector))), scale_vector);     
    _mm512_storeu_ps(dst, mul);
    src += INT8_PER_VECTOR;
    dst += INT8_PER_VECTOR;
  }
  
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&] (const std::int8_t &src) { return (std::int8_t((src-zero_point))*scale); });
  }
}
#endif

void fix2float(const std::int8_t *src, float *dst, std::size_t num_elements, const float scale, const int8_t zero_point)
{
#ifdef _WIN32  //TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f
    && cpu_support::cpuSupport.avx512bw
    && cpu_support::cpuSupport.avx512vl; 
  if (avxSupport) {
    if(isAligned((void*)dst, sizeof(__m256))) {
      fix2float_avx512_stream256(src, dst, num_elements, scale, zero_point);
    } else {
      fix2float_avx512(src, dst, num_elements, scale, zero_point);   
    }
  } else {
    std::transform(src, src + num_elements, dst, [&] (const std::int8_t &src) { return (std::int8_t((src-zero_point))*scale); });
  }
  #else
  std::transform(src, src + num_elements, dst, [&] (const std::int8_t &src) { return (std::int8_t((src-zero_point))*scale); });
  #endif
}

#ifdef _WIN32 
inline void int162float_avx512_stream256(const std::int16_t *src, float *dst, std::size_t num_elements, const float scale, const int16_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m256i);
  constexpr std::size_t INT16_SIZE_BYTES = sizeof(int16_t);
  constexpr std::size_t INT16_PER_VECTOR = VECTOR_SIZE_BYTES / INT16_SIZE_BYTES;

  static_assert(INT16_SIZE_BYTES == 2, "Unexpected int16_t size!");
  const std::size_t num_iter = num_elements / INT16_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT16_PER_VECTOR);
  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m256i zero_point_vector =  _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)(src + 32), _MM_HINT_T0);      
    __m256i in = _mm256_load_si256((__m256i*)src);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_sub_epi16(in, zero_point_vector))), scale_vector);
    // _mm512_storeu_ps(dst, mul);
    __m256 temp0 = _mm512_extractf32x8_ps(mul, 0);
    __m256 temp1 = _mm512_extractf32x8_ps(mul, 1);
    _mm256_stream_ps(dst, temp0);
    _mm256_stream_ps(dst + 8, temp1);     
    src += INT16_PER_VECTOR;
    dst += INT16_PER_VECTOR;
  }
  
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&] (const std::int16_t &src) { return (std::int16_t((src-zero_point))*scale); });
  }
}
#endif

#ifdef _WIN32 
inline void int162float_avx512(const std::int16_t *src, float *dst, std::size_t num_elements, const float scale, const int16_t zero_point)
{
  constexpr std::size_t VECTOR_SIZE_BYTES = sizeof(__m256i);
  constexpr std::size_t INT16_SIZE_BYTES = sizeof(int16_t);
  constexpr std::size_t INT16_PER_VECTOR = VECTOR_SIZE_BYTES / INT16_SIZE_BYTES;

  static_assert(INT16_SIZE_BYTES == 2, "Unexpected int16_t size!");
  const std::size_t num_iter = num_elements / INT16_PER_VECTOR;
  const std::size_t remainder = num_elements - (num_iter * INT16_PER_VECTOR);
  const __m512 scale_vector = _mm512_set1_ps(scale);
  const __m256i zero_point_vector =  _mm256_set1_epi16(zero_point);
  for (std::size_t i = 0; i < num_iter; ++i) {
    _mm_prefetch((const char*)(src + 32), _MM_HINT_T0);      
    __m256i in = _mm256_load_si256((__m256i*)src);
    __m512 mul = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_sub_epi16(in, zero_point_vector))), scale_vector);
    _mm512_storeu_ps(dst, mul);
  
    src += INT16_PER_VECTOR;
    dst += INT16_PER_VECTOR;
  }
  
  if (remainder > 0) {
    std::transform(src, src + remainder, dst, [&] (const std::int16_t &src) { return (std::int16_t((src-zero_point))*scale); });
  }
}
#endif

void int162float(const std::int16_t *src, float *dst, std::size_t num_elements, const float scale, const int8_t zero_point)
{
#ifdef _WIN32  //TODO _mm_storeu_epi8 is not defined on Linux
  static const bool avxSupport = cpu_support::cpuSupport.avx512f
    && cpu_support::cpuSupport.avx512bw
    && cpu_support::cpuSupport.avx512vl;
  if (avxSupport) {
    if(isAligned((void*)dst, sizeof(__m256))) {
      int162float_avx512_stream256(src, dst, num_elements, scale, zero_point);
    } else {
      int162float_avx512(src, dst, num_elements, scale, zero_point);   
    }
  } else {
    std::transform(src, src + num_elements, dst, [&] (const std::int16_t &src) { return (std::int16_t((src-zero_point))*scale); });
  }
  #else
  std::transform(src, src + num_elements, dst, [&] (const std::int16_t &src) { return (std::int16_t((src-zero_point))*scale); });
  #endif
}
