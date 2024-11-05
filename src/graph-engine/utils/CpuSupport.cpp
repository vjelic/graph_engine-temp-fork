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

#include "graph-engine/utils/CpuSupport.hpp"

#ifdef _WIN32
# include <intrin.h>
#endif

#ifdef __GNUC__
# include <cpuid.h>
#endif

namespace cpu_support {

static __inline void cpuidex_wrapper(int __cpuid_info[4], int __leaf, int __subleaf)
{
#ifdef __GNUC__
  __cpuid_count(__leaf, __subleaf, __cpuid_info[0], __cpuid_info[1], __cpuid_info[2], __cpuid_info[3]);
#else
  __cpuidex(__cpuid_info, __leaf, __subleaf);
#endif
}

CPUSupport getCPUSupport()
{
  // ebx, ecx, edx in cpuid[1..3], see doc cpp/intrinsics/cpuid-cpuidex

  CPUSupport res;
  int cpuid[4];

  // Detect avx
  cpuidex_wrapper(cpuid, 1, 0);
  res.avx = (cpuid[2] & (1 << 28)) != 0;

  // Detect avx2/avx512
  cpuidex_wrapper(cpuid, 0, 0);
  if (cpuid[0] >= 7) {
    cpuidex_wrapper(cpuid, 7, 0);

    res.avx2 = (cpuid[1] & (1 << 5)) != 0;
    res.avx512f = (cpuid[1] & (1 << 16)) != 0;
    res.avx512bw = (cpuid[1] & (1 << 30)) != 0;
    res.avx512vbmi2 = (cpuid[2] & (1 << 6)) != 0;
    res.avx512vl = (cpuid[1] & (1 << 31)) != 0;
  }

  return res;
}

const CPUSupport cpuSupport = getCPUSupport();

} // cpu_support
