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

namespace cpu_support {

struct CPUSupport
{
  bool avx = false;
  bool avx2 = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512vbmi2 = false;
  bool avx512vl = false;
};

extern const CPUSupport cpuSupport;

} // cpu_support
