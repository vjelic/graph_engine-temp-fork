# Copyright 2022 Xilinx, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set(KERNEL_RUNNERS_DIR ${PROJECT_SOURCE_DIR}/src/graph-engine/kernel_runners)

# IMPROVE THIS LATER OR GIVE UP ON AUTOMATIC REGISTRATION
file(GLOB KERNEL_RUNNER_FILES ${KERNEL_RUNNERS_DIR}/*.hpp)

file(WRITE ${PROJECT_BINARY_DIR}/kernel_runners.hpp "#pragma once\n")
file(WRITE ${PROJECT_BINARY_DIR}/kernel_runners.cpp "// Register All Known Kernel Runners\n")
foreach(KERNEL_RUNNER_FILE_ABS ${KERNEL_RUNNER_FILES})
  get_filename_component(KERNEL_RUNNER_FILE ${KERNEL_RUNNER_FILE_ABS} NAME)
  get_filename_component(KERNEL_RUNNER ${KERNEL_RUNNER_FILE_ABS} NAME_WE)
  file(APPEND ${PROJECT_BINARY_DIR}/kernel_runners.hpp "#include \"graph-engine/kernel_runners/${KERNEL_RUNNER_FILE}\"\n")
  file(APPEND ${PROJECT_BINARY_DIR}/kernel_runners.cpp "KernelRunnerFactory::register_kernel_runner(\"${KERNEL_RUNNER}\", KernelRunner::create<${KERNEL_RUNNER}>);\n")
endforeach()

