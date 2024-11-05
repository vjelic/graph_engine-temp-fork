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


set(OP_RUNNERS_DIR ${PROJECT_SOURCE_DIR}/src/graph-engine/op_runners)

# IMPROVE THIS LATER OR GIVE UP ON AUTOMATIC REGISTRATION
file(GLOB OP_RUNNER_FILES ${OP_RUNNERS_DIR}/*.hpp)

file(WRITE ${PROJECT_BINARY_DIR}/op_runners.hpp "#pragma once\n")
file(WRITE ${PROJECT_BINARY_DIR}/op_runners.cpp "// Register All Known Op Runners\n")
foreach(OP_RUNNER_FILE_ABS ${OP_RUNNER_FILES})
  get_filename_component(OP_RUNNER_FILE ${OP_RUNNER_FILE_ABS} NAME)
  get_filename_component(OP_RUNNER ${OP_RUNNER_FILE_ABS} NAME_WE)
  file(APPEND ${PROJECT_BINARY_DIR}/op_runners.hpp "#include \"graph-engine/op_runners/${OP_RUNNER_FILE}\"\n")
  file(APPEND ${PROJECT_BINARY_DIR}/op_runners.cpp "OpRunnerFactory::register_op_runner(\"${OP_RUNNER}\", OpRunner::create<${OP_RUNNER}>);\n")
endforeach()

