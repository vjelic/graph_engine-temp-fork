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


addSnlTest( test_concurrency ${CMAKE_CURRENT_SOURCE_DIR}/concurrency_unittest.cpp )

# Add required model test files to build tree for testing
file(GLOB TEST_FILES ${PROJECT_SOURCE_DIR}/tests/models/convolution/*)
file(COPY ${TEST_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/tests/models/xclbins/the.xclbin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

