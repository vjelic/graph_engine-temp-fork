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


enable_testing()
include(CTest)
include(GoogleTest)

find_package(GTest REQUIRED)

function( addTest target test )
  add_executable(
    ${target}
    ${test}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${GTest_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/src
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/external/concurrentqueue
    ${PROJECT_SOURCE_DIR}/tests/utility
    ${XRT_INCLUDE_DIRS}
    )
  target_link_libraries(
    ${target}
    PRIVATE
    ${PROJECT_NAME}
    vart::runner
    GTest::GTest
    GTest::Main
    )
  if(NOT MSVC)
    gtest_discover_tests(
      ${target}
      )
  endif()
endfunction()

function( addSnlTest target test )
  add_executable(
    ${target}
    ${test}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${GTest_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/src
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/external/concurrentqueue
    ${PROJECT_SOURCE_DIR}/tests/utility
    ${XRT_INCLUDE_DIRS}
    )
  target_link_libraries(
    ${target}
    PRIVATE
    ${PROJECT_NAME}
    vart::runner
    GTest::GTest
    GTest::Main
    XRT::xrt_coreutil
    )
  add_test(
    NAME
    ${target}
    COMMAND
    bash -c "(run_simnow.sh &); ./${target}"
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

add_subdirectory(aksbuffers)
add_subdirectory(buffers)
add_subdirectory(engine)
add_subdirectory(kernel_runner)
add_subdirectory(graph_runner)
add_subdirectory(vart_runner)
add_subdirectory(pad)
add_subdirectory(convolution)
add_subdirectory(dpurunner)
add_subdirectory(nohwcontext)
add_subdirectory(twooutput)
add_subdirectory(dwc)
add_subdirectory(createio)
add_subdirectory(float)
add_subdirectory(facedetect)

option(LONG_TESTS "Build Long Test Cases" OFF)

if(${LONG_TESTS})
  add_subdirectory(padconv)
  add_subdirectory(resnet50)
  add_subdirectory(eyegaze)
endif()

option(DISABLED_TESTS "Build Disabled Test Cases" OFF)
if(${DISABLED_TESTS})
  add_subdirectory(concurrency)
  add_subdirectory(multi_hwcontext)
endif()

