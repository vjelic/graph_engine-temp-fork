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


cmake_minimum_required(VERSION 3.9)
project(graph-engine VERSION 1.0.0)
option(BUILD_TEST "build test or not" "ON")

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
  if(BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  endif(BUILD_SHARED_LIBS)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mavx512vl -mavx512vbmi")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(kernel_runners)
include(op_runners)

find_package(Boost REQUIRED)
include_directories(
  ${Boost_INCLUDE_DIRS}
  )
find_package(XRT REQUIRED)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  find_package(unilog)
  find_package(xir)
  if(MSVC)
    find_package(trace-logging QUIET)
	  if(trace-logging_FOUND)
      find_package(vart COMPONENTS util trace runner REQUIRED)
	  else(trace-logging_FOUND)
	    find_package(vart COMPONENTS util trace runner trace-logging REQUIRED)
	  endif()
  else()
    find_package(vart COMPONENTS util trace runner REQUIRED)
  endif()
endif()

add_subdirectory(src)

#option(BUILD_TESTS "Build Tests" ON)
#if(BUILD_TESTS)
#  add_subdirectory(tests)
#endif()

find_package(Doxygen)
if(${Doxygen_FOUND})
  add_subdirectory(docs)
endif()
