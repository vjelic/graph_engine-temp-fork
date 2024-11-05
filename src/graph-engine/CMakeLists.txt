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

option(VART_IN_BUILD_TREE "vart module is in build tree" OFF)

if(VART_IN_BUILD_TREE)
   add_compile_definitions(VART_IN_BUILD_TREE)
endif()

if(MSVC)
set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_runner.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel_runner_factory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/op_runner_factory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/delayed_dll_path.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/CpuSupport.cpp
  )
else()
set(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_runner.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel_runner_factory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/op_runner_factory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/CpuSupport.cpp
  )
endif()

add_library(
  ${PROJECT_NAME}
  ${SRCS}
  )

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${XRT_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/vart/trace/include
  ${CMAKE_SOURCE_DIR}/vart/util/include
  ${CMAKE_SOURCE_DIR}/xir/include
  ${PROJECT_SOURCE_DIR}/external/concurrentqueue
  ${PROJECT_BINARY_DIR}/src
  ${PROJECT_BINARY_DIR}
  )

if(BUILD_SHARED_LIBS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DGRAPH_ENGINE_USE_DLL=1)
else(BUILD_SHARED_LIBS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DGRAPH_ENGINE_USE_DLL=0)
endif(BUILD_SHARED_LIBS)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
  vart::util
  vart::trace
)

if(MSVC)
  target_link_options(${PROJECT_NAME} PRIVATE "/DELAYLOAD:xrt_coreutil.dll")
  if(BUILD_SHARED_LIBS)
  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    CfgMgr32.lib
    vart::runner
    PRIVATE
    vart-trace
    delayimp
    vart::trace-logging
    ${XRT_COREUTIL_LIBRARIES}
    )
  else(BUILD_SHARED_LIBS)
  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    CfgMgr32.lib
    vart::runner
    PRIVATE
    delayimp
    vart::trace-logging
    ${XRT_COREUTIL_LIBRARIES}
    )
  endif(BUILD_SHARED_LIBS)
else(MSVC)
  if(BUILD_SHARED_LIBS)
    target_link_libraries(
      ${PROJECT_NAME}
      PUBLIC
      vart::runner
      PRIVATE
      vart::trace
      ${XRT_COREUTIL_LIBRARIES}
      )
  else(BUILD_SHARED_LIBS)
    target_link_libraries(
      ${PROJECT_NAME}
      PUBLIC
      vart::runner
      PRIVATE
      ${XRT_COREUTIL_LIBRARIES}
      )
  endif(BUILD_SHARED_LIBS)
endif(MSVC)

include(CMakePackageConfigHelpers)

file(RELATIVE_PATH REL_INCLUDE_DIR
  ${CMAKE_INSTALL_PREFIX}/share/cmake/${PROJECT_NAME}
  ${CMAKE_INSTALL_PREFIX}/include
)

set(CONF_INCLUDE_DIRS "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}")

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/config.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake
  @ONLY
  )

write_basic_package_version_file (
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
  )
if(CMAKE_SOURCE_DIR STREQUAL graph-engine_SOURCE_DIR)
install(
  TARGETS
  ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  INCLUDES DESTINATION include
  RUNTIME  DESTINATION bin
  ARCHIVE  DESTINATION lib
  LIBRARY  DESTINATION lib
  )

install(
  FILES
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/graph-engine.hpp
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/create_graph_runner.hpp
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/create_dpu_runner.hpp
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/buffers.hpp
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/utility.hpp
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/qos.hpp
  COMPONENT runtime
  DESTINATION include/${PROJECT_NAME}
  )

install(
  FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  COMPONENT runtime
  DESTINATION share/cmake/${PROJECT_NAME}
  )

install(
  EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT runtime
  DESTINATION share/cmake/${PROJECT_NAME}
  )

install(
  FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  COMPONENT runtime
  DESTINATION share/cmake/${PROJECT_NAME}
  )
endif()
