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


# Get all project files
file(GLOB_RECURSE ALL_SOURCE_FILES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/graph-engine/*.cpp
    ${PROJECT_SOURCE_DIR}/src/graph-engine/*.hpp
    ${PROJECT_SOURCE_DIR}/include/graph-engine/*.hpp
)

foreach(SOURCE_FILE ${ALL_SOURCE_FILES})
  message(STATUS "Generating documentation for: ${SOURCE_FILE})")
endforeach()

# This will be the main output of our command
set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/xml/index.xml)
set(DOXYGEN_INPUT_DIRS ${PROJECT_SOURCE_DIR}/README.md ${PROJECT_SOURCE_DIR}/src/graph-engine ${PROJECT_SOURCE_DIR}/include/graph-engine)
set(DOXYGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/graph-engine)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

string (REPLACE ";" "\" \"" DOXYGEN_INPUT_DIR "${DOXYGEN_INPUT_DIRS}")

# Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# intermittent failures if we only create the parent directory
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html)
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}/xml)

message(STATUS "Using Doxygen Executable: ${DOXYGEN_EXECUTABLE}")

# Only regenerate Doxygen when the Doxyfile or files change
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${ALL_SOURCE_FILES}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
    COMMENT "Generating docs"
    VERBATIM
)

# Nice named target so we can run the job easily
add_custom_target(doxygen DEPENDS ${DOXYGEN_INDEX_FILE})
