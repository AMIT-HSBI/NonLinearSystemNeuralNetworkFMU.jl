#
# Copyright (c) 2022 Andreas Heuermann
#
# This file is part of NonLinearSystemNeuralNetworkFMU.jl.
#
# NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.16)

project(onnxWrapper)
set(CMAKE_BUILD_TYPE "Debug")

if(NOT DEFINED ENV{ORT_DIR} AND NOT DEFINED ORT_DIR)
  message(FATAL_ERROR "Environment variable ORT_DIR not set.")
elseif(DEFINED ENV{ORT_DIR})
  set(ORT_DIR $ENV{ORT_DIR})
endif()
message(STATUS "Using ORT_DIR: ${ORT_DIR}")

# Find ONNX Runtime (ORT)
find_path(ORT_INCLUDE
          "onnxruntime_c_api.h"
          PATHS ${ORT_DIR}/include
          REQUIRED)
if(ORT_INCLUDE)
  message(STATUS "ONNX Runtime include directory: ${ORT_INCLUDE}")
else()
  message(FATAL_ERROR "ONNX Runtime include directory not found.")
endif()

# Set RPATH
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Find ORT libraries
find_library(ORT_LIB
             NAMES onnxruntime libonnxruntime libonnxruntime.so
             PATHS ${ORT_DIR}/lib
             REQUIRED
             NO_DEFAULT_PATH)
message(STATUS "Using ORT_LIBR: ${ORT_LIB}")

add_library(onnxWrapper SHARED
            errorControl.c
            onnxWrapper.c
            measureTimes.c)

target_include_directories(onnxWrapper PUBLIC ${ORT_INCLUDE})
target_link_libraries(onnxWrapper PRIVATE ${ORT_LIB})

install(TARGETS onnxWrapper
        RUNTIME_DEPENDENCIES
          DIRECTORIES ${ORT_DIR}/lib
          PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
          POST_EXCLUDE_REGEXES "^\\/lib.*" "^\\/usr\\/lib.*" "^\\/usr\\/local\\/lib.*" ".*system32/.*\\.dll"
        ARCHIVE DESTINATION "."
        LIBRARY DESTINATION "."
        RUNTIME DESTINATION ".")
