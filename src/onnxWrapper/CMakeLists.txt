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

# Change default CMake install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install CACHE PATH "Default installation directory" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
endif()

if(NOT DEFINED ENV{ORT_DIR})
  message(FATAL_ERROR "Environment variable ORT_DIR not set.")
else()
  set(ORT_DIR $ENV{ORT_DIR})
  message(STATUS "Using ORT_DIR: ${ORT_DIR}")
endif()

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

include_directories(${ORT_INCLUDE})

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${ORT_DIR}/lib")

# Find ORT libraries
find_library(ORT_LIB
             NAMES onnxruntime libonnxruntime libonnxruntime.so
             PATHS ${ORT_DIR}/lib
             REQUIRED
             NO_DEFAULT_PATH)
message(STATUS "Using ORT_LIBR: ${ORT_LIB}")

add_library(onnxWrapper SHARED
            onnxWrapper.c)
set_target_properties(onnxWrapper PROPERTIES PUBLIC_HEADER "onnxWrapper.h")

target_link_libraries(onnxWrapper ${ORT_LIB})

install(TARGETS onnxWrapper)
