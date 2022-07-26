cmake_minimum_required(VERSION 3.16)

project(onnxWrapper)

if(NOT DEFINED ORT_DIR)
  message(FATAL_ERROR "ORT_DIR not set. Provide a path to ONNX Runtime with\n\t-DORT_DIR=\"/path/to/ort\"")
else()
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
