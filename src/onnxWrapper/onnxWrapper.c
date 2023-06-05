//
//
// Copyright (c) 2022 Andreas Heuermann
//
// This file is part of NonLinearSystemNeuralNetworkFMU.jl.
//
// NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
//
//
// ORT_ABORT_ON_ERROR taken from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "onnxWrapper.h"

#ifdef _WIN32
#include <windows.h>

wchar_t* wideCharCopy(const char* str) {
  size_t length = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
  wchar_t* str_wide = (wchar_t*) malloc(length * sizeof(str_wide));
  MultiByteToWideChar(CP_UTF8, 0, str, -1, str_wide, length);
  return str_wide;
}
#endif

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

/**
 * @brief Verify that the ONNX model has one input and one output.
 *
 * @param g_ort       ONNX runtime API
 * @param session     ONNX session
 */
void verify_input_output_count(const OrtApi* g_ort, OrtSession* session) {
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

/**
 * @brief Initialize ORT data for ONNX model.
 *
 * @param equationName              Name of equation.
 * @param pathToONNX                Path to ONNX model.
 * @param modelName                 Name of ONNX model.
 * @param nInputs                   Number of inputs to ONNX model.
 * @param nOutputs                  Number of outputs of ONNX model.
 * @return struct OrtWrapperData*   Pointer to ORT wrapper data.
 */
struct OrtWrapperData* initOrtData(const char* equationName, const char* pathToONNX, const char* modelName, unsigned int nInputs, unsigned int nOutputs) {
  struct OrtWrapperData* ortData = calloc(1, sizeof (struct OrtWrapperData));

  /* Initialize ORT */
  const OrtApi* g_ort;
  OrtEnv* env;
  OrtSessionOptions* session_options;
  OrtSession* session;

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
    return NULL;
  }
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, modelName, &env));
  assert(env != NULL);
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
  /* Use only 1 additional thread */
  ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(session_options, 1));
  ORT_ABORT_ON_ERROR(g_ort->SetInterOpNumThreads(session_options, 1));

#ifdef _WIN32
  wchar_t* pathToONNX_utf = wideCharCopy(pathToONNX);
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, pathToONNX_utf, session_options, &session));
  free(pathToONNX_utf);
#else
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, pathToONNX, session_options, &session));
#endif
  verify_input_output_count(g_ort, session);

  ortData->g_ort = g_ort;
  ortData->env = env;
  ortData->session_options = session_options;
  ortData->session = session;

  OrtAllocator* allocator;
  ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  /* Initialize model_input and model_output */
  ortData->nInputs = nInputs;
  ortData->model_input = calloc(nInputs, sizeof ortData->model_input[0]);
  ortData->model_output = calloc(nOutputs, sizeof ortData->model_output[0]);

  /* Initialize input and output tensors */
  unsigned int model_input_ele_count = nInputs;
  unsigned int model_output_ele_count = nOutputs;

  OrtMemoryInfo* memory_info = ortData->memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, model_input_ele_count};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);
  ortData->input_names = calloc(1, sizeof ortData->input_names[0]);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, 0, allocator, (char**) ortData->input_names));

  const int64_t output_shape[] = {1, model_output_ele_count};
  const size_t output_shape_len = sizeof(output_shape) / sizeof(output_shape[0]);
  const size_t model_output_len = model_output_ele_count * sizeof(float);
  ortData->output_names = calloc(1, sizeof ortData->output_names[0]);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, 0, allocator, (char**) ortData->output_names));

  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, ortData->model_input, model_input_len, input_shape,
                                                           input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &ortData->input_tensor));

  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, ortData->model_output, model_output_len, output_shape,
                                                           output_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &ortData->output_tensor));

  /* Initialize residuum arrays */
  ortData->nRes = (size_t) nOutputs;
  ortData->x = calloc(ortData->nRes, sizeof ortData->x[0]);
  ortData->res = calloc(ortData->nRes, sizeof ortData->res[0]);
  char csvFilePath[2048];
  snprintf(csvFilePath, 2048, "%s_residuum.csv", equationName);
  ortData->csvFile = fopen(csvFilePath, "w");
  fprintf(ortData->csvFile, "time,");
  fprintf(ortData->csvFile, "inBounds,");
  fprintf(ortData->csvFile, "scaled_res_norm,");
  // TODO: Log scaled residual vector (norm)
  for(int i=0; i<ortData->nRes-1; i++) {
    fprintf(ortData->csvFile, "res[%i],", i);
  }
  fprintf(ortData->csvFile, "res[%li]\n", ortData->nRes-1);

  /* Initialize training area boundaries */
  ortData->min = calloc(nInputs, sizeof ortData->min[0]);
  ortData->max = calloc(nInputs, sizeof ortData->max[0]);

  return ortData;
}

/**
 * @brief Deinitialize ORT data.
 *
 * @param ortData   Pointer to ORT data to free.
 */
void deinitOrtData(struct OrtWrapperData* ortData) {
  /* Free memory */
  ortData->g_ort->ReleaseMemoryInfo(ortData->memory_info);
  ortData->g_ort->ReleaseValue(ortData->output_tensor);
  ortData->g_ort->ReleaseValue(ortData->input_tensor);

  free(ortData->model_input);
  free((char*)ortData->input_names[0]);
  free(ortData->input_names);
  free(ortData->model_output);
  free((char*)ortData->output_names[0]);
  free(ortData->output_names);

  /* Free ORT */
  ortData->g_ort->ReleaseSessionOptions(ortData->session_options);
  ortData->g_ort->ReleaseSession(ortData->session);
  ortData->g_ort->ReleaseEnv(ortData->env);

  /* Free residuum data */
  free(ortData->x);
  free(ortData->res);
  fclose(ortData->csvFile);

  /* Free training are boundaries */
  free(ortData->min);
  free(ortData->max);

  free(ortData);
}

/**
 * @brief Return pointer to input array of ONNX model.
 *
 * @param ortData   Pointer to ORT wrapper data.
 * @return float*   Pointer to input array.
 */
float* inputDataPtr(struct OrtWrapperData* ortData) {
  return ortData->model_input;
}

/**
 * @brief Return pointer to output array of ONNX model.
 *
 * @param ortData   Pointer to ORT wrapper data.
 * @return float*   Pointer to output array.
 */
float* outputDataPtr(struct OrtWrapperData* ortData) {
  return ortData->model_output;
}

/**
 * @brief Evaluate ONNX model.
 *
 * @param ortData 
 */
void evalModel(struct OrtWrapperData* ortData) {
  const OrtApi* g_ort = ortData->g_ort;
  ORT_ABORT_ON_ERROR(
    g_ort->Run(
      ortData->session,
      NULL,
      ortData->input_names,
      (const OrtValue* const*)&ortData->input_tensor,
      1,
      ortData->output_names,
      1,
      &ortData->output_tensor));
}
