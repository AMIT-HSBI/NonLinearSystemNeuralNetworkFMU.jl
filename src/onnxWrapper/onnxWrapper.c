//
// Copyright (c) 2022 Andreas Heuermann
// Licensed under the MIT license. See LICENSE.md file in the
// NonLinearSystemNeuralNetworkFMU.jl root for details.
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
 * @param pathToONNX                Path to ONNX model.
 * @param modelName                 Name of ONNX model.
 * @param nInputs                   Number of inputs to ONNX model.
 * @param nOutputs                  Number of outputs of ONNX model.
 * @return struct OrtWrapperData*   Pointer to ORT wrapper data.
 */
struct OrtWrapperData* initOrtData(const char* pathToONNX, const char* modelName, unsigned int nInputs, unsigned int nOutputs) {
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

  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, pathToONNX, session_options, &session));
  verify_input_output_count(g_ort, session);

  ortData->g_ort = g_ort;
  ortData->env = env;
  ortData->session_options = session_options;
  ortData->session = session;

  /* Initialize model_input and model_output */
  ortData->model_input = calloc(nInputs, sizeof ortData->model_input);
  ortData->model_output = calloc(nInputs, sizeof ortData->model_output);

  /* Initialize input and output tensors */
  unsigned int model_input_ele_count = nInputs;
  unsigned int model_output_ele_count = nOutputs;

  OrtMemoryInfo* memory_info = ortData->memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, model_input_ele_count};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);
  // TODO: Get input names automatically
  const char* input_name = "onnx::Flatten_0";
  ortData->input_names = calloc(1, sizeof ortData->input_names);
  ortData->input_names[0] = strdup(input_name);
  const int64_t output_shape[] = {1, model_output_ele_count};
  const size_t output_shape_len = sizeof(output_shape) / sizeof(output_shape[0]);
  const size_t model_output_len = model_output_ele_count * sizeof(float);
  // TODO: Get output names automatically
  const char* output_name = "13";
  ortData->output_names = calloc(1, sizeof ortData->output_names);
  ortData->output_names[0] = strdup(output_name);

  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, ortData->model_input, model_input_len, input_shape,
                                                           input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &ortData->input_tensor));

  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, ortData->model_output, model_output_len, output_shape,
                                                           output_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &ortData->output_tensor));

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
  free((char**)ortData->input_names[0]);
  free(ortData->input_names);
  free(ortData->model_output);
  free((char**)ortData->output_names[0]);
  free(ortData->output_names);

  /* Free ORT */
  ortData->g_ort->ReleaseSessionOptions(ortData->session_options);
  ortData->g_ort->ReleaseSession(ortData->session);
  ortData->g_ort->ReleaseEnv(ortData->env);

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
