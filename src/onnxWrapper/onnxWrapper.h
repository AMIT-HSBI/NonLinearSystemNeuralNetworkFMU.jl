//
// Copyright (c) 2022 Andreas Heuermann
// Licensed under the MIT license. See LICENSE.md file in the
// NonLinearSystemNeuralNetworkFMU.jl root for details.
//

#include "onnxruntime_c_api.h"

struct OrtWrapperData {
  const OrtApi* g_ort;
  OrtEnv* env;
  OrtSessionOptions* session_options;
  OrtSession* session;
  float* model_input;
  const char** input_names;
  float* model_output;
  const char** output_names;
  OrtMemoryInfo* memory_info;
  OrtValue* input_tensor;
  OrtValue* output_tensor;
};

struct OrtWrapperData* initOrtData(const char* pathToONNX, const char* modelName, unsigned int nInputs, unsigned int nOutputs, const char* input_name, const char* output_name);
void deinitOrtData(struct OrtWrapperData* ortData);
void evalModel(struct OrtWrapperData* ortData);
