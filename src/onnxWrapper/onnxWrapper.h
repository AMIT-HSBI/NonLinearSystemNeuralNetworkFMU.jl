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
