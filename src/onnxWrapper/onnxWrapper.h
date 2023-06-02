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

#ifndef ONNX_WWRAPPER_H
#define ONNX_WWRAPPER_H

#include "onnxruntime_c_api.h"
#include "errorControl.h"

struct OrtWrapperData {
  const OrtApi* g_ort;
  OrtEnv* env;
  OrtSessionOptions* session_options;
  OrtSession* session;
  size_t nInputs;                     /* Number of inputs */
  float* model_input;                 /* Input variables (used variables) */
  const char** input_names;           /* Names of input variables */
  float* model_output;                /* Output variables (iteration variables x) */
  const char** output_names;          /* Names of output variables */
  OrtMemoryInfo* memory_info;
  OrtValue* input_tensor;
  OrtValue* output_tensor;

  /* Residuum */
  double* x;                          /* Double version of model_output */
  double* res;                        /* Residuum f(x), x is model_output */
  size_t nRes;                        /* Length of arrays x and res */
  FILE * csvFile;                     /* Log file for residuum values */

  /* Training area */
  double* min;                        /* Minimum allowed values for model_input, size nInputs */
  double* max;                        /* Maximum allowed values for model_input, size nInputs */
};

struct OrtWrapperData* initOrtData(const char* equationName, const char* pathToONNX, const char* modelName, unsigned int nInputs, unsigned int nOutputs, int logResiduum, int numThreads);
void deinitOrtData(struct OrtWrapperData* ortData);
void evalModel(struct OrtWrapperData* ortData);

#endif // ONNX_WWRAPPER_H
