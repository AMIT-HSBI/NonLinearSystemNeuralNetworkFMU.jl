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

#include "onnxWrapper.h"
#include "errorControl.h"

/* Private function prototypes */
void float2DoubleArray(const float* floatArray, double* doubleArray, const size_t len);

/**
 * @brief Evaluate residuum function.
 *
 * Converts model_output to double array and evaluate residuum function.
 *
 * @param f         Residuum function.
 * @param userData  User data provided by caller.
 * @param ortData   Pointer to ORT data.
 */
void evalResiduum(resFunction f, void* userData, struct OrtWrapperData* ortData) {
  const int iflag = 0; /* unused by resFunc */
  float2DoubleArray(ortData->model_output, ortData->x, ortData->nRes);
  f(userData, ortData->x, ortData->res, &iflag);
}

/**
 * @brief Print residuum values to stdout.
 *
 * @param id        Equation number of non-linear system.
 * @param time      Simulation time.
 * @param ortData   Pointer to ortData with residuum.
 */
void printResiduum(unsigned int id, double time, struct OrtWrapperData* ortData) {
  printf("Non-linear system %u residuum at time %f:\n", id, time);
  printf("res = [");
  for(int i = 0; i < ortData->nRes-1; i++) {
    printf("%e, ", ortData->res[i]);
  }
  printf("%e]\n", ortData->res[ortData->nRes-1]);
}

/**
 * @brief Save residuum values to CSV file.
 *
 * @param time      Simulation time.
 * @param ortData   Pointer to ortData with residuum.
 */
void writeResiduum(double time, struct OrtWrapperData* ortData) {
  fprintf(ortData->csvFile, "%f,", time);
  for(int i = 0; i < ortData->nRes-1; i++) {
    fprintf(ortData->csvFile, "%e,", ortData->res[i]);
  }
  fprintf(ortData->csvFile, "%e\n", ortData->res[ortData->nRes-1]);
}

/**
 * @brief Copy float array into double array.
 *
 * @param floatArray      Pointer to float array.
 * @param doubleArray     Pointer to double array.
 * @param len             Length of floatArray and doubleArray.
 */
void float2DoubleArray(const float* floatArray, double* doubleArray, const size_t len) {
  for(int i = 0; i < len; i++) {
    doubleArray[i] = floatArray[i];
  }
}
