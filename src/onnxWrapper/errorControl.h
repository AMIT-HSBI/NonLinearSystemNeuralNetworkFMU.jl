//
// Copyright (c) 2022-2023 Andreas Heuermann
//
// This file is part of NonLinearSystemNeuralNetworkFMU.jl.
//
// NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ERROR_CONTROL_H
#define ERROR_CONTROL_H

#include <stdio.h>

/* forward types */
struct OrtWrapperData;

/* Residual function prototype */
typedef void (*resFunction)(void*, const double*, double*, const int*);

/* Function prototypes */
void evalResidual(resFunction f, void* userData, struct OrtWrapperData* ortData);
void printResiduum(unsigned int id, double time, struct OrtWrapperData* ortData);
double residualNorm(double time, struct OrtWrapperData* ortData);

#endif  // ERROR_CONTROL_H
