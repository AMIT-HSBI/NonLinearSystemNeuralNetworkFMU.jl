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

#include "fmi2Functions.h"
#include "fmi2FunctionTypes.h"
#include "fmi2TypesPlatform.h"
#include "../simulation_data.h"

#ifdef __cplusplus
extern "C" {
#endif

FMI2_Export fmi2Status myfmi2EvaluateEq(fmi2Component c, const size_t eqNumber);
fmi2Status myfmi2EvaluateRes(fmi2Component c, const size_t eqNumber, double* x, double* res);
fmi2Status myfmi2EvaluateJacobian(fmi2Component c, const size_t eqNumber, double* x, double* res);
double* getJac(DATA* data, const size_t sysNumber);
int scaleResidual(double* jac, double* res, size_t n);

#ifdef __cplusplus
}  /* end of extern "C" { */
#endif
