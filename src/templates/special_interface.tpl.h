//
// Copyright (c) 2022 Andreas Heuermann
// Licensed under the MIT license. See LICENSE.md file in the
// NonLinearSystemNeuralNetworkFMU.jl root for details.
//

#include "../simulation_data.h"
#include "../simulation/solver/solver_main.h"
#include "../<<MODELNAME>>_model.h"
#include "fmu2_model_interface.h"
#include "fmu_read_flags.h"
#include "fmi2Functions.h"

#ifdef __cplusplus
extern "C" {
#endif

FMI2_Export fmi2Status myfmi2evaluateEq(fmi2Component c, const size_t eqNumber);

#ifdef __cplusplus
}  /* end of extern "C" { */
#endif
