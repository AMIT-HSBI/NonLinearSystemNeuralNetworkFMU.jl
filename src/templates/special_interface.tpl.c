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

fmi2Boolean isCategoryLogged(ModelInstance *comp, int categoryIndex);

static fmi2String logCategoriesNames[] = {"logEvents", "logSingularLinearSystems", "logNonlinearSystems", "logDynamicStateSelection", "logStatusWarning", "logStatusDiscard", "logStatusError", "logStatusFatal", "logStatusPending", "logAll", "logFmi2Call"};

#ifndef FILTERED_LOG
  #define FILTERED_LOG(instance, status, categoryIndex, message, ...) if (isCategoryLogged(instance, categoryIndex)) { \
      instance->functions->logger(instance->functions->componentEnvironment, instance->instanceName, status, \
          logCategoriesNames[categoryIndex], message, ##__VA_ARGS__); }
#endif

/* Forwarded equations */
<<FORWARD_EQUATION_BLOCK>>

fmi2Status myfmi2evaluateEq(fmi2Component c, const size_t eqNumber)
{
  ModelInstance *comp = (ModelInstance *)c;
  DATA* data = comp->fmuData;
  threadData_t *threadData = comp->threadData;

  useStream[LOG_NLS] = 0 /* false */;
  useStream[LOG_NLS_V] = 0 /* false */;
  FILTERED_LOG(comp, fmi2OK, LOG_FMI2_CALL, "myfmi2evaluateEq: Evaluating equation %u", eqNumber)

  switch (eqNumber)
  {
<<EQUATION_CASES>>
  default:
    return fmi2Error;
  }
  comp->_need_update = 0;

  return fmi2OK;
}
