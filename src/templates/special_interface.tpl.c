//
// Copyright (c) 1998-2022, Open Source Modelica Consortium (OSMC)
// Copyright (c) 2022 Andreas Heuermann
//
// This file is provided under the terms of GPL Version 3.
// GNU version 3 is obtained from: http://www.gnu.org/copyleft/gpl.html.
//

#include "special_interface.h"
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

static inline void resetThreadData(ModelInstance* comp)
{
  if (comp->threadDataParent) {
    pthread_setspecific(mmc_thread_data_key, comp->threadDataParent);
  }
  /* Clear the extra memory pools */
  omc_alloc_interface.collect_a_little();
}

static inline void setThreadData(ModelInstance* comp)
{
  if (comp->threadDataParent) {
    pthread_setspecific(mmc_thread_data_key, comp->threadData);
  }
}

/* Forwarded equations */
<<FORWARD_EQUATION_BLOCK>>

fmi2Status myfmi2evaluateEq(fmi2Component c, const size_t eqNumber)
{
  ModelInstance *comp = (ModelInstance *)c;
  DATA* data = comp->fmuData;
  threadData_t *threadData = comp->threadData;
  int success = 0;

  useStream[LOG_NLS] = 0 /* false */;
  useStream[LOG_NLS_V] = 0 /* false */;
  FILTERED_LOG(comp, fmi2OK, LOG_FMI2_CALL, "myfmi2evaluateEq: Evaluating equation %u", eqNumber)

  setThreadData(comp);
  /* try */
  MMC_TRY_INTERNAL(simulationJumpBuffer)

  switch (eqNumber)
  {
<<EQUATION_CASES>>
  default:
    return fmi2Error;
  }
  comp->_need_update = 0;

  success=1;

  /* catch */
  MMC_CATCH_INTERNAL(simulationJumpBuffer)
  resetThreadData(comp);

  if(!success) {
    FILTERED_LOG(comp, fmi2Error, LOG_FMI2_CALL, "myfmi2evaluateEq: Caught an error.")
    return fmi2Error;
  }

  return fmi2OK;
}
