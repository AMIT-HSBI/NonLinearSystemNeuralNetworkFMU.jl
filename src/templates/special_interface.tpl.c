//
// Copyright (c) 1998-2022, Open Source Modelica Consortium (OSMC)
// Copyright (c) 2022-2023 Andreas Heuermann
//
// This file is provided under the terms of GPL Version 3.
// GNU version 3 is obtained from: http://www.gnu.org/copyleft/gpl.html.
//

#include "special_interface.h"
#include "../simulation/solver/solver_main.h"
#include "../simulation/solver/nonlinearSolverHybrd.h"
#include "../simulation/solver/nonlinearSolverHomotopy.h"
#include "../simulation/solver/omc_math.h"
#include "../util/simulation_options.h"
#include "../<<MODELNAME>>_model.h"
#include "../<<MODELNAME>>_12jac.h"
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

/**
 * @brief Evaluate equation.
 *
 * Result variables of equation have to be retrieved with fmi2GetXXX functions.
 *
 * @param c             Pointer to FMU component.
 * @param eqNumber      Equation to evaluate.
 * @return fmi2Status   Return fmi2OK on success, fmi2Error otherwise.
 */
fmi2Status myfmi2EvaluateEq(fmi2Component c, const size_t eqNumber)
{
  ModelInstance *comp = (ModelInstance *)c;
  DATA* data = comp->fmuData;
  threadData_t *threadData = comp->threadData;
  int success = 0;

  useStream[LOG_NLS] = 0 /* false */;
  useStream[LOG_NLS_V] = 0 /* false */;
  useStream[LOG_ASSERT] = 0 /* false */;
  FILTERED_LOG(comp, fmi2OK, LOG_FMI2_CALL, "myfmi2EvaluateEq: Evaluating equation %u", eqNumber)

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
    FILTERED_LOG(comp, fmi2Error, LOG_FMI2_CALL, "myfmi2EvaluateEq: Caught an error.")
    return fmi2Error;
  }

  return fmi2OK;
}

/**
 * @brief Evaluate residual equation.
 *
 * f(x) = res
 *
 * @param c             Pointer to FMU component.
 * @param eqNumber      Residual equation to evaluate.
 * @param x             X vector.
 * @param res           Residual vector on return.
 * @return fmi2Status   Return fmi2OK on success, fmi2Error otherwise.
 */
fmi2Status myfmi2EvaluateRes(fmi2Component c, const size_t eqNumber, double* x, double* res)
{
  ModelInstance *comp = (ModelInstance *)c;
  DATA* data = comp->fmuData;
  threadData_t *threadData = comp->threadData;
  int success = 0;
  int iflag = 0;

  RESIDUAL_USERDATA resUserData = {
    .data       = data,
    .threadData = threadData,
    .solverData = NULL
  };

  useStream[LOG_NLS] = 0 /* false */;
  useStream[LOG_NLS_V] = 0 /* false */;
  useStream[LOG_ASSERT] = 0 /* false */;
  FILTERED_LOG(comp, fmi2OK, LOG_FMI2_CALL, "myfmi2EvaluateRes: Evaluating residual %u", eqNumber)

  setThreadData(comp);
  /* try */
  MMC_TRY_INTERNAL(simulationJumpBuffer)

  switch (eqNumber)
  {
<<RESIDUAL_CASES>>
  default:
    return fmi2Error;
  }

  success=1;

  /* catch */
  MMC_CATCH_INTERNAL(simulationJumpBuffer)
  resetThreadData(comp);

  if(!success) {
    FILTERED_LOG(comp, fmi2Error, LOG_FMI2_CALL, "myfmi2EvaluateRes: Caught an error.")
    return fmi2Error;
  }

  return fmi2OK;
}

/**
 * @brief Get Jacobian of non-linear equation system.
 *
 * @param c             Pointer to FMU component.
 * @param sysNumber     Number of non-linear system.
 * @return jac          Return pointer to Jacobian matrix in row-major-format or NULL in error case.
 */
double* getJac(DATA* data, const size_t sysNumber) {
  // maybe pass double* x, double* res additionally
  double* jac;
  NONLINEAR_SYSTEM_DATA* nlsSystem = &(data->simulationInfo->nonlinearSystemData[sysNumber]);

  switch (nlsSystem->nlsMethod)
  {
  //case NLS_HYBRID:
  //  DATA_HYBRD* solverData = (DATA_HYBRD*) nlsSystem->solverData;
  //  jac = solverData->fjac;
  //  break;
  //case NLS_NEWTON:
  //  DATA_NEWTON* solverData = (DATA_NEWTON*) nlsSystem->solverData;
  //  jac = solverData->fjac;
  //  break;
  case NLS_HOMOTOPY:
    // here evaluate jac at x and set it equal to res and return res
    return getHomotopyJacobian(nlsSystem);
  default:
    printf("Unknown NLS method  %d in myfmi2GetJac\n", (int)nlsSystem->nlsMethod);
    return NULL;
  }
}



/**
 * @brief Scale residual vector.
 *
 * @param jac             Pointer to n times n Jacobian in row-major format.
 * @param res             Pointer to residual vector to scale.
 * @param n               Size of jacobian and residual vector.
 * @return isRegular      Return 1 (true) if matrix is regular and 0 (false) if it's singular.
 */
int scaleResidual(double* jac, double* res, size_t n) {
  int jac_row_start;
  int isRegular = 1;
  double scaling;

  for(int i=0; i<n; i++)
  {
    jac_row_start = i*n;
    scaling = _omc_gen_maximumVectorNorm(&(jac[jac_row_start]), n);
    if(scaling <= 0.0) {
      //printf("Jacobian matrix is singular!\n");
      scaling = 1e-16;
      isRegular = 0;
    }
    res[i] = res[i] / scaling;
  }

  return isRegular;
}


fmi2Status myfmi2EvaluateJacobian(fmi2Component c, const size_t sysNumber, double* x, double* jac)
{
  ModelInstance *comp = (ModelInstance *)c;
  DATA* data = comp->fmuData;
  threadData_t *threadData = comp->threadData;
  NONLINEAR_SYSTEM_DATA* nlsSystem = &(data->simulationInfo->nonlinearSystemData[sysNumber]);

  switch(nlsSystem->nlsMethod)
  {
    case NLS_HOMOTOPY:
      DATA_HOMOTOPY* solverData = (DATA_HOMOTOPY*) nlsSystem->solverData;
      int status = getAnalyticalJacobianHomotopy(solverData, jac);
      break;
    default:
      printf("fehler");
      abort();
      break;
  }
  return fmi2OK;
}