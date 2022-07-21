#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

"""
fmiEvaluateEq(fmu, eqNumber)

Call equation function
fmi2Status myfmi2evaluateEq(fmi2Component c, const size_t eqNumber)
for given equation number.

# Arguments
  - `fmu::FMICore.FMU2`: FMU object containing C void pointer to FMU component.
  - `eqNumber::Int`: Equation index specifying equation to evaluate.

# Returns
  - Returns status of Libdl.ccall for `:myfmi2evaluateEq`.
"""
function fmiEvaluateEq(fmu::FMIImport.FMU2, eqNumber::Integer)
  return fmiEvaluateEq(fmu.components[1], eqNumber)
end

function fmiEvaluateEq(comp::FMICore.FMU2Component, eq::Integer)

  @assert eq>=0 "Equation index has to be non-negative!"

  fmiEvaluateEq = Libdl.dlsym(comp.fmu.libHandle, :myfmi2evaluateEq)

  eqCtype = Csize_t(eq)

  status = ccall(fmiEvaluateEq,
                 Cuint,
                 (Ptr{Nothing}, Csize_t),
                 comp.compAddr, eqCtype)

  return status
end
