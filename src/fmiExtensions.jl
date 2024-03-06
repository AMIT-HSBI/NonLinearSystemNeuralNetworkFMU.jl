#
# Copyright (c) 2022-2023 Andreas Heuermann
#
# This file is part of NonLinearSystemNeuralNetworkFMU.jl.
#
# NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
#

fmi2Status = UInt32
fmi2OK = Cuint(0)
fmi2Warning = Cuint(1)
fmi2Discard = Cuint(2)
fmi2Error = Cuint(3)
fmi2Fatal = Cuint(4)
fmi2Pending = Cuint(5)

"""
    fmiEvaluateEq(fmu, eqNumber)

Call equation function
fmi2Status myfmi2EvaluateEq(fmi2Component c, const size_t eqNumber)
for given equation number.

# Arguments
  - `fmu::FMICore.FMU2`: FMU object containing C void pointer to FMU component.
  - `eqNumber::Int`: Equation index specifying equation to evaluate.

# Returns
  - Returns status of Libdl.ccall for `:myfmi2EvaluateEq`.
"""
function fmiEvaluateEq(fmu::FMIImport.FMU2, eqNumber::Integer)::fmi2Status
  return fmiEvaluateEq(fmu.components[1], eqNumber)
end

function fmiEvaluateEq(comp::FMICore.FMU2Component, eq::Integer)::fmi2Status

  @assert eq>=0 "Equation index has to be non-negative!"

  fmiEvaluateEq = Libdl.dlsym(comp.fmu.libHandle, :myfmi2EvaluateEq)

  eqCtype = Csize_t(eq)

  status = ccall(fmiEvaluateEq,
                 Cuint,
                 (Ptr{Nothing}, Csize_t),
                 comp.compAddr, eqCtype)

  return status
end

"""
    fmiEvaluateRes(fmu, eqNumber, x)

Call residual function
`fmi2Status myfmi2EvaluateRes(fmi2Component c, const size_t eqNumber, double* x, double* res)`
for given equation number.

# Arguments
  - `fmu::FMICore.FMU2`: FMU object containing C void pointer to FMU component.
  - `eqNumber::Int`: Equation index specifying equation to evaluate.
  - `x::Array{Float64}` Values of iteration variables at which to evaluate the residuals.

# Returns
  - Status of Libdl.ccall for `:myfmi2EvaluateRes`.
  - Array of residual values.
"""
function fmiEvaluateRes(fmu::FMIImport.FMU2, eqNumber::Integer, x::Array{Float64})::Tuple{fmi2Status, Array{Float64}}
  return fmiEvaluateRes(fmu.components[1], eqNumber, x)
end

function fmiEvaluateRes(comp::FMICore.FMU2Component, eq::Integer, x::Array{Float64})::Tuple{fmi2Status, Array{Float64}}

  @assert eq>=0 "Residual index has to be non-negative!"

  fmiEvaluateRes = Libdl.dlsym(comp.fmu.libHandle, :myfmi2EvaluateRes)

  res = Array{Float64}(undef, length(x))

  eqCtype = Csize_t(eq)

  status = ccall(fmiEvaluateRes,
                 Cuint,
                 (Ptr{Nothing}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
                 comp.compAddr, eqCtype, x, res)

  return status, res
end


"""
    fmiEvaluateJacobian(fmu, eqNumber, x)

Evaluate Jacobian Matrix of given equation system 'eq' at a vector of iteration variables 'x'.

# Arguments
  - `fmu::FMIImport.FMU2`: FMU object containing C void pointer to FMU component.
  - `eq::Int`: Equation index specifying equation to evaluate jacobian of.
  - `x::Array{Float64}` Values of iteration variables at which to evaluate the jacobian.

# Returns
  - Status of Libdl.ccall for `:getJac`.
  - jacobian of eq at x.
"""
function fmiEvaluateJacobian(fmu::FMIImport.FMU2, eqNumber::Integer, x::Array{Float64})::Tuple{fmi2Status, Array{Float64}}
  return fmiEvaluateJacobian(fmu.components[1], eqNumber, x)
end

function fmiEvaluateJacobian(comp::FMICore.FMU2Component, eq::Integer, x::Array{Float64})::Tuple{fmi2Status, Array{Float64}}
  # wahrscheinlich braucht es eine c-Funktion, die nicht nur einen pointer auf die Jacobi-Matrix returned, sondern gleich die Auswertung an der Stelle x
  # diese Funktion muss auch ein Argument res nehmen welches dann die Evaluation enthält.?
  @assert eq>=0 "Residual index has to be non-negative!"

  # this is a pointer to Jacobian matrix in row-major-format or NULL in error case.
  fmiEvaluateJacobian = Libdl.dlsym(comp.fmu.libHandle, :myfmi2EvaluateJacobian)

  jac = Array{Float64}(undef, length(x)*length(x))

  eqCtype = Csize_t(eq)

  status = ccall(fmiEvaluateJacobian,
                 Cuint,
                 (Ptr{Nothing}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
                 comp.compAddr, eqCtype, x, jac)
  return status, jac
end
