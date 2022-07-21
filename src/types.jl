#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

struct EqInfo
  id::Int64
  ncall::Int64
  time::Float64
  maxTime::Float64
  fraction::Float64
end

function Base.show(io::IO, ::MIME"text/plain", eqInfo::EqInfo)
  Printf.@printf(io, "EqInfo(id: %s, ", eqInfo.id)
  Printf.@printf(io, "ncall: %s, ", eqInfo.ncall)
  Printf.@printf(io, "time: %s, ", eqInfo.time)
  Printf.@printf(io, "maxTime: %s, ", eqInfo.maxTime)
  Printf.@printf(io, "fraction: %s)", eqInfo.fraction)
end

struct ProfilingInfo
  eqInfo::EqInfo
  iterationVariables::Array{String}
  loopEquations::Array{Int64}
  usingVars::Array{String}
end

function Base.show(io::IO, ::MIME"text/plain", profilingInfo::ProfilingInfo)
  Printf.@printf(io, "ProfilingInfo(%s, ", profilingInfo.eqInfo)
  Printf.@printf(io, "iterationVariables: %s, ", profilingInfo.iterationVariables)
  Printf.@printf(io, "loopEquations: %s, ", profilingInfo.loopEquations)
  Printf.@printf(io, "usingVars: %s)", profilingInfo.usingVars)
end
