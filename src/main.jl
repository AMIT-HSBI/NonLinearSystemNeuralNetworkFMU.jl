#
# Copyright (c) 2022 Andreas Heuermann
#
# This file is part of NonLinearSystemNeuralNetworkFMU.jl.
#
# NonLinearSystemNeuralNetworkFMU.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NonLinearSystemNeuralNetworkFMU.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NonLinearSystemNeuralNetworkFMU.jl. If not, see <http://www.gnu.org/licenses/>.
#

"""
    main(modelName, moFiles;  workdir=joinpath(pwd(),modelName), reuseArtifacts=false, N=1000)

Main routine to generate training data from Modelica file(s).
Generate BSON artifacts and FMUs for each step. Artifacts can be re-used when restarting
main routine to skip already performed stepps.

Will perform profiling, min-max value compilation, FMU generation and data
generation for all non-linear equation systems of `modelName`.

  # Arguments
  - `modelName::String`:      Name of Modelica model to simulate.
  - `moFiles::Array{String}`: Path to .mo file(s).

# Keywords
  - `workingDir::String=pwd()`: Working directory for omc. Defaults to the current directory.
  - `reuseArtifacts=false`:     Use artifacts to skip already performed steps if true.
   -`N=1000::Integer`:          Number of data points fto genreate or each non-linear equation system.

# Returns
  - `csvFiles::Array{String}`:              Array of generate CSV files with training data.
  - `fmu::String`:                          Path to unmodified 2.0 ME FMU.
  - `profilingInfo::Array{ProfilingInfo}`:  Array of profiling information for each non-linear equation system.

See also [`profiling`](@ref), [`minMaxValuesReSim`](@ref), [`generateFMU`](@ref),
[`addEqInterface2FMU`](@ref), [`generateTrainingData`](@ref).
"""
function main(modelName::String,
              moFiles::Array{String};
              workdir=joinpath(pwd(),modelName)::String,
              reuseArtifacts = false,
              N=1000::Integer)
  mkpath(workdir)
  tempDir = joinpath(workdir, "temp")

  # Profiling
  profilingInfoFile = joinpath(workdir, "profilingInfo.bson")
  local profilingInfo
  if(reuseArtifacts && isfile(profilingInfoFile))
    @info "Reusing $profilingInfoFile"
    BSON.@load profilingInfoFile profilingInfo
  else
    @info "Profile $modelName"
    profilingInfo = profiling(modelName, moFiles; threshold=0, workingDir=tempDir)
    BSON.@save profilingInfoFile profilingInfo
    rm(tempDir, force=true, recursive=true)
  end

  # Min-max values
  minMaxFile = joinpath(workdir, "minMax.bson")
  local min
  local max
  allUsedvars = unique(vcat([prof.usingVars for prof in profilingInfo]...))
  if(reuseArtifacts && isfile(minMaxFile))
    @info "Reusing $minMaxFile"
    BSON.@load minMaxFile min max
  else
    @info "Find min-max values of used varaibles"
    (min, max) = minMaxValuesReSim(allUsedvars, modelName, moFiles, workingDir=tempDir)
    BSON.@save minMaxFile min max
    rm(tempDir, force=true, recursive=true)
  end

  # FMU
  fmuFile = joinpath(workdir, modelName*".fmu")
  local fmu
  if(reuseArtifacts && isfile(fmuFile))
    @info "Reusing $fmuFile"
    fmu = fmuFile
  else
    @info "Generate default FMU"
    fmu = generateFMU(modelName, moFiles, workingDir=tempDir)
    mv(fmu, joinpath(workdir, basename(fmu)), force=true)
    fmu = joinpath(workdir, basename(fmu))
    rm(tempDir, force=true, recursive=true)
  end

  # Extended FMU
  fmuFile = joinpath(workdir, modelName*".interface.fmu")
  local fmu
  if(reuseArtifacts && isfile(fmuFile))
    @info "Reusing $fmuFile"
    fmu_interface = fmuFile
  else
    @info "Generate extended FMU"
    allEqs = [prof.eqInfo.id for prof in profilingInfo]
    fmu_interface = addEqInterface2FMU(modelName, fmu, allEqs, workingDir=tempDir)
    mv(fmu_interface, joinpath(workdir, basename(fmu_interface)), force=true)
    fmu_interface = joinpath(workdir, basename(fmu_interface))
    rm(tempDir, force=true, recursive=true)
  end

  # Data
  @info "Generate training data"
  csvFiles = String[]
  for prof in profilingInfo
    eqIndex = prof.eqInfo.id
    inputVars = prof.usingVars
    outputVars = prof.iterationVariables

    mi = Array{Float64}(undef, length(inputVars))
    ma = Array{Float64}(undef, length(inputVars))
  
    for (i,var) in enumerate(inputVars)
      idx = findfirst(x->x==var, allUsedvars)
      if idx === nothing
        error("Variable " * var * "not found in all used vars array.")
      end
      mi[i] = min[idx]
      ma[i] = max[idx]
    end

    fileName = abspath(joinpath(workdir, "data", "eq_$(prof.eqInfo.id).csv"))
    csvFile = generateTrainingData(fmu_interface, fileName, eqIndex, inputVars, mi, ma, outputVars; N = N)
    push!(csvFiles, csvFile)
  end

  return csvFiles, fmu, profilingInfo
end
