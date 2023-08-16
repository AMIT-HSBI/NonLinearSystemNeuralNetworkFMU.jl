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

"""
    main(modelName, moFiles; options=OMOptions(workingDir=joinpath(pwd(), modelName)), dataGenOptions=DataGenOptions(method = RandomMethod(), n=1000, nBatches=Threads.nthreads()), reuseArtifacts=false)

Main routine to generate training data from Modelica file(s).
Generate BSON artifacts and FMUs for each step. Artifacts can be re-used when restarting
main routine to skip already performed stepps.

Will perform profiling, min-max value compilation, FMU generation and data
generation for all non-linear equation systems of `modelName`.

  # Arguments
  - `modelName::String`:      Name of Modelica model to simulate.
  - `moFiles::Array{String}`: Path to .mo file(s).

# Keywords
  - `omOptions::OMOptions`:           Settings for OpenModelcia compiler.
  - `dataGenOptions::DataGenOptions`  Settings for data generation.
  - `reuseArtifacts=false`:           Use artifacts to skip already performed steps if true.

# Returns
  - `csvFiles::Array{String}`:              Array of generate CSV files with training data.
  - `fmu::String`:                          Path to unmodified 2.0 ME FMU.
  - `profilingInfo::Array{ProfilingInfo}`:  Array of profiling information for each non-linear equation system.

See also [`profiling`](@ref), [`minMaxValuesReSim`](@ref), [`generateFMU`](@ref),
[`addEqInterface2FMU`](@ref), [`generateTrainingData`](@ref).
"""
function main(modelName::String,
              moFiles::Array{String};
              omOptions::OMOptions = OMOptions(workingDir=joinpath(pwd(), modelName)),
              dataGenOptions::DataGenOptions = DataGenOptions(method=RandomMethod(), n=1000, nBatches=Threads.nthreads()),
              reuseArtifacts::Bool = false)

  mkpath(omOptions.workingDir)

  # Profiling and min-max values
  profilingInfoFile = joinpath(omOptions.workingDir, "profilingInfo.bson")
  profOptions = OMOptions(pathToOmc = omOptions.pathToOmc,
                          workingDir = joinpath(omOptions.workingDir, "temp-profiling"),
                          outputFormat = "csv",
                          clean = omOptions.clean,
                          commandLineOptions = omOptions.commandLineOptions)
  local profilingInfo
  if(reuseArtifacts && isfile(profilingInfoFile))
    @info "Reusing $profilingInfoFile"
    BSON.@load profilingInfoFile profilingInfo
  else
    @info "Profile $modelName"
    profilingInfo = profiling(modelName, moFiles; options=profOptions, threshold=0)
    if length(profilingInfo) == 0
      @warn "No equation slower than given threshold. Nothing to do."
      return
    end

    BSON.@save profilingInfoFile profilingInfo
    if omOptions.clean
      rm(profOptions.workingDir, force=true, recursive=true)
    end
  end

  # FMU
  genFmuOptions = OMOptions(pathToOmc = omOptions.pathToOmc,
                            workingDir = joinpath(omOptions.workingDir, "temp-fmu"),
                            outputFormat = nothing,
                            clean = omOptions.clean,
                            commandLineOptions = omOptions.commandLineOptions)
  fmuFile = joinpath(omOptions.workingDir, modelName*".fmu")
  local fmu
  if(reuseArtifacts && isfile(fmuFile))
    @info "Reusing $fmuFile"
    fmu = fmuFile
  else
    @info "Generate default FMU"
    fmu = generateFMU(modelName, moFiles, options=genFmuOptions)
    mv(fmu, joinpath(omOptions.workingDir, basename(fmu)), force=true)
    fmu = joinpath(omOptions.workingDir, basename(fmu))
    if omOptions.clean
      rm(genFmuOptions.workingDir, force=true, recursive=true)
    end
  end

  # Extended FMU
  tempDir = joinpath(omOptions.workingDir, "temp-extendfmu")
  fmuFile = joinpath(omOptions.workingDir, modelName*".interface.fmu")
  if(reuseArtifacts && isfile(fmuFile))
    @info "Reusing $fmuFile"
    fmu_interface = fmuFile
  else
    @info "Generate extended FMU"
    allEqs = [prof.eqInfo.id for prof in profilingInfo]
    fmu_interface = addEqInterface2FMU(modelName, fmu, allEqs, workingDir=tempDir)
    mv(fmu_interface, joinpath(omOptions.workingDir, basename(fmu_interface)), force=true)
    fmu_interface = joinpath(omOptions.workingDir, basename(fmu_interface))
    if omOptions.clean
      rm(tempDir, force=true, recursive=true)
    end
  end

  # Data
  @info "Generate training data"
  tempDir = joinpath(omOptions.workingDir, "temp-data")
  csvFiles = String[]
  for prof in profilingInfo
    eqIndex = prof.eqInfo.id
    inputVars = prof.usingVars
    outputVars = prof.iterationVariables
    minBoundary = prof.boundary.min
    maxBoundary = prof.boundary.max

    fileName = abspath(joinpath(omOptions.workingDir, "data", "eq_$(prof.eqInfo.id).csv"))
    csvFile = generateTrainingData(fmu_interface, tempDir, fileName, eqIndex, inputVars, minBoundary, maxBoundary, outputVars; options = dataGenOptions)
    push!(csvFiles, csvFile)
  end
  if omOptions.clean
    rm(tempDir, force=true, recursive=true)
  end

  return csvFiles, fmu, profilingInfo
end
