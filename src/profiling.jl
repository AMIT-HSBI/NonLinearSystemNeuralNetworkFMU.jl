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
Simulate Modelica model with profiling enabled using given omc.
"""
function simulateWithProfiling(;modelName,
                                pathToMo,
                                pathToOmc,
                                tempDir,
                                outputFormat = "mat",
                                clean = false)

  if !isdir(tempDir)
    mkdir(tempDir)
  elseif clean
    rm(tempDir, force=true, recursive=true)
    mkdir(tempDir)
  end

  if Sys.iswindows()
    pathToMo = replace(pathToMo, "\\"=> "\\\\")
    tempDir = replace(tempDir, "\\"=> "\\\\")
  end

  logFilePath = joinpath(tempDir,"calls.log")
  logFile = open(logFilePath, "w")

  omc = OMJulia.OMCSession(pathToOmc)
  try
    msg = OMJulia.sendExpression(omc, "getVersion()")
    write(logFile, msg*"\n")
    OMJulia.sendExpression(omc, "loadFile(\"$(pathToMo)\")")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")
    OMJulia.sendExpression(omc, "cd(\"$(tempDir)\")")

    @info "setCommandLineOptions"
    msg = OMJulia.sendExpression(omc, "setCommandLineOptions(\"-d=newInst,infoXmlOperations,backenddaeinfo --profiling=all\")")
    write(logFile, string(msg)*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")

    @info "simulate"
    msg = OMJulia.sendExpression(omc, "simulate($(modelName), outputFormat=\"$(outputFormat)\", simflags=\"-lv=LOG_STATS -clock=RT -cpu -w\")")
    write(logFile, msg["messages"]*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")
  finally
    close(logFile)
    OMJulia.sendExpression(omc, "quit()", parsed=false)
  end

  profJsonFile = abspath(joinpath(tempDir, modelName*"_prof.json"))
  infoJsonFile = abspath(joinpath(tempDir, modelName*"_info.json"))
  resultFile = abspath(joinpath(tempDir, modelName*"_res."*outputFormat))
  return (profJsonFile, infoJsonFile, resultFile)
end

function isnonlinearequation(eq::Dict{String, Any}, id::Number)
  return eq["tag"] == "tornsystem" && eq["display"] == "non-linear"
end


"""
    findSlowEquations(profJsonFile, infoJsonFile; threshold)

Read JSON profiling file and find slowest non-linear loop equatiosn that need more then `threshold` of total simulation time.

# Arguments

  * `profJsonFile::String`: Path to profiling JSON file.
  * `infoJsonFile::String`: Path to info JSON file.

# Keywords
  * `threshold`: Lower bound on time consumption of equation.
                 0 <= threshold <= 1
"""
function findSlowEquations(profJsonFile::String, infoJsonFile::String; threshold = 0.03)
  infoFile = JSON.parsefile(infoJsonFile)
  equations = infoFile["equations"]

  profileFile = JSON.parsefile(profJsonFile)
  totalTime = profileFile["totalTime"]
  profileBlocks = profileFile["profileBlocks"]
  profileBlocks = sort(profileBlocks, by=x->x["time"], rev=true)

  block = profileBlocks[1]
  fraction = block["time"] / totalTime
  @info "Slowest eq $(block["id"]): ncall: $(block["ncall"]), time: $(block["time"]), maxTime: $(block["maxTime"]), fraction: $(fraction)"

  bigger = true
  i = 0
  slowesEq = EqInfo[]
  while(bigger)
    i += 1
    block = profileBlocks[i]
    fraction = block["time"] / totalTime
    bigger = fraction > threshold
    id = block["id"]
    if bigger && isnonlinearequation(equations[id+1], id)
      push!(slowesEq, EqInfo(block["id"], block["ncall"], block["time"], block["maxTime"], fraction))
    end
  end

  # Workaround for Windows until https://github.com/JuliaIO/JSON.jl/issues/347 is fixed.
  GC.gc()

  return slowesEq
end


"""
Return variables that are defined by equation with `eqIndex`.
"""
function findUsedVars(infoFile, eqIndex; filterParameters = true)
  equations = infoFile["equations"]
  eq = (equations[eqIndex+1])

  if eq["eqIndex"] != eqIndex
    error("Found wrong equation")
  end

  usingVars = nothing
  try
    usingVars = Array{String}(eq["uses"])
  catch
  end
  definingVars = []
  if haskey(eq, "defines")
    definingVars = Array{String}(eq["defines"])
  end

  # Check if used variable is parameter
  if filterParameters && usingVars !==nothing
    parameters = String[]
    variables = infoFile["variables"]
    for usedVar in usingVars
      try
        var = variables[usedVar]
        if var["kind"] == "parameter"
          push!(parameters, usedVar)
        end
      catch
      end
    end
    setdiff!(usingVars, parameters)
  end

  return (definingVars, usingVars)
end


"""
Read JSON info file and find all variables needed for equation with index `eqIndex`
as well as inner (torn) equations.
"""
function findDependentVars(jsonFile, eqIndex)
  infoFile = JSON.parsefile(jsonFile)

  equations = infoFile["equations"]
  eq = (equations[eqIndex+1])

  if eq["eqIndex"] != eqIndex
    error("Found wrong equation")
  end

  (definingVars, _) = findUsedVars(infoFile, eqIndex)

  iterationVariables = Array{String}(eq["defines"])
  loopEquations = collect(Iterators.flatten(eq["equation"]))
  innerEquations = Int64[]

  usingVars = String[]
  innerVars = String[]

  for loopeq in loopEquations
    if infoFile["equations"][loopeq+1]["tag"] =="jacobian"
      continue
    end
    (def, use) =findUsedVars(infoFile, loopeq)
    append!(usingVars, use)
    append!(innerVars, def)
    if !isempty(def)
      append!(innerEquations, loopeq)
    end
  end

  for v in vcat(innerVars, iterationVariables)
    deleteat!(usingVars, findall(x->x==v, usingVars))
  end

  # Workaround for Windows until https://github.com/JuliaIO/JSON.jl/issues/347 is fixed.
  GC.gc()

  return (unique(iterationVariables), innerEquations, unique(usingVars))
end


"""
Find smallest and biggest value for each variable using the CSV result file ± epsilon.
"""
function minMaxValues(variables::Array{String}; epsilon=0.05, resultFile)
  df = DataFrames.DataFrame(CSV.File(resultFile))

  min = Array{Float64}(undef, length(variables))
  max = Array{Float64}(undef, length(variables))
  for (i,var) in enumerate(variables)
    min[i] = minimum(df[!,var]) - abs(epsilon*minimum(df[!,var]))
    max[i] = maximum(df[!,var]) + abs(epsilon*maximum(df[!,var]))
  end

  return min, max
end


"""
    profiling(modelName, pathToMo, pathToOmc, workingDir; threshold = 0.03)

Find equations of Modelica model that are slower then threashold.

# Arguments
  - `modelName::String`:  Name of the Modelica model.
  - `pathToMo::String`:   Path to the *.mo file containing the model.
  - `pathToOm::Stringc`:  Path to omc used for simulating the model.

# Keywords
  - `workingDir::String = pwd()`: Working directory for omc. Defaults to the current directory.
  - `threshold = 0.01`: Slowest equations that need more then `threshold` of total simulation time.

# Returns
  - `profilingInfo::Vector{ProfilingInfo}`: Profiling information with non-linear equation systems slower than `threshold`.
"""
function profiling(modelName::String, pathToMo::String, pathToOmc::String; workingDir=pwd()::String, threshold = 0.01)::Vector{ProfilingInfo}

  omcWorkingDir = abspath(joinpath(workingDir, modelName))
  (profJsonFile, infoJsonFile, _) = simulateWithProfiling(modelName=modelName,
                                                          pathToMo=pathToMo,
                                                          pathToOmc=pathToOmc,
                                                          tempDir = omcWorkingDir,
                                                          outputFormat="mat")

  slowestEqs = findSlowEquations(profJsonFile, infoJsonFile; threshold=threshold)

  profilingInfo = Array{ProfilingInfo}(undef, length(slowestEqs))

  for (i,slowEq) in enumerate(slowestEqs)
    (iterationVariables, innerEquations, usingVars) = findDependentVars(infoJsonFile, slowestEqs[i].id)
    profilingInfo[i] = ProfilingInfo(slowEq, iterationVariables, innerEquations, usingVars)
  end

  return profilingInfo
end

"""
    minMaxValuesReSim(vars::Array{String}, modelName::String, pathToMo::String, pathToOmc::String; workingDir::String = pwd())

(Re-)simulate Modelica model and find miminum and maximum value each variable has during simulation.

# Arguments
  - `vars::Array{String}`:  Array of variables to get min-max values for.
  - `modelName::String`:    Name of Modelica model to simulate.
  - `pathToMo::String`:     Path to .mo file.
  - `pathToOm::Stringc`:    Path to OpenModelica Compiler omc.

# Keywords
  - `workingDir::String = pwd()`: Working directory for omc. Defaults to the current directory.

# Returns
  - `min::Array{Float64}`: Minimum values for each variable listed in `vars`, minus some small epsilon.
  - `max::Array{Float64}`: Maximum values for each variable listed in `vars`, plus some small epsilon.

See also [`profiling`](@ref).
"""
function minMaxValuesReSim(vars::Array{String}, modelName::String, pathToMo::String, pathToOmc::String; workingDir::String = pwd())::Tuple{Array{Float64},Array{Float64}}

  # FIXME don't simulate twice and use mat instead
  # But the MAT.jl doesn't work with v4 mat files.....
  omcWorkingDir = abspath(joinpath(workingDir, modelName))
  (_,_,result_file_csv) = simulateWithProfiling(modelName=modelName,
                                                pathToMo=pathToMo,
                                                pathToOmc=pathToOmc,
                                                tempDir = omcWorkingDir,
                                                outputFormat="csv")
  (min, max) = minMaxValues(vars, epsilon=0.05, resultFile=result_file_csv)

  return (min, max) 
end
