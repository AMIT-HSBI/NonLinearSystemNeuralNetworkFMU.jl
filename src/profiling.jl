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
    getomc([omc])

Find omc executable from `PATH`, `OPENMODELICAHOME` or given string.
"""
function getomc(omc::String="")

  # Find omc if nothing is provided
  if isempty(strip(omc))
    if Sys.iswindows()
      @assert(haskey(ENV, "OPENMODELICAHOME"), "Environment variable OPENMODELICAHOME not set.")
      omc = abspath(joinpath(ENV["OPENMODELICAHOME"], "bin", "omc.exe"))
      if !isfile(omc)
        throw(ProgramNotFoundError("omc", [omc]))
      end
    else
      omc = string(strip(read(`which omc`, String)))
    end
  end

  # Check minimum version
  testOmcVersion(omc)

  return omc
end


"""
    simulateWithProfiling(modelName, pathToMo; [pathToOmc], workingDir=pwd(), outputFormat="mat", clean=false])

Simulate Modelica model with profiling enabled using given omc.

# Arguments
- `modelName::String`:  Name of the Modelica model.
- `moFiles::Array{String}`:   Path to the *.mo file(s) containing the model.

# Keywords
  - `pathToOmc::String=""`:       Path to omc used for simulating the model.
                                  Use omc from `PATH`/`OPENMODELICAHOME` if nothing is provided.
  - `workingDir::String=pwd()`:   Working directory for omc. Defaults to the current directory.
  - `outputFormat::String="mat"`: Output format for result file. Can be `"mat"` or `"csv"`.
  - `clean::Bool=false`:          Remove everything in `workingDir` when set to `true`.
"""
function simulateWithProfiling(modelName::String,
                               moFiles::Array{String};
                               pathToOmc::String="",
                               workingDir::String=pwd(),
                               outputFormat::String="mat",
                               clean::Bool=false)

  pathToOmc = getomc(pathToOmc)

  @assert outputFormat === "mat" || outputFormat === "csv"  "Output format has to be \"mat\" or \"csv\"."

  if !isdir(workingDir)
    mkpath(workingDir)
  elseif clean
    rm(workingDir, force=true, recursive=true)
    mkpath(workingDir)
  end

  if Sys.iswindows()
    pathToMo = replace.(moFiles, "\\"=> "\\\\")
    workingDir = replace(workingDir, "\\"=> "\\\\")
  end

  logFilePath = joinpath(workingDir,"calls.log")
  logFile = open(logFilePath, "w")

  local omc
  Suppressor.@suppress begin
    omc = OMJulia.OMCSession(pathToOmc)
  end
  try
    msg = OMJulia.sendExpression(omc, "getVersion()")
    write(logFile, msg*"\n")
    for file in moFiles
      msg = OMJulia.sendExpression(omc, "loadFile(\"$(file)\")")
      @assert msg==true "Failed to load file $(file)!"
      write(logFile, string(msg)*"\n")
      msg = OMJulia.sendExpression(omc, "getErrorString()")
      write(logFile, msg*"\n")
    end
    OMJulia.sendExpression(omc, "cd(\"$(workingDir)\")")

    @debug "setCommandLineOptions"
    msg = OMJulia.sendExpression(omc, "setCommandLineOptions(\"-d=newInst,infoXmlOperations,backenddaeinfo --profiling=all\")")
    write(logFile, string(msg)*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")

    @debug "simulate"
    msg = OMJulia.sendExpression(omc, "simulate($(modelName), outputFormat=\"$(outputFormat)\", simflags=\"-lv=LOG_STATS -clock=RT -cpu -w\")")
    write(logFile, msg["messages"]*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")
  catch e
    @error "Failed to simulate $modelName."
    rethrow(e)
  finally
    close(logFile)
    OMJulia.sendExpression(omc, "quit()", parsed=false)
  end

  profJsonFile = abspath(joinpath(workingDir, modelName*"_prof.json"))
  infoJsonFile = abspath(joinpath(workingDir, modelName*"_info.json"))
  resultFile = abspath(joinpath(workingDir, modelName*"_res."*outputFormat))
  if !(isfile(profJsonFile) && isfile(infoJsonFile) && isfile(resultFile))
    throw(OpenModelicaError("Simulation failed, no files generated.", abspath(logFilePath)))
  end
  return (profJsonFile, infoJsonFile, resultFile)
end


"""
    isnonlinearequation(eq)

Return `true` if equation system is non-linear.
"""
function isnonlinearequation(eq::Dict{String, Any})
  if eq["tag"] == "tornsystem" && eq["display"] == "non-linear"
    return true
  end
  if eq["tag"] == "system" && eq["display"] == "non-linear"
    return true
  end
  return false
end


"""
    isinitial(eq)

Return `true` is equation is part of the initial system.
"""
function isinitial(eq::Dict{String, Any})
  if eq["section"] == "initial"
    return true
  end
  if eq["section"] == "initial-lambda0"
    return true
  end
  return false
end


"""
    findSlowEquations(profJsonFile, infoJsonFile; threshold)

Read JSON profiling file and find slowest non-linear loop equatiosn that need more then `threshold` of total simulation time.

# Arguments

  - `profJsonFile::String`: Path to profiling JSON file.
  - `infoJsonFile::String`: Path to info JSON file.

# Keywords
  - `threshold`: Lower bound on time consumption of equation.
                 0 <= threshold <= 1
  - `ignoreInit::Bool=true`:    Ignore equations from initialization system if `true`.
"""
function findSlowEquations(profJsonFile::String, infoJsonFile::String; threshold = 0.03, ignoreInit::Bool=true)
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
    if bigger && isnonlinearequation(equations[id+1]) && !(ignoreInit && isinitial(equations[id+1]))
      push!(slowesEq, EqInfo(block["id"], block["ncall"], block["time"], block["maxTime"], fraction))
    end
  end

  # Workaround for Windows until https://github.com/JuliaIO/JSON.jl/issues/347 is fixed.
  GC.gc()

  return slowesEq
end


"""
    findUsedVars(infoFile, eqIndex; filterParameters = true)

Read `infoFile` and return `(definingVars, usingVars)` that are defined or used by equation with `eqIndex`.
"""
function findUsedVars(infoFile, eqIndex; filterParameters::Bool = true)::Tuple{Array{String}, Array{String}}
  equations = infoFile["equations"]
  eq = (equations[eqIndex+1])
  variables = infoFile["variables"]

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

  # Remove parameter and external objects used vars
  removeVars = String[]
  notFoundVars = String[]
  for usedVar in usingVars
    if haskey(variables, usedVar)
      var = variables[usedVar]
      if filterParameters && var["kind"] == "parameter"
        push!(removeVars, usedVar)
      end
      if filterParameters && var["kind"] == "constant"
        push!(removeVars, usedVar)
      end
      if var["kind"] == "external object"
        push!(removeVars, usedVar)
      end
    else
      @error "Variable $usedVar not found"
      push!(removeVars, usedVar)
      push!(notFoundVars, usedVar)
    end
  end
  @debug "Removed $removeVars from usingVars"
  setdiff!(usingVars, removeVars)

  # Expand array variables

  return (definingVars, usingVars)
end


"""
    findDependentVars(jsonFile, eqIndex)

Read JSON info file `jsonFile` and return iteration, inner and used variables
`(iterationVariables, innerEquations, usingVars)`.
"""
function findDependentVars(jsonFile::String, eqIndex)::Tuple{Array{String}, Array{Int64}, Array{String}}
  infoFile = JSON.parsefile(jsonFile)

  equations = infoFile["equations"]
  eq = (equations[eqIndex+1])

  if eq["eqIndex"] != eqIndex
    error("Found wrong equation")
  end

  iterationVariables = Array{String}(eq["defines"])
  loopEquations = collect(Iterators.flatten(eq["equation"]))
  innerEquations = Int64[]

  usingVars = String[]
  innerVars = String[]

  for loopeq in loopEquations
    if infoFile["equations"][loopeq+1]["tag"] =="jacobian"
      continue
    end
    (def, use) = findUsedVars(infoFile, loopeq)
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
    minMaxValues(resultFile, variables; epsilon=0.05)


Read CSV `resultFile` to find smallest and biggest value for each variable in `variables`.
Add ± `epsilon` to minimum and maximum.
"""
function minMaxValues(resultFile::String, variables::Array{String}; epsilon=0.05)
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
    profiling(modelName, moFiles; pathToOmc, workingDir, threshold = 0.03)

Find equations of Modelica model that are slower then threashold.

# Arguments
  - `modelName::String`:  Name of the Modelica model.
  - `moFiles::Array{String}`:   Path to the *.mo file(s) containing the model.

# Keywords
  - `pathToOm::String=""`:      Path to omc used for simulating the model.
                                Use omc from PATH/OPENMODELICAHOME if nothing is provided.
  - `workingDir::String=pwd()`: Working directory for omc. Defaults to the current directory.
  - `threshold=0.01`:           Slowest equations that need more then `threshold` of total simulation time.
  - `ignoreInit::Bool=true`:    Ignore equations from initialization system if `true`.

# Returns
  - `profilingInfo::Vector{ProfilingInfo}`: Profiling information with non-linear equation systems slower than `threshold`.
"""
function profiling(modelName::String,
                   moFiles::Array{String};
                   pathToOmc::String="",
                   workingDir::String=pwd(),
                   threshold=0.01,
                   ignoreInit::Bool=true)::Vector{ProfilingInfo}

  omcWorkingDir = abspath(joinpath(workingDir, modelName))
  (profJsonFile, infoJsonFile, _) = simulateWithProfiling(modelName,
                                                          moFiles;
                                                          pathToOmc = pathToOmc,
                                                          workingDir = omcWorkingDir)

  slowestEqs = findSlowEquations(profJsonFile, infoJsonFile; threshold=threshold, ignoreInit=ignoreInit)

  profilingInfo = Array{ProfilingInfo}(undef, length(slowestEqs))

  for (i,slowEq) in enumerate(slowestEqs)
    (iterationVariables, innerEquations, usingVars) = findDependentVars(infoJsonFile, slowestEqs[i].id)
    profilingInfo[i] = ProfilingInfo(slowEq, iterationVariables, innerEquations, usingVars)
  end

  return profilingInfo
end


"""
    minMaxValuesReSim(vars, modelName, moFiles; pathToOmc="" workingDir=pwd())

(Re-)simulate Modelica model and find miminum and maximum value each variable has during simulation.

# Arguments
  - `vars::Array{String}`:    Array of variables to get min-max values for.
  - `modelName::String`:      Name of Modelica model to simulate.
  - `moFiles::Array{String}`: Path to .mo file(s).

# Keywords
  - `pathToOmc::String=""`:     Path to OpenModelica Compiler omc.
  - `workingDir::String=pwd()`: Working directory for omc. Defaults to the current directory.

# Returns
  - `min::Array{Float64}`: Minimum values for each variable listed in `vars`, minus some small epsilon.
  - `max::Array{Float64}`: Maximum values for each variable listed in `vars`, plus some small epsilon.

See also [`profiling`](@ref).
"""
function minMaxValuesReSim(vars::Array{String},
                           modelName::String,
                           moFiles::Array{String};
                           pathToOmc::String="",
                           workingDir::String=pwd())::Tuple{Array{Float64},Array{Float64}}

  # FIXME don't simulate twice and use mat instead
  # But the MAT.jl doesn't work with v4 mat files.....
  omcWorkingDir = abspath(joinpath(workingDir, modelName))
  (_,_,result_file_csv) = simulateWithProfiling(modelName,
                                                moFiles,
                                                pathToOmc=pathToOmc,
                                                workingDir = omcWorkingDir,
                                                outputFormat="csv")
  (min, max) = minMaxValues(result_file_csv, vars)

  return (min, max)
end
