#
# Copyright (c) 2022 Andreas Heuermann
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
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
    msg = OMJulia.sendExpression(omc, "simulate($(modelName), outputFormat=\"$(outputFormat)\", simflags=\"-lv=LOG_STATS -clock=CPU -cpu -w\")")
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
Read JSON info file and find all variables needed for equation with index `eqIndex`.
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

  usingVars = String[]
  innerVars = String[]

  for loopeq in loopEquations
    if infoFile["equations"][loopeq+1]["tag"] =="jacobian"
      continue
    end
    (def, use) =findUsedVars(infoFile, loopeq)
    append!(usingVars, use)
    append!(innerVars, def)
  end

  for v in vcat(innerVars, iterationVariables)
    deleteat!(usingVars, findall(x->x==v, usingVars))
  end

  return (unique(iterationVariables), loopEquations, unique(usingVars))
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
  - `workingDir::String`: Working directory for omc.

# Keywords
  - `threshold`: Slowest equations that need more then `threshold` of total simulation time.
"""
function profiling(modelName::String, pathToMo::String, pathToOmc::String, workingDir::String; threshold = 0.01)

  omcWorkingDir = abspath(joinpath(workingDir, modelName))
  (profJsonFile, infoJsonFile, _) = simulateWithProfiling(modelName=modelName,
                                                          pathToMo=pathToMo,
                                                          pathToOmc=pathToOmc,
                                                          tempDir = omcWorkingDir,
                                                          outputFormat="mat")

  slowestEqs = findSlowEquations(profJsonFile, infoJsonFile; threshold=threshold)

  profilingInfo = Array{ProfilingInfo}(undef, length(slowestEqs))

  for (i,slowEq) in enumerate(slowestEqs)
    (iterationVariables, loopEquations, usingVars) = findDependentVars(infoJsonFile, slowestEqs[1].id)
    profilingInfo[i] = ProfilingInfo(slowEq, iterationVariables, loopEquations, usingVars)
  end

  return profilingInfo
end


function minMaxValuesReSim(vars::Array{String}, modelName::String, pathToMo::String, pathToOmc::String, workingDir::String)

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

