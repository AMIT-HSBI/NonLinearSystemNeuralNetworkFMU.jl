"""
Simulate Modelica model with profiling enabled using given omc.
"""
function simulateWithProfiling(;model_name,
                                path_to_mo,
                                path_to_omc,
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

  omc = OMJulia.OMCSession(path_to_omc)
  try
    msg = OMJulia.sendExpression(omc, "getVersion()")
    write(logFile, msg*"\n")
    OMJulia.sendExpression(omc, "loadFile(\"$(path_to_mo)\")")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")
    OMJulia.sendExpression(omc, "cd(\"$(tempDir)\")")

    @info "setCommandLineOptions"
    msg = OMJulia.sendExpression(omc, "setCommandLineOptions(\"-d=newInst,infoXmlOperations,backenddaeinfo --profiling=all\")")
    write(logFile, string(msg)*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")

    @info "simulate"
    msg = OMJulia.sendExpression(omc, "simulate($(model_name), outputFormat=\"$(outputFormat)\", simflags=\"-lv=LOG_STATS -clock=CPU -cpu -w\")")
    write(logFile, msg["messages"]*"\n")
    msg = OMJulia.sendExpression(omc, "getErrorString()")
    write(logFile, msg*"\n")
  finally
    close(logFile)
    OMJulia.sendExpression(omc, "quit()", parsed=false)
  end

  prof_json_file = abspath(joinpath(tempDir, model_name*"_prof.json"))
  info_json_file = abspath(joinpath(tempDir, model_name*"_info.json"))
  result_file = abspath(joinpath(tempDir, model_name*"_res."*outputFormat))
  return (prof_json_file, info_json_file, result_file)
end


"""
Read JSON profiling file and find slowest equation that need more then `threshold` of total simulation time.
"""
function findSlowEquations(json_file; threshold = 0.03)
  profileFile = JSON.parsefile(json_file)

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
    bigger = fraction >= threshold
    if bigger
      push!(slowesEq, EqInfo(block["id"], block["ncall"], block["time"], block["maxTime"], fraction) )
    end
  end

  return slowesEq
end


"""
Return variables that are defined by equation with `eqIndex`.
"""
function findUsedVars(infoFile, eqIndex; filter_parameters = true)
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
  if filter_parameters && usingVars !==nothing
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
function findDependentVars(json_file, eqIndex)
  infoFile = JSON.parsefile(json_file)

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
function minMaxValues(variables::Array{String}; epsilon=0.05, result_file)
  df = DataFrames.DataFrame(CSV.File(result_file))

  min = Array{Float64}(undef, length(variables))
  max = Array{Float64}(undef, length(variables))
  for (i,var) in enumerate(variables)
    min[i] = minimum(df[!,var]) - abs(epsilon*minimum(df[!,var]))
    max[i] = maximum(df[!,var]) + abs(epsilon*maximum(df[!,var]))
  end

  return min, max
end


"""
    profiling(model_name, path_to_mo, path_to_omc, working_dir; threshold = 0.03)

Find equations of Modelica model that are slower then threashold.

`model_name` is full name of the Modelica model.
`path_to_mo` is the path to the *.mo file containing the model.
`path_to_omc` is the path to omc used for simulating the model.
`working_dir` is the working directory for omc.
`threshold` slowest equations that need more then `threshold` of total simulation time.
"""
function profiling(model_name::String, path_to_mo::String, path_to_omc::String, working_dir::String; threshold = 0.01)

  omc_working_dir = abspath(joinpath(working_dir, model_name))
  (prof_json_file, info_json_file, _) = simulateWithProfiling(model_name=model_name,
                                                              path_to_mo=path_to_mo,
                                                              path_to_omc=path_to_omc,
                                                              tempDir = omc_working_dir,
                                                              outputFormat="mat")

  slowestEqs = findSlowEquations(prof_json_file; threshold=threshold)

  profilingInfo = Array{ProfilingInfo}(undef, length(slowestEqs))

  for (i,slowEq) in enumerate(slowestEqs)
    (iterationVariables, loopEquations, usingVars) = findDependentVars(info_json_file, slowestEqs[1].id)
    profilingInfo[i] = ProfilingInfo(slowEq, iterationVariables, loopEquations, usingVars)
  end

  return profilingInfo
end


function minMaxValuesReSim(vars::Array{String}, model_name::String, path_to_mo::String, path_to_omc::String, working_dir::String)

  # FIXME don't simulate twice and use mat instead
  # But the MAT.jl doesn't work with v4 mat files.....
  omc_working_dir = abspath(joinpath(working_dir, model_name))
  (_,_,result_file_csv) = simulateWithProfiling(model_name=model_name,
                                                path_to_mo=path_to_mo,
                                                path_to_omc=path_to_omc,
                                                tempDir = omc_working_dir,
                                                outputFormat="csv")
  (min, max) = minMaxValues(vars, epsilon=0.05, result_file=result_file_csv)

  return (min, max) 
end

