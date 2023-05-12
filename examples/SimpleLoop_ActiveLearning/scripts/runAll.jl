using DrWatson
@quickactivate "SimpleLoop_ActiveLearning"

include(srcdir("activeLearn.jl"))

modelName = "simpleLoop"
N = 1_000

function mymain(modelName::String, N::Integer)
  moFiles = [srcdir("$modelName.mo")]

  omOptions = OMOptions(
    workingDir=datadir("sims", "$(modelName)_$(N)"),
    clean=false,
    commandLineOptions="--preOptModules-=wrapFunctionCalls --postOptModules-=wrapFunctionCalls"
  )

  reuseArtifacts = true

  mkpath(omOptions.workingDir)

  # Profiling and min-max values
  profilingInfoFile = joinpath(omOptions.workingDir, "profilingInfo.bson")
  profOptions = OMOptions(pathToOmc=omOptions.pathToOmc,
    workingDir=joinpath(omOptions.workingDir, "temp-profiling"),
    outputFormat="csv",
    clean=omOptions.clean,
    commandLineOptions=omOptions.commandLineOptions)
  local profilingInfo
  if (reuseArtifacts && isfile(profilingInfoFile))
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
  genFmuOptions = OMOptions(pathToOmc=omOptions.pathToOmc,
    workingDir=joinpath(omOptions.workingDir, "temp-fmu"),
    outputFormat=nothing,
    clean=omOptions.clean,
    commandLineOptions=omOptions.commandLineOptions)
  fmuFile = joinpath(omOptions.workingDir, modelName * ".fmu")
  local fmu
  if (reuseArtifacts && isfile(fmuFile))
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
  fmuFile = joinpath(omOptions.workingDir, modelName * ".interface.fmu")
  local fmu
  if (reuseArtifacts && isfile(fmuFile))
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

  # Train models
  @info "Train models"
  csvFiles = String[]
  for prof in profilingInfo
    eqIndex = prof.eqInfo.id
    inputVars = prof.usingVars
    outputVars = prof.iterationVariables
    minBoundary = prof.boundary.min
    maxBoundary = prof.boundary.max
    
    # Das interface hier sollte so passen, also keine Änderungen nötig
    # Die beim AL generierten Daten werden in csv geschrieben und man kann später nachvollziehen wo trainiert wurde
    csvFile = abspath(joinpath(omOptions.workingDir, "data", "eq_$(prof.eqInfo.id).csv"))
    activeLearn(fmu_interface, csvFile, eqIndex, inputVars, minBoundary, maxBoundary, outputVars; options=dataGenOptions)
    push!(csvFiles, csvFile)
  end
  if omOptions.clean
    rm(tempDir, force=true, recursive=true)
  end

  return csvFiles, fmu, profilingInfo
end


mymain(modelName, N)