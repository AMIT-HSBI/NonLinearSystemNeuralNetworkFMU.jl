using BenchmarkTools
using CSV
using DataFrames
import FMI
import FMICore

function DataFrame(result::FMICore.FMU2Solution{FMICore.FMU2Component{FMICore.FMU2}}, names::Array{String})
  df = DataFrames.DataFrame(time=result.values.t)
  for i in 1:length(result.values.saveval[1])
    df[!, Symbol(names[i])] = [val[i] for val in result.values.saveval]
  end
  return df
end

"""
Benchmark simulation of FMU with FMI.jl

Use FMI.jl to simulate FMUs and save results given by `outputVars`.
"""
function simulateFMU_FMIjl(pathToFMU::String, resultFile::String; outputVars::Array{String})
  @info "Simulating FMU $pathToFMU with FMI.jl"

  fmu = FMI.fmiLoad(pathToFMU)
  result = FMI.fmiSimulate(fmu, (0.0, 10.0); recordValues=outputVars)
  df = DataFrame(result, outputVars)

  bench = @benchmark FMI.fmiSimulate($fmu, (0.0, 10.0); recordValues=$outputVars) samples=5 seconds=120*5 evals=1   # Use $() to interpolate external variables
  FMI.fmiUnload(fmu)

  mkpath(dirname(resultFile))
  CSV.write(resultFile, df)
  return bench
end

"""
Benchmark simulation of FMU with OMSimulator
"""
function simulateFMU_OMSimulator(luaFile::String, pathToFMU::String, modelName::AbstractString, resultFile::String; workdir::String, logFile::String, samples=1)
  @info "Simulating FMU $pathToFMU with OMSimulator"

  local bench
  mkpath(dirname(logFile))
  mkpath(dirname(resultFile))
  mkpath(dirname(workdir))

  script = generateLuaScriptOMS(luaFile, pathToFMU, modelName; stepSize=1e-3, workdir=workdir, resultFile=resultFile, stopTime=10)
  cmd = `OMSimulator --deleteTempFiles=true --stripRoot=true "$(script)"`
  try
    redirect_stdio(stdout=logFile, stderr=logFile) do
     bench = @benchmark run(Cmd($cmd, dir=$workdir)) samples=samples seconds=60*10 evals=1   # Use $() to interpolate external variables
   end
  catch e
    @error "Printing log file: $logFile"
    @error read(logFile, String)
    rethrow(e)
  end
  display(bench)
  return bench
end

function generateLuaScriptOMS(luaFile::String,
                              pathToFMU::String,
                              modelName::AbstractString;
                              stepSize::Float64,
                              workdir::String,
                              resultFile::String,
                              stopTime=1::Float64)

  content = """
    oms_setTempDirectory("$(workdir)")
    oms_newModel("model")
    oms_addSystem("model.root", oms_system_sc)

    -- instantiate FMUs
    oms_addSubModel("model.root.$(modelName)", "$(pathToFMU)")

    -- simulation settings
    oms_setSolver("model", oms_solver_sc_explicit_euler)
    oms_setResultFile("model", "$(resultFile)")
    oms_setStopTime("model", $(stopTime))
    oms_setFixedStepSize("model.root", $(stepSize))

    oms_instantiate("model")

    oms_initialize("model")
    oms_simulate("model")
    oms_terminate("model")
    oms_delete("model")
    """

    open(luaFile, "w") do file
      write(file, content)
    end

    return luaFile
end