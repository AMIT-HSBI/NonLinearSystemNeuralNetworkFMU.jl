using NonLinearSystemNeuralNetworkFMU

function simulateFMU(modelName, N)
  workdir = datadir("sims", "$(modelName)_$(N)")

  tempdir = joinpath(workdir, "temp-omsimulator")
  logFile = joinpath(tempdir, "oms_call.log")
  mkpath(dirname(tempdir))
  mkpath(dirname(logFile))

  fmu = joinpath(workdir, "$(modelName).fmu")
  @assert isfile(fmu) "FMU $fmu not found"
  resultFile = joinpath(tempdir, "$(modelName)_ref.csv")
  mkpath(dirname(resultFile))
  cmd = `OMSimulator --resultFile=$resultFile --stopTime=2 "$(fmu)"`
  @time begin
    NonLinearSystemNeuralNetworkFMU.omrun(cmd, dir=tempdir, logFile=logFile, timeout=1*60)
  end
  @info "Finished original FMU"

  fmu = joinpath(workdir, "$(modelName).onnx.fmu")
  @assert isfile(fmu) "FMU $fmu not found"
  resultFile = joinpath(tempdir, "$(modelName)_onnx_res.csv")
  cmd = `OMSimulator --resultFile=$resultFile --stopTime=2 "$(fmu)"`
  @time begin
    NonLinearSystemNeuralNetworkFMU.omrun(cmd, dir=tempdir, logFile=logFile, timeout=60*60)
  end

  @info "Finished ONNX FMU"
end
