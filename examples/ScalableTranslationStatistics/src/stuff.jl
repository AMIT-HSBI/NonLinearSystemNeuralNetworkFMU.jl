
function DataFrame(result::FMICore.FMU2Solution{FMU2}, names::Array{String})
  df = DataFrames.DataFrame(time=result.values.t)
  for i in 1:length(result.values.saveval[1])
    df[!, Symbol(names[i])] = [val[i] for val in result.values.saveval]
  end
  return df
end


function simulateFMU(pathToFMU, resultFile)
  mkpath(dirname(resultFile))

  outputVars = ["outputs[1]", "outputs[2]", "outputs[3]", "outputs[4]", "outputs[5]", "outputs[6]", "outputs[7]", "outputs[8]"]

  fmu = fmiLoad(pathToFMU)
  result = fmiSimulate(fmu, 0.0, 10.0; recordValues=outputVars)
  df = DataFrame(result, outputVars)

  bench = @benchmark fmiSimulate($fmu, 0.0, 10.0; recordValues=$outputVars) samples=5 seconds=120*5 evals=1
  fmiUnload(fmu)

  CSV.write(resultFile, df)
  return bench
end
