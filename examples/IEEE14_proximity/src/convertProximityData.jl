using BSON
using CSV
using DataFrames
using NonLinearSystemNeuralNetworkFMU

function convertProximityData(workdir::String, N::Integer, eq::Integer)
  csvFile = joinpath(workdir, "data", "eq_$(eq).csv")
  df = CSV.read(csvFile, DataFrame; ntasks=1)

  dict = BSON.load(joinpath(workdir, "profilingInfo.bson"))
  profilingInfo = Array{ProfilingInfo}(dict[first(keys(dict))])[1]

  # Generate proximity training set
  df_proximity = NonLinearSystemNeuralNetworkFMU.data2proximityData(df, profilingInfo.usingVars, profilingInfo.iterationVariables, neighbors=10, weight=0.05)
  csvFile_proximity = datadir("exp_pro", "eq_$(eq)_$(N)_proximity.csv")
  mkpath(dirname(csvFile_proximity))
  CSV.write(csvFile_proximity, df_proximity)

  return (csvFile_proximity, df_proximity)
end

