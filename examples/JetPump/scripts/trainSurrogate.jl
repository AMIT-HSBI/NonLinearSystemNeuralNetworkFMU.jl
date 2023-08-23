using DrWatson
@quickactivate "JetPump"

using NonLinearSystemNeuralNetworkFMU
using CSV
using DataFrames
using LinearAlgebra
using Random
Random.seed!(100)
include(srcdir("symbolicRegression.jl"))

# JetPump training data
modelName = "Scenario_01"
n = 100_000
workdir = datadir("sims", "$(modelName)_$(n)")
eq = 46
dataFile = joinpath(workdir, "data", "eq_$(eq).csv")
df = CSV.read(dataFile, DataFrame)

# Get m random data points of total training set
m = 1000
idx_row = shuffle(1:size(df,1))
data = df[idx_row[1:m],:]

prof = getProfilingInfo(joinpath(workdir, "profilingInfo.bson"))

n_inputs = length(prof[1].usingVars)
n_outputs = length(prof[1].iterationVariables)

X = transpose(Matrix{Float32}(data[:,1:n_inputs]))
Y = transpose(Matrix{Float32}(data[:,n_inputs+1:n_inputs+n_outputs]))

# Symbolic Regression
bestEq = fitEquation(X, Y; tempDir=joinpath(workdir, "temp-symbolicRegression"))

function genJuliaFunc(equation::Array{Any},
                      inputs::Array{String},
                      outputs::Array{String},
                      eq::Int,
                      outFile::String)

  open(outFile,"w") do file
    write(file, """
    function eq_$eq(u::Array{Float64,1}, y:::Array{Float64,1})
      @assert size(u) == (1, $(length(inputs))) "Dimension mismatch input u"
      @assert size(y) == (1, $(length(outputs))) "Dimension mismatch output y"
    """
    )

    for (i,inputName) in enumerate(inputs)
      write(file, """
        x$(i) = u[$(i)] # $(inputName)
      """
      )
    end

    for (i,eq) in enumerate(equation)
      outputName = outputs[i]
      write(file, """
        y[$(i)] #= $(outputName) =# = $(string(eq))
      """
      )
    end

    write(file, """
    end
    """
    )
  end
end

function genMoCode(equation::Array{Any},
                   inputs::Array{String},
                   outputs::Array{String};
                   replaceDots::Bool = false)

  if replaceDots
    inputs = replace.(inputs, "." => "_")
    outputs = replace.(outputs, "." => "_")
  end
  equations = Array{String}(undef, length(equation))
  for (i, eq) in enumerate(equation)
    eq = string(eq)
    for (j, inputName) in enumerate(inputs)
      eq = replace(eq, "x$j"=>inputName)
    end
    equations[i] = "$(outputs[i]) = $(string(eq));"
  end

  return equations
end

surrogatEqns = genMoCode(bestEq, prof[1].usingVars, prof[1].iterationVariables)
println.(surrogatEqns);

surrogatEqnsMo = joinpath(workdir, "surrogatEqns.mo")
open(surrogatEqnsMo,"w") do file
  write(file, join(surrogatEqns, "\n"))
end

# TODO:
# Automate replacing residual equations with explicit modelica fucntions in flat model.
