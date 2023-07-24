using DrWatson
@quickactivate "JetPump"

using NonLinearSystemNeuralNetworkFMU
using DataFrames
using CSV
using LinearAlgebra
using Random
Random.seed!(100)
include(srcdir("symbolicRegression.jl"))

# JetPump training data
modelName = "JetPumpInverse"
N = 100_000
workdir = datadir("sims", "$(modelName)_$(N)")
eq = 101
dataFile = joinpath(workdir, "data", "eq_$(eq).csv")
df = CSV.read(dataFile, DataFrame)

# Get n random data points of total training set
n = 1000
idx_row = shuffle(1:size(df,1))
data = df[idx_row[1:n],:]

prof = getProfilingInfo(joinpath(workdir, "profilingInfo.bson"))

n_inputs = length(prof[1].usingVars)            # 4
n_outputs = length(prof[1].iterationVariables)  # 6

X = transpose(Matrix{Float32}(data[:,1:n_inputs]))
Y = transpose(Matrix{Float32}(data[:,n_inputs+1:n_inputs+n_outputs]))
#y = Y[1,:]

# Symbolic Regression
bestEq = fitEquation(X, Y; tempDir=joinpath(workdir, "temp-symbolicRegression"))

# Code generation with SymbolicUtils.jl
function generateFunc(bestEq)
  @syms x1 x2 x3 x4

  func = Func[]

  for i in eachindex(bestEq)
    push!(func,Func([x1, x2, x3, x4], # args
                    [],               # kwargs
                    bestEq[i]))
  end

  f1 = eval(toexpr(func[1]))
  f2 = eval(toexpr(func[2]))
  f3 = eval(toexpr(func[3]))
  f4 = eval(toexpr(func[4]))
  f5 = eval(toexpr(func[5]))
  f6 = eval(toexpr(func[6]))

  f = (x1,x2,x3,x4) -> [f1(x1,x2,x3,x4), f2(x1,x2,x3,x4), f3(x1,x2,x3,x4), f4(x1,x2,x3,x4), f5(x1,x2,x3,x4), f6(x1,x2,x3,x4)]
  return f
end

f = generateFunc(bestEq)

# Test reference solution
refSol = CSV.read(srcdir("referenceSolution.csv"), DataFrame)
x_ref = Matrix{Float32}(refSol[1:1,1:n_inputs])
y_ref = Matrix{Float32}(refSol[1:1,n_inputs+1:n_inputs+n_outputs])

abs_err = f(x_ref[1], x_ref[2], x_ref[3], x_ref[4]) - y_ref[1,:]
@info abs_err
rel_error = norm(abs_err)/norm(y_ref)
@info rel_error
