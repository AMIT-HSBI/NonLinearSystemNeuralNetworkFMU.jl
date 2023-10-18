using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU
using ChainRulesCore
using Zygote
using BSON
using Flux
using LinearAlgebra
using FMI
using FMIImport

using Statistics
using Plots
using Metrics

import DataFrames
import CSV
import InvertedIndices
import StatsBase


function readData(filename::String, nInputs::Integer; ratio=0.8, shuffle::Bool=true)
    df = DataFrames.select(CSV.read(filename, DataFrames.DataFrame; ntasks=1), InvertedIndices.Not([:Trace]))
    m = Matrix{Float32}(df)
    n = length(m[:,1]) # num samples
    num_train = Integer(round(n*ratio))
    if shuffle
      trainIters = StatsBase.sample(1:n, num_train, replace = false)
    else
      trainIters = 1:num_train
    end
    testIters = setdiff(1:n, trainIters)

    train_in  = [m[i, 1:nInputs]     for i in trainIters]
    train_out = [m[i, nInputs+1:end] for i in trainIters]
    test_in   = [m[i, 1:nInputs]     for i in testIters]
    test_out  = [m[i, nInputs+1:end] for i in testIters]
    return train_in, train_out, test_in, test_out
end




#-------------------------------
# when using residual loss, load fmu
#(status, res) = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu_comp, eq_num, rand(Float64, 110))
fmu = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/IEEE_14_Buses.interface.fmu")
FMI.fmiInstantiate!(fmu) # this or only load?
FMI.fmiSetupExperiment(fmu)
FMI.fmiEnterInitializationMode(fmu)
FMI.fmiExitInitializationMode(fmu)
eq_num = 1403 # hardcoded but okay

profilinginfo = getProfilingInfo("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/profilingInfo.bson")

row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].usingVars)
row_vr_y = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].iterationVariables)


function prepare_x(x, y, row_vr, row_vr_y, fmu)
  FMIImport.fmi2SetReal(fmu, row_vr, x[2:end])
  FMIImport.fmi2SetReal(fmu, row_vr_y, y)
  #if time !== nothing
  FMIImport.fmi2SetTime(fmu, x[1])
  #end
end

function loss(y_hat, y, fmu, eq_num)
  res_out = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu, eq_num, Float64.(y_hat))
  return LinearAlgebra.norm(res_out), res_out
end


# rrule for loss(x,y)
function ChainRulesCore.rrule(::typeof(loss), x, y, fmu, eq_num)
  # x is model output, y is target
  l, res_out = loss(x, y, fmu, eq_num) # res_out: residual output, what shape is that?

  # maybe like this: status, res = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(fmu, eq_num, Float64.(x))?
  jac = rand(110,110) # jacobian dresidual/dx, still need that, probably of shape (110x16), what shape is that?
  # IST DIE QUADRATISCH?

  # x̄ (110,) so wie model output

  function loss_pullback(l̄)
    # print(l̄[1])
    # print(size(res_out[2]))
    # print(size(jac * res_out[2]))
    # print(size(l̄[1] * ((jac * res_out[2]) / l)))
    f̄ = NoTangent()
    # https://math.stackexchange.com/questions/291318/derivative-of-the-2-norm-of-a-multivariate-function
    x̄ = l̄[1] * ((jac' * res_out[2]) / l) # <-------------- ACTUAL derivative, result should be of shape (110,)
    # res_out[2] (110,) jac' (110,110) jac'*res_out[2] (110,) x̄ (110,)
    ȳ = ZeroTangent()
    fmū = NoTangent()
    eq_num̄ = NoTangent()
    #print(size(x̄), size(jac'), size(res_out[2]))
    return (f̄, x̄, ȳ, fmū, eq_num̄)
  end

  return l, loss_pullback
end
#-------------------------------





fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/data/eq_1403.csv"
nInputs = 16
nOutputs = 110

# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)

dataloader = Flux.DataLoader((train_in, train_out), batchsize=1, shuffle=true)

# specify network architecture
# maybe add normalization layer at Start
model = Flux.Chain(Flux.Dense(nInputs, 32, relu),
                  Flux.Dense(32, 32, relu),
                  Flux.Dense(32, 32, relu),
                  Flux.Dense(32, nOutputs))

ps = Flux.params(model)
opt = Flux.Adam(1e-3)
opt_state = Flux.setup(opt, model)



# problem: geht nur mit batchsize = 1
losses = []
for epoch in 1:1
    for (x, y) in dataloader
        #print(size(x), size(x[1]), size(y), size(y[1]))
        prepare_x(x[1], y[1], row_vr, row_vr_y, fmu)
        grads = Flux.gradient(model) do m  
          result = m(x[1])
          #print(size(result))
          loss(result, y[1], fmu, eq_num)
          #Flux.mse(result, y[1])
          #return l
        end
        
        #push!(losses, l)  # logging, outside gradient context
        Flux.update!(opt_state, model, grads[1])
        break
    end
end



loss(rand(110,), rand(110,), fmu, eq_num)
rrule(loss, rand(110,), rand(110,), fmu, eq_num)
