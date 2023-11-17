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

import FMICore
using Libdl


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





function fmiEvaluateJacobian(comp::FMICore.FMU2Component, eq::Integer, vr::Array{FMI.fmi2ValueReference}, x::Array{Float64})::Tuple{FMI.fmi2Status, Array{Float64}}
    # wahrscheinlich braucht es eine c-Funktion, die nicht nur einen pointer auf die Jacobi-Matrix returned, sondern gleich die Auswertung an der Stelle x
    # diese Funktion muss auch ein Argument res nehmen welches dann die Evaluation enthält.?
  
    @assert eq>=0 "Residual index has to be non-negative!"
  
    FMIImport.fmi2SetReal(comp, vr, x)
  
    # this is a pointer to Jacobian matrix in row-major-format or NULL in error case.
    fmiEvaluateJacobian = Libdl.dlsym(comp.fmu.libHandle, :myfmi2EvaluateJacobian)
  
    jac = Array{Float64}(undef, length(x)*length(x))
  
    eqCtype = Csize_t(eq)
  
    status = ccall(fmiEvaluateJacobian,
                   Cuint,
                   (Ptr{Nothing}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
                   comp.compAddr, eqCtype, x, jac)
  
    return status, jac
  end




#-------------------------------
# when using residual loss, load fmu
#(status, res) = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu_comp, eq_num, rand(Float64, 110))
#"/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/IEEE_14_Buses.interface.fmu"
fmu = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/unzipped_fmu_sl/simpleLoop.fmu")
comp = FMI.fmiInstantiate!(fmu)
FMI.fmiSetupExperiment(comp)
FMI.fmiEnterInitializationMode(comp)
FMI.fmiExitInitializationMode(comp)

vr = FMI.fmiStringToValueReference(fmu, ["x","y"])
# status, jac = fmiEvaluateJacobian(comp, 0, vr, [1.,2.])

# jac


eq_num = 0 # hardcoded but okay

profilinginfo = getProfilingInfo("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/profilingInfo.bson")

row_vr = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].usingVars)
row_vr_y = FMI.fmiStringToValueReference(fmu.modelDescription, profilinginfo[1].iterationVariables)


function prepare_x(x, y, row_vr, row_vr_y, fmu)
  FMIImport.fmi2SetReal(fmu, row_vr, x[1:end])
  FMIImport.fmi2SetReal(fmu, row_vr_y, y)
#   if time !== nothing
#     FMIImport.fmi2SetTime(fmu, x[1])
#     FMIImport.fmi2SetReal(fmu, row_vr, x[2:end])
#   else
#     FMIImport.fmi2SetReal(fmu, row_vr, x[1:end])
#   end
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
  status, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, eq_num, vr, Float64.(x))
  mat_dim = trunc(Int,sqrt(length(jac)))
  jac = reshape(jac, (mat_dim,mat_dim))
  println(jac)
  #jac = rand(110,110) # jacobian dresidual/dx, still need that, probably of shape (110x16), what shape is that?
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





fileName = "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/data/eq_16.csv"
nInputs = 2
nOutputs = 2

# prepare train and test data
(train_in, train_out, test_in, test_out) = readData(fileName, nInputs)

dataloader = Flux.DataLoader((train_in, train_out), batchsize=1, shuffle=true)

# specify network architecture
# maybe add normalization layer at Start
model = Flux.Chain(Flux.Dense(nInputs, 32, relu),
                  Flux.Dense(32, nOutputs))

ps = Flux.params(model)
opt = Flux.Adam(1e-3)
opt_state = Flux.setup(opt, model)


println(loss(model(train_in[1]), train_out[1], fmu, eq_num)[1])

# problem: geht nur mit batchsize = 1
losses = []
for epoch in 1:5
    for (x, y) in dataloader
        print(x[1], y[1])
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
        println(grads[1])
        Flux.update!(opt_state, model, grads[1])
        break
    end
end

println(loss(model(train_in[1]), train_out[1], fmu, eq_num)[1])


# loss(rand(110,), rand(110,), fmu, eq_num)
# rrule(loss, rand(110,), rand(110,), fmu, eq_num)
