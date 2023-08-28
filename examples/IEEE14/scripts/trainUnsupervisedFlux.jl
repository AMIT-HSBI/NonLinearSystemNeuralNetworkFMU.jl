using DrWatson
@quickactivate "IEEE14"

using NonLinearSystemNeuralNetworkFMU
using BSON
using Flux
using LinearAlgebra
using FMI
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

# loading a saved model
# eq_1403_out.bson
function getNN(model_path="eq_1403_out.bson")
  outname = abspath(model_path)
  dict = BSON.load(outname, @__MODULE__)  # SafeTestsets puts everything into a module
  return dict[first(keys(dict))]
end

# compute test loss
function testLoss(test_in, test_out, loss_f)
  model = getNN()
  disp_loss() = sum(loss_f.(test_in,test_out))/length(test_in)
  @info "Test loss = $(disp_loss())"
end


function trainSurrogate!(model, dataloader::Flux.DataLoader, train_in, train_out, test_in, test_out, comp, eq, savename::String; losstol::Real=1e-6, nepochs=100, eta=1e-3)
    # Create directory for bson file
    mkpath(dirname(savename))

    # specify optimizer
    ps = Flux.params(model)
    opt = Flux.Adam(eta)


    #------------------
    # specify loss function, choose from multiple
    # standard supervised loss

    loss(x,y) = Flux.mse(model(x),y)

    #loss(x,y) = LinearAlgebra.norm(model(x) - y)
    # these two should be similar

    # residual loss
    # l2 norm of residual equation output, evaluated at model output
    # doesnt work: undefined symbol

    # loss(x,y) = LinearAlgebra.norm(NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(comp, eq, Float64.(model(x)))) 

    # https://physicsbaseddeeplearning.org/physicalloss.html
    # combined loss:
    # alpha = 0.5
    # beta = 0.5
    # loss(x,y) = alpha * Flux.mse(model(x),y) + beta * LinearAlgebra.norm(NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(comp, eq, Float64.(model(x))))


    #------------------

    # average loss function over training data
    disp_loss() = sum(loss.(train_in,train_out))/length(train_in)

    callback = Flux.throttle(disp_loss, 5)

    # Compute initial loss
    minL = disp_loss()
    @info "Initial loss: $(minL)"
    if minL < losstol
      BSON.@save savename model
      return
    end

    # training for n epochs
    @info "Start training"
    epoch = 0
    try
      while epoch < nepochs
        # one epoch
        epoch += 1
        for (ti,to) in dataloader
          Flux.train!(loss, ps, zip(ti,to), opt, cb=callback)
        end

        # loss after last epoch
        l = disp_loss()
        @info "Epoch $(epoch): Train loss = $(l)"
        if l < minL
          BSON.@save savename model
          minL = l
          if minL < losstol
            @info "Success"
            break
          end
        end
      end
    catch e
      if isa(e, InterruptException)
        @info "Aborted"
      else
        rethrow(e)
      end
    end

    # best train loss after end of training
    @info "Best loss = $(minL)\n\tAfter $(epoch) epochs"

    # compute test loss at end of training
    testLoss(test_in, test_out, loss)
end


function runTrainNNTest(fileName, nInputs, nOutputs, outname)
    fileName = abspath(fileName)
    outname = abspath(outname)
    rm(outname, force = true)

    # prepare train and test data
    (train_in, train_out, test_in, test_out) = readData(fileName, nInputs)

    dataloader = Flux.DataLoader((train_in, train_out), batchsize=64, shuffle=true)

    # specify network architecture
    # maybe add normalization layer at Start
    model = Flux.Chain(Flux.Dense(nInputs, 32, relu),
                      Flux.Dense(32, 32, relu),
                      Flux.Dense(32, 32, relu),
                      Flux.Dense(32, nOutputs))

    
    # when using residual loss, load fmu
    #(status, res) = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu_comp, eq_num, rand(Float64, 110))
    fmu_from_string = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/IEEE_14_Buses.fmu")
    fmu_comp = FMI.fmiInstantiate!(fmu_from_string; loggingOn=true) # this or only load?
    eq_num = 1403 # hardcoded but okay

    trainSurrogate!(model, dataloader, train_in, train_out, test_in, test_out, fmu_comp, eq_num, outname; nepochs=100)
end

runTrainNNTest(
    "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/data/eq_1403.csv",
    16,
    110,
    "eq_1403_out.bson")



# runTrainNNTest(
#   "eq_1403.csv",
#   16,
#   110,
#   "eq_1403_out.bson"))
