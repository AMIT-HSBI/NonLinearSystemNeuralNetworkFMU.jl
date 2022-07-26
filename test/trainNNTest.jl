#
# Copyright (c) 2022 Andreas Heuermann, Philip Hannebohm
# Licensed under the MIT license. See LICENSE.md file in the project root for details.
#

import BSON
import CSV
import DataFrames
import Flux
import StatsBase

"""
    readData(filename, nInputs; ratio=0.8)

# Arguments
  - `filename::String`: Path of CSV file with training data.
  - `nInputs::Integer`: Number of input varaibles for model.

# Keywords
  - `ratio=0.8`: Ratio between training and testing data points.
                 Defaults to 80% training and 20% testing.
"""
function readData(filename::String, nInputs::Integer; ratio=0.8)
  m = Matrix{Float32}(CSV.read(filename, DataFrames.DataFrame))
  n = length(m[:,1])
  num_train = Integer(round(n*ratio))
  trainIters = StatsBase.sample(1:n, num_train, replace = false)
  testIters = setdiff(1:n, trainIters)

  train_in  = [m[i, 1:nInputs]     for i in trainIters]
  train_out = [m[i, nInputs+1:end] for i in trainIters]
  test_in   = [m[i, 1:nInputs]     for i in testIters]
  test_out  = [m[i, nInputs+1:end] for i in testIters]
  return train_in, train_out, test_in, test_out
end

"""
    trainSurrogate!(model, dataloader, test_in, test_out, savename; losstol=1e-6, nepochs=100, eta=1e-3)

Train surrogate on data from `dataloader`.
Stop when `losstol` is reached. Saves model with minimal loss.

# Arguments
  - `model`: Flux model to train.
  - `dataloader::Flux.DataLoader`: DataLoader with trainig data.
  - `train_in`: Single btatch of train input data.
  - `train_out`: Single btatch of train output data.
  - `savename::String`: Path to location where trained model is saved to.

# Keywords
  - `losstol::Real=1e-6`: Loss to reach for model.
  - `nepochs=100`: Number of epochs to train.
  - `eta=1e-3`: η parameter for Flux.ADAM.
"""
function trainSurrogate!(model, dataloader::Flux.DataLoader, train_in, train_out, savename::String; losstol::Real=1e-6, nepochs=100, eta=1e-3)
  # Create directory for bson file
  mkpath(dirname(savename))

  # Compute initial loss
  ps = Flux.params(model)
  opt = Flux.Adam(eta)
  loss(x,y) = Flux.mse(model(x),y)
  disp_loss() = sum(loss.(train_in,train_out))/length(train_in)
  callback = Flux.throttle(disp_loss, 5)
  minL = disp_loss()
  @info "Initial loss: $(minL)"
  if minL < losstol
    BSON.@save savename model
    return
  end

  @info "Start training"
  epoch = 0
  try
    while epoch < nepochs
      epoch += 1
      for (ti,to) in dataloader
        Flux.train!(loss, ps, zip(ti,to), opt, cb=callback)
      end
      l = disp_loss()
      @info "Epoch $(epoch): Train loss=$(l)"
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

  @info "Best loss = $(minL)\n\tAfter $(epoch) epochs"
end

function runTrainNNTest()
  fileName = abspath(joinpath(@__DIR__, "data", "simpleLoop_eq14.csv"))
  nInputs = 2
  nOutputs = 1
  outname = abspath(joinpath(@__DIR__, "nn", "simpleLoop_eq14.bson"))

  (train_in, train_out, test_in, test_out) = readData(fileName, nInputs)
  dataloader = Flux.DataLoader((train_in, train_out), batchsize=64, shuffle=true)

  model = Flux.Chain(Flux.Dense(nInputs, 5, tanh),
                     Flux.Dense(5, 5, tanh),
                     Flux.Dense(5, nOutputs))

  trainSurrogate!(model, dataloader, test_in, test_out, outname; nepochs=100)
end

# dict = BSON.load("test/nn/simpleLoop_eq14.bson")
# model = dict[first(keys(dict))]
