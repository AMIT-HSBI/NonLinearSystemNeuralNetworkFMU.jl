# Train Machine Learning Surrogate

With the [`generated training data`](@ref data_gen_example_id) it is possible to train a
machine learning (ML) method of your choice, as log as it can be exported as an ONNX.

This step has to be performed by the user.

## Example

For a naive feed-forward neural network exported to ONNX see
[AnHeuermann/NaiveONNX.jl](https://github.com/AnHeuermann/NaiveONNX.jl).

```@example trainexample
using NonLinearSystemNeuralNetworkFMU # hide
import NaiveONNX
trainingData = "simpleLoop_data.csv"
profilingInfo = getProfilingInfo("simpleLoop.bson")[1]
onnxModel = "eq_$(profilingInfo.eqInfo.id).onnx" # Name of ONNX to generate

model = NaiveONNX.trainONNX(trainingData,
                            onnxModel,
                            profilingInfo.usingVars,
                            profilingInfo.iterationVariables;
                            nepochs=10,
                            losstol=1e-8)
```
