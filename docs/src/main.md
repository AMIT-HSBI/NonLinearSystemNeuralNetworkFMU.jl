# Main Data Generation Routine

To perform all needed steps for data generation the following functions have to be executed:

  1. [`profiling`](@ref)
  2. [`generateFMU`](@ref)
  3. [`addEqInterface2FMU`](@ref)
  4. [`generateTrainingData`](@ref)

These functionalities are bundled in [`main`](@ref).

## Functions

```@docs
main
```

## Example

```@repl
using NonLinearSystemNeuralNetworkFMU
modelName = "simpleLoop";
moFiles = [joinpath("test","simpleLoop.mo")];
omOptions = OMOptions(workingDir="tempDir")
dataGenOptions = DataGenOptions(method=NonLinearSystemNeuralNetworkFMU.RandomMethod(),
                                n=10,
                                nBatches=2)

(csvFiles, fmu, profilingInfo) = main(modelName,
                                      moFiles;
                                      omOptions=omOptions,
                                      dataGenOptions=dataGenOptions,
                                      reuseArtifacts=false)
```
