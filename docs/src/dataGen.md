# Training Data Generation

To generate training data for the slowest non-linear equations found during
[Profiling Modelica Models](@ref) we now simulate the equations multiple time
and save in- and outputs.

We will use the [Functional Mock-up Interface (FMI)](https://fmi-standard.org/) standard
to generate FMU that we extend with some function to evaluate single equations without
the need to simulate the rest of the model.

## Functions

```@docs
generateTrainingData
addEqInterface2FMU
generateFMU
```

## Structures
```@docs
DataGenOptions
```

```@docs
RandomMethod
RandomWalkMethod
```

## [Examples](@id data_gen_example_id)

First we need to create a Model-Exchange 2.0 FMU with OpenModelica.

This can be done directly from OpenModelica or with [`generateFMU`](@ref):

```@example dataexample
using NonLinearSystemNeuralNetworkFMU # hide
moFiles = ["test/simpleLoop.mo"]
options = OMOptions(workingDir = "tempDir")

fmu = generateFMU("simpleLoop",
                  moFiles;
                  options = options)
```

Next we need to add non-standard FMI function

```C
fmi2Status myfmi2EvaluateEq(fmi2Component c, const size_t eqNumber)
```

that will call `<modelname>_eqFunction_<eqIndex>(DATA* data, threadData_t *threadData)`
for all non-linear equations we want to generate data for.

Using [`addEqInterface2FMU`](@ref) this C code will be generated and added to the FMU.

```@example dataexample
interfaceFmu = addEqInterface2FMU("simpleLoop",
                                  fmu,
                                  [14],
                                  workingDir = "tempDir")
```

Now we can create evaluate equation `14` for random values and save the outputs to
generate training data.

```@example dataexample
using CSV
using DataFrames
options=DataGenOptions(n=10, nThreads=1)
generateTrainingData(interfaceFmu,
                     "tempDir",
                     "simpleLoop_data.csv",
                     14,
                     ["s", "r"],
                     [0.0, 0.95],
                     [1.5, 3.15],
                     ["y"];
                     options = options)
df =  DataFrame(CSV.File("simpleLoop_data.csv"))
```
