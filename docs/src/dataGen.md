# Training Data Generation

To generate training data for the slowest non-linear equations found during
[Profiling Modelica Models](@ref) we now simulate the equations multiple time
and save in- and outputs.

We will use the [Functional Mock-up Interface (FMI)](https://fmi-standard.org/) standard
to generate FMU that we extend with some function to evaluate single equations without
the need to simulate the rest of the model.

## Functions

```@docs
generateFMU
```

```@docs
addEqInterface2FMU
```

```@docs
generateTrainingData
```

## Examples

First we need to create a Model-Exchange 2.0 FMU with OpenModelica.

This can be done directly from OpenModelica or with [`generateFMU`](@ref):

```@example dataexample
using NonLinearSystemNeuralNetworkFMU #hide
omc = string(strip(read(`which omc`, String))) #hide

fmu = generateFMU("simpleLoop",
                  ["test/simpleLoop.mo"];
                  pathToOmc = omc,
                  workingDir = "tempDir")
rm("tempDir", recursive=true, force=true) #hide
```

Next we need to add non-standard C function

```C
fmi2Status myfmi2evaluateEq(fmi2Component c, const size_t eqNumber)
```

that will call `<modelname>_eqFunction_<eqIndex>(DATA* data, threadData_t *threadData)`
for all non-linear equations we want to generate data for.

Using [`addEqInterface2FMU`](@ref) this C code will be generated and added to the FMU.

```@example dataexample
interfaceFmu = addEqInterface2FMU("simpleLoop",
                                  fmu,
                                  [14],
                                  workingDir = "tempDir")
rm("tempDir", recursive=true, force=true) #hide
```

Now we can create evaluate equation `14` for random values and save the outputs to generate training data.

```@example dataexample
using CSV
using DataFrames
generateTrainingData(interfaceFmu,
                     "simpleLoop_data.csv",
                     14,
                     ["s", "r"],
                     [0.0, 0.95],
                     [1.5, 3.15],
                     ["y"];
                     N = 10)
df =  CSV.File("simpleLoop_data.csv")
rm("simpleLoop_data.csv", force=true) #hide
```
