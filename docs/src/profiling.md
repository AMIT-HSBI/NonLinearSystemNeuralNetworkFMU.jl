
# Profiling Modelica Models

The profiling functionalities of OpenModelica are used to decide if an equation is slow
enough to be replaced by a surrogate.

## Functions

### Profiling

Simulate Modelica model to find slowest equations and what variables are used and what
values these variables have during simulation.

```@docs
profiling
minMaxValuesReSim
```

### Getting Profiling

The [`main`](@ref) function will save profiling artifacts that can be loaded with the
following functions.

```@docs
getProfilingInfo
getUsingVars
getIterationVars
getInnerEquations
getMinMax
```

## Structures

```@docs
OMOptions
ProfilingInfo
EqInfo
```

## Examples

### Find Slowest Non-linear Equation Systems

We have a Modelica model `SimpleLoop`, see [test/simpleLoop.mo](https://github.com/AMIT-HSBI/NonLinearSystemNeuralNetworkFMU.jl/blob/main/test/simpleLoop.mo) with some non-linear equation system

```math
\begin{align*}
  r^2 &= x^2 + y^2 \\
  rs  &= x + y
\end{align*}
```

We want to see how much simulation time is spend solving this equation.
So let's start [`profiling`](@ref):

```@repl profilingexample
using NonLinearSystemNeuralNetworkFMU
modelName = "simpleLoop";
moFiles = [joinpath("test","simpleLoop.mo")];
options = OMOptions(workingDir = "tempDir")
profilingInfo = profiling(modelName, moFiles; options=options, threshold=0)
```

We can see that non-linear equation system `14` is using variables `s` and `r`
as input and has iteration variable `y`.
`x` will be computed in the inner equation.

```@repl profilingexample
profilingInfo[1].usingVars
profilingInfo[1].iterationVariables
```

So we can see, that equations `14` is the slowest non-linear equation system. It is called 2512 times and needs around 15% of the total simulation time, in this case that is around 592 $\mu s$.

During [`profiling`](@ref) function [`minMaxValuesReSim`](@ref) is called to re-simulate
the Modelica model and read the simulation results to find the smallest and largest
values for each given variable.

We can check them by looking into

```@repl profilingexample
profilingInfo[1].boundary.min
profilingInfo[1].boundary.min
```

It's possible to save and later load the profilingInfo in binary JSON format:

```@repl profilingexample
using BSON
BSON.@save "simpleLoop.bson" profilingInfo
getProfilingInfo("simpleLoop.bson")
```
