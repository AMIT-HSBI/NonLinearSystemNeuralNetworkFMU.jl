# JetPump

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> IEEE14

Authors: Andreas Heuermann.

## Dependencies

To (locally) reproduce this project, do the following:

  1. Make sure you have Modelica library `JetPumpTool` accessible by OpenModelica.

  2. OpenModelica: Tested omc version v1.22.0-dev-156-g5e403e6442-cmake.

  3. Open a Julia console and run:

     ```julia
     julia> using Pkg
     julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
     julia> Pkg.activate("examples/JetPump/")
     julia> Pkg.instantiate()
     ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "SimpleLoop"
```
which auto-activate the project and enable local path handling from DrWatson.

## Run Scripts

The Modelica model `JetPumpTool.Test.Run_JetPump_p2_T2_p3_T3_mflow_3` from the
JetPumpTool library has one non-linear equation system that we want to replace with an
explicit equation surrogate to remove iterative solvers from the simulation executable.

### Data Generation

Run script [scripts/genAllData.jl](scripts/genAllData.jl) to generate training data for
the `JetPumpInverse` example.
The default number of data points to generate is `N=100_000`, but can be changed in the
scripts.

```julia
julia> include("scripts/genAllData.jl")
```

The resulting training data can be found in
[data/sims/JetPumpInverse_<N>/data](data/sims/JetPumpInverse_100000/data).

### Train with SymbolicRegression

Run script [scripts/trainSurrogate.jl](scripts/trainSurrogate.jl) to fit equations with
genetic algorithms.

```julia
julia> include("scripts/trainSurrogate.jl")
```

The resulting equation system is not put back into the original Modelica model or
simulation executable.
