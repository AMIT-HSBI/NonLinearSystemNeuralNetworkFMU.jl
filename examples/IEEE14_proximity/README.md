# IEEE14_proximity

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> IEEE14_proximity

Authors: Andreas Heuermann.


## Dependencies

To (locally) reproduce this project, do the following:

  1. Make sure you have local Julia package
     [NaiveONNX](https://github.com/AMIT-HSBI/NaiveONNX.jl) in [../NaiveONNX.jl](../NaiveONNX.jl).
     If not update your git submodule with

     ```bash
     $ git submodule update --force --init --recursive
     ```

  2. Open a Julia console and run:

     ```julia
     julia> using Pkg
     julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
     julia> Pkg.activate("examples/IEEE14_proximity/")
     julia> Pkg.instantiate()
     ```

  4. OpenModelica: Tested omc version v1.21.0-dev-288-g01b6764df5-cmake
     with OMSimulator version OMSimulator v2.1.1.post194-g75de4c6-linux-debug.

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "IEEE14"
```
which auto-activate the project and enable local path handling from DrWatson.

## Run Scripts

The Modelica model OpenIPSL.Examples.IEEE14.IEEE from the
[OpenIPSL](https://doc.openipsl.org/) library has one large non-linear equation system
that is very hard to replace with a working surrogate.

### Data Generation

Run script [scripts/runAll.jl](scripts/genAllData.jl) to run the example.
Change `N` to specify how many data points should be generated.

```julia
julia> include("scripts/runAll.jl")
```
