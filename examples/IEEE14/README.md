# IEEE14

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> IEEE14

Authors: Andreas Heuermann.


## Dependencies

To (locally) reproduce this project, do the following:

  1. Make sure you have local Julia package
     [NaiveONNX](https://github.com/AnHeuermann/NaiveONNX.jl) in [../NaiveONNX.jl](../NaiveONNX.jl).
     If not update your git submodule with

     ```bash
     $ git submodule update --force --init --recursive
     ```

  2. Open a Julia console and run:

     ```julia
     julia> using Pkg
     julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
     julia> Pkg.activate("examples/IEEE14/")
     julia> Pkg.instantiate()
     ```

   3. Python dependencies: You'll need Python 3 and the following packages installed:
      - pandas
      - numpy
      - tensorflow
      - tf2onnx

  4. OpenModelica: Tested omc version v1.21.0-dev-288-g01b6764df5-cmake
     with OMSimulator version OMSimulator v2.1.1.post194-g75de4c6-linux-debug.

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "SimpleLoop"
```
which auto-activate the project and enable local path handling from DrWatson.

## Run Scripts

The Modelica model Examples.IEEE14.IEEE_14_Buses from the
[OpenIPSL](https://doc.openipsl.org/) library has one large non-linear equation system
that is very hard to replace with a working surrogate.

### Data Generation

Run script [scripts/genAllData.jl](scripts/genAllData.jl) to generate training data for
the `IEEE_14_Buses` example.
The default number of data points to generate is `N=1000`, but can be changed in the
scripts.

```julia
julia> include("scripts/genAllData.jl")
```

The resulting training data can be found in
[data/sims/IEEE_14_Buses_<N>/data](data/sims/IEEE_14_Buses_1000/data).

### Train with Flux

Run script [scripts/trainFlux.jl](scripts/trainFlux.jl]) to train an ANN with Flux.jl.

```julia
julia> include("scripts/trainFlux.jl")
```
### Train with Tensorflow

Run script [scripts/trainTensorflow.jl](scripts/trainTensorflow.jl]) to train an ANN with
Tensorflow by calling a Python script.

```julia
julia> include("scripts/trainTensorflow.jl")
```
