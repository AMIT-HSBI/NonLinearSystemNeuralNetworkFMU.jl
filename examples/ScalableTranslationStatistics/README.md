# ScalableTranslationStatistics

This example for
[NonLinearSystemNeuralNetworkFMU.jl](https://github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl)
is using [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to make
a reproducible example named **ScalableTranslatonStatistics**.

Authors: Andreas Heuermann.

These scripts can be used to generate benchmarks for simulation times of the
ONNX surrogate FMUs.

## Dependencies

  1. Download
     [ScalableTranslationStatistics](https://github.com/ltx-simulation/ScalableTranslationStatistics)
     and create environment variable `SCALABLETRANSLATIONSTATISTICS_ROOT` that
     points to it root directory.
     This example uses commit 8a96bcb036fb9a6e56a3d1e369af8f36fcbc3c1e.

  2. Make sure you have local Julia package
     [NaiveONNX](https://github.com/AnHeuermann/NaiveONNX.jl) in [../NaiveONNX.jl](../NaiveONNX.jl).
     If not update your git submodule with

     ```bash
     $ git submodule update --force --init --recursive
     ```

  3. To create plots with CairoMakie an X-server has to be available.
     So the environmental variable `DISPLAY` has to be set.

     If you are using ssh to
     connect to a remote machine use `ssh -X user@remote.com`.
     Check your Julia session has the variable available or set it:
     ```julia
     julia> ENV["DISPLAY"]
     julia> ENV["DISPLAY"] = "localhost:10.0"
     ```

  4. Open a Julia console and run:

     ```julia
     julia> using Pkg
     julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
     julia> Pkg.activate("examples/SimpleLoop/")
     julia> Pkg.instantiate()
     ```

  5. OpenModelica: Tested omc version v1.21.0-dev-288-g01b6764df5-cmake
     with OMSimulator version OMSimulator v2.1.1.post194-g75de4c6-linux-debug.

This will install all necessary packages for you to be able to run the scripts
and everything should work out of the box, including correctly finding local
paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "SimpleLoop"
```
which auto-activate the project and enable local path handling from DrWatson.

## Run Scripts

You can change the size of the Modelica model by changing

```julia
sizes = [5, 10, 20, 40, 80, 160]
```

in the script [runAll.jl](./scripts/runAll.jl).

Run

```julia
include("scripts/runAll.jl")
```
for all tests.
