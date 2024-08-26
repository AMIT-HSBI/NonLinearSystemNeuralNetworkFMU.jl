# SimpleLoop

This example for
[NonLinearSystemNeuralNetworkFMU.jl](https://github.com/AMIT-HSBI/NonLinearSystemNeuralNetworkFMU.jl)
is using [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to make a
reproducible example named **SimpleLoop**.

Authors: Andreas Heuermann, Philip Hannebohm.

These scripts were used to generate the plots and animations for the presentation
[Accelerating the simulation of equation based models by replacing non linear algebraic
loops with error controlled machine learning surrogates](https://modprodblog.wordpress.com/modprod-2023/)
at
[MODPROD 2023](https://modprodblog.wordpress.com/).

## Dependencies

> [!WARNING]
> Only tested on Ubuntu. It is very unlikely that these scripts will work on
> Windows and impossible on MacOS.

> [!NOTE]
> The dependencies are a nightmare. While this could work with later versions of
> dependent packages most likely it's easier to install the exact versions
> specified in [Manifest.toml](./Manifest.toml).

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
     julia> Pkg.activate("examples/SimpleLoop/")
     julia> Pkg.instantiate()
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

The Modelica model SimpleLoop is defined in [src/simpleLoop.mo](src/simpleLoop.mo).
It describes a moving line and growing circle and has a non-linear system solving for the
intersection points of the two.

```math
r^2 = x^2 + y^2
rs + b = x + y
```

for unknown coordinates `x` and `y`.

You can run
```julia
include("scripts/runall.jl")
```

to run all scripts or read along to see what the scripts are doing.

### Data and FMU Surrogate Generation

Run script [scripts/SimpleLoop_example.jl](scripts/SimpleLoop_example.jl) to generate four
FMU surrogates based on `N ∈ {100, 500, 750, 1000}` data points.

```julia
julia> include("scripts/SimpleLoop_example.jl")
```

The resulting FMUs can be found in
[data/sims/fmus/simpleLoop.onnx_NXXX.fmu](data/sims/fmus/simpleLoop.onnx_N100.fmu).

After the FMUs are generated the plots and animations can be created.

### Animate Intersection Points

Run script [scripts/SimpleLoop_intersection.jl](scripts/SimpleLoop_intersection.jl]) to
generate a animation of the circle and line.

```julia
julia> include("scripts/SimpleLoop_intersection.jl")
```

The resulting animation can be found in
[plots/SimpleLoop_intersection.gif](plots/SimpleLoop_intersection.gif).

### Plot Data Set

Run script [scripts/SimpleLoop_dataGenPlots.jl](scripts/SimpleLoop_dataGenPlots.jl) to
generate plots showing how the solution of SimpleLoop model isn't unique and how to filter
it.

```julia
julia> include("scripts/SimpleLoop_dataGenPlots.jl")
```

The resulting plots can be found in [plots/SimpleLoop_data.svg](plots/SimpleLoop_data.svg)
and [plots/SimpleLoop_data_filtered.svg](plots/SimpleLoop_data_filtered.svg).

### Plot Comparison between Simulation Results

To compare the results of the four surrogates with the expected results of `y` run
the script [scripts/SimpleLoop_simresults.jl](scripts/SimpleLoop_simresults.jl).

```julia
julia> include("scripts/SimpleLoop_simresults.jl")
```

The generated plot can be found in
[plots/SimpleLoop_simresults_y.svg](plots/SimpleLoop_simresults_y.svg).

