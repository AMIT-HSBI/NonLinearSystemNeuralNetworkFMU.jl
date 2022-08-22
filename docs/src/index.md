# NonLinearSystemNeuralNetworkFMU.jl

Generate Neural Networks to replace non-linear systems inside OpenModelica 2.0 FMUs.

## Table of Contents
```@contents
  pages = [
    "Home" => "index.md",
    "Profiling" => "profiling.md",
    "Data Generation" => "dataGen.md"
  ]
```

## Overview

The package generates an FMU from a modelica file in 3 steps (+ 1 user step):

  1. Find non-linear equation systems to replace.

      * Simulate and profile Modelica model with OpenModelica using
        [OMJulia.jl](https://github.com/OpenModelica/OMJulia.jl).
      * Find slowest equations below given threshold.
      * Find depending variables specifying input and output for every
        non-linear equation system.
      * Find min-max ranges for input variables by analyzing the simulation results.

  2. Generate training data.

      * Generate 2.0 Model Exchange FMU with OpenModelica.
      * Add C interface to evaluate single non-linear equation system without evaluating anything else.
      * Re-compile FMU.
      * Initialize FMU using [FMI.jl](https://github.com/ThummeTo/FMI.jl).
      * Generate training data for each equation system by calling new interface.

  3. Train neural network.

      * Step performed by user.

  4. Integrate neural network into FMU

      * Replace equations with neural network in generated C code.
      * Re-compile FMU.

## Installation

Clone this repository to your machine and use the package manager Pkg to develop this package.

```julia-repl
(@v1.7) pkg> dev /path/to/NonLinearSystemNeuralNetworkFMU
julia> using NonLinearSystemNeuralNetworkFMU
```
