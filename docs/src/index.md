# NonLinearSystemNeuralNetworkFMU.jl

Generate Neural Networks to replace non-linear systems inside OpenModelica 2.0 FMUs.

## Table of Contents
```@contents
  pages = [
    "Home" => "index.md",
    "Main" => "main.md",
    "Profiling" => "profiling.md",
    "Data Generation" => "dataGen.md",
    "ONNX Generation" => "train.md",
    "Integrate ONNX" => "integrateONNX.md"
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
      * Add C interface to evaluate single non-linear equation system without evaluating
        anything else.
      * Re-compile FMU.
      * Initialize FMU using [FMI.jl](https://github.com/ThummeTo/FMI.jl).
      * Generate training data for each equation system by calling new interface.

  3. Create ONNX (performed by user).

      * Use your favorite environment to create a trained Open Neural Network Exchange
        ([ONNX](https://onnx.ai/)) model.
          * Use the generated training data to train artificial neural network.

  4. Integrate ONNX into FMU.

      * Replace equations with ONNX evaluation done by [ONNX Runtime](https://onnxruntime.ai/)
        in generated C code.
      * Re-compile FMU.
          * Environment variable ORT_DIR has to be set and point to the ONNX runtime
            directory (with include/ and lib/ inside).

## Installation

See [AMIT-HSBI/NonLinearSystemNeuralNetworkFMU.jl README.md](https://github.com/AMIT-HSBI/NonLinearSystemNeuralNetworkFMU.jl#nonlinearsystemneuralnetworkfmujl) for installation instructions.
