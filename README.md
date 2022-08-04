# NonLinearSystemNeuralNetworkFMU.jl

Generate Neural Networks to replace non-linear systems inside [OpenModelica](https://openmodelica.org/) 2.0 FMUs.

## Usage

The package generates an FMU from a modelica file in XXX steps:

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

## Known Limitations

  - MAT.jl doesn't support the v4 mat files OpenModelica generates, so one
    needs to use CSV result files.

## LICENSE

NonLinearSystemNeuralNetworkFMU.jl is licensed under MIT License (see [LICENSE.md](./LICENSE.md)).

NonLinearSystemNeuralNetworkFMU.jl uses header files from the FMI standard, licensed under BSD 2-Clause (see [FMI-Standard-2.0.3/LICENSE.txt](./FMI-Standard-2.0.3/LICENSE.txt)).

## Acknowledgments

This package was developed as part of the [Proper Hybrid Models for Smarter Vehicles (PHyMoS)](https://phymos.de/en/) project,
supported by the German [Federal Ministry for Economic Affairs and Climate Action](https://www.bmwk.de/Navigation/EN/Home/home.html)
with project number 19|200022G.
