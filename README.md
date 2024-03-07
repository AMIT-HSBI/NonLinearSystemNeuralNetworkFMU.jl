# NonLinearSystemNeuralNetworkFMU.jl

*Generate Neural Networks to replace non-linear systems inside [OpenModelica](https://openmodelica.org/) 2.0 FMUs.*

[![][docs-dev-img]][docs-dev-url] [![][GHA-img-linux]][GHA-url-linux] [![][GHA-img-win]][GHA-url-win]

## Working with this repository

This repository uses (private) submodules for the examples.

Clone with `--recursive`:

```bash
git clone git@github.com:AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl.git --recursive
```

To initialize or update your local git repository to use the latest submodules run:

```bash
git submodule update --init
```

## Requirements

  - Julia v1.9 or newer.
  - OpenModelica version v1.23.0-dev-83 or newer.
    - Path has to contain the OpenModelica bin directory `/path/to/OpenModelica/bin/`.
    - For running the tests: Environment variable `OPENMODELICAHOME` set to point to the installation directory of OpenModelica.
  - CMake version 3.21 or newer.
  - ONNX Runtime 1.12 or newer.
    - Environment variable `ORT_DIR` set to point to the installation directory.

## Usage

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

  3. Create ONNX (performed by user).

      * Use your favorite environment to create a trained Open Neural Network Exchange ([ONNX](https://onnx.ai/)) model.
        * Use the generated training data to train artificial neural network.

  4. Integrate ONNX into FMU.

      * Replace equations with ONNX evaluation done by [ONNX Runtime](https://onnxruntime.ai/) in generated C code.
      * Re-compile FMU.
        * Environment variable `ORT_DIR` has to be set and point to the ONNX runtime directory (with include/ and lib/ inside).

## Examples

### SimpleLoop

In [examples/SimpleLoop/](examples/SimpleLoop/) and
[examples/SimpleLoop_proximity](examples/SimpleLoop_proximity/) is a simple example
of a non-linear system with two unknowns replaced by a ONNX surrogate. It's less
explanatory but was used to generate plots for a presentation.

All dependencies are managed by [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/),
checkout the [README.md](examples/SimpleLoop/README.md) respectively
[README.md](examples/SimpleLoop_proximity/README.md) (with proximity) for more details.

### IEEE14

There is another example with some larger algebraic systems in
[examples/IEEE14/](./examples/IEEE14/) and
[examples/IEEE14_proximity/](./examples/IEEE14_proximity/) using the
[OpenIPSL](https://github.com/OpenIPSL/OpenIPSL) Modelica library.

All dependencies are managed by [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/),
checkout the [README.md](examples/IEEE14/README.md) respectively
[README.md](examples/IEEE14_proximity/README.md) (with proximity) for more details.

### Scalable Translation Statistics

You'll need access to the PHyMoS GitLab / the ScalableTranslationStatistics Modelica
library to run this example. Check
[examples/ScalableTranslationStatistics/README.md](examples/ScalableTranslationStatistics/README.md)
for more information.

## Debugging

  - It's not possible to debug Julia when OMJulia is used, see
    [OpenModelica/OMJulia.jl#66](https://github.com/OpenModelica/OMJulia.jl/issues/66).
  - Enable debug prints with `ENV["JULIA_DEBUG"] = "all"`.

## Documentation

- [**Main**][docs-dev-url] &mdash; *documentation of the in-development version.*

## Known Limitations

  - MAT.jl doesn't support the v4 mat files OpenModelica generates, so one
    needs to use CSV result files.
  - The Windows build can't link to the ONNX Runtime, because it is not compatible with MSYS2 MINGW environment. See [OpenModelica/OpenModelica #9514](https://github.com/OpenModelica/OpenModelica/issues/9514).

## LICENSE

Copyright (c) 2022-2024 Andreas Heuermann, Philip Hannebohm

NonLinearSystemNeuralNetworkFMU.jl is licensed under the GNU Affero General Public License
version 3 (GNU AGPL v3), see [LICENSE.md](./LICENSE.md).

NonLinearSystemNeuralNetworkFMU.jl uses, modifies and re-distributes source code generated
by [OpenModelica](https://openmodelica.org/) which is provided under the terms of GNU AGPL
v3 license or the [OSMC Public License (OSMC-PL) version 1.8](https://openmodelica.org/osmc-pl/osmc-pl-1.8.txt).

## Acknowledgments

This package was developed as part of the [Proper Hybrid Models for Smarter Vehicles (PHyMoS)](https://phymos.de/en/) project,
supported by the German [Federal Ministry for Economic Affairs and Climate Action](https://www.bmwk.de/Navigation/EN/Home/home.html)
with project number 19|200022G.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://anheuermann.github.io/NonLinearSystemNeuralNetworkFMU.jl/dev/

[GHA-img-linux]: https://github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl/actions/workflows/base-tests-linux.yml/badge.svg?branch=main
[GHA-url-linux]: https://github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl/actions/workflows/base-tests-linux.yml
[GHA-img-win]: https://github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl/actions/workflows/base-tests-windows.yml/badge.svg?branch=main
[GHA-url-win]: https://github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl/actions/workflows/base-tests-windows.yml
