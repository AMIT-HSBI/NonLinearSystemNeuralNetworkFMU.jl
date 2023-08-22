# Documentation

## Build and host locally

Make sure at least the fetch address of your remote is https, so that Documenter.jl can
fetch information using git without entering SSH key password.

```bash
git remote set-url origin https://github.com/AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl.git
git remote set-url --push origin git@github.com:AnHeuermann/NonLinearSystemNeuralNetworkFMU.jl.git
```

Make sure you developed NonLinearSystemNeuralNetworkFMU.jl and NaiveONNX.jl.
To run Documenter.jl along with LiveServer to render the docs and track any modifications run:

```julia
using Pkg; Pkg.activate("docs/"); Pkg.resolve()
using NonLinearSystemNeuralNetworkFMU, NaiveONNX, LiveServer
servedocs()
```
