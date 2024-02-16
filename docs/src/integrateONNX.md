## Include ONNX into exported FMU

After an ONNX is generated it can be compiled into the FMU.


!!! warning
    The FMU can't be compiled on Windows systems, because the ONNX Runtime is
    incompatible with the MSYS2 shell used by OpenModelica to compile the FMU.

## Functions

```@docs
buildWithOnnx
```

## Example

```@example
using NonLinearSystemNeuralNetworkFMU # hide
rm("onnxTempDir", recursive=true, force=true) # hide
modelName = "simpleLoop"
fmu = joinpath("tempDir", "simpleLoop.interface.fmu")
profilingInfo = getProfilingInfo("simpleLoop.bson")[1:1]
onnxFiles = ["eq_14.onnx"]

buildWithOnnx(fmu,
              modelName,
              profilingInfo,
              onnxFiles,
              tempDir = "onnxTempDir")
```

This FMU can now be simulated with most FMI importing tools.
