using Test
using NonLinearSystemNeuralNetworkFMU

path_to_omc = ""
if Sys.iswindows()
  path_to_omc = string(strip(read(`where omc.exe`, String)))
else
  path_to_omc = string(strip(read(`which omc`, String)))
end
if !isfile(path_to_omc)
  error("omc not found")
else
  @info "Using omc: $path_to_omc"
end

path_to_fmi_header = abspath(joinpath(@__DIR__, "..", "FMI-Standard-2.0.3","headers"))

include("profiling_tests.jl")
include("gen_fmus_test.jl")
