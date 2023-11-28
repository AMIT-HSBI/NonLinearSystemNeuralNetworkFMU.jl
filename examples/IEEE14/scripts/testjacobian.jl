using NonLinearSystemNeuralNetworkFMU
using BSON
using FMI
using FMIImport
import FMICore

#IEEE14 "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/temp-extendfmu/IEEE_14_Buses.fmu"
#       "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/profilingInfo.bson"

#simpleLoop "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/simpleLoop.interface.fmu"
#           "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/profilingInfo.bson"

#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/simpleLoop.interface.fmu
#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson

fmu = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/simpleLoop.interface.fmu")
comp = FMI.fmiInstantiate!(fmu)
FMI.fmiSetupExperiment(comp)
FMI.fmiEnterInitializationMode(comp)
FMI.fmiExitInitializationMode(comp)

profilinginfo = getProfilingInfo("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/profilingInfo.bson")

vr = FMI.fmiStringToValueReference(fmu, profilinginfo[1].iterationVariables)

eq_num = 16
x = rand(length(profilinginfo[1].iterationVariables),)
status, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, eq_num, vr, Float64.(x))

#IEEE14 [1:5,1:5]
display(reshape(jac, (length(profilinginfo[1].iterationVariables),length(profilinginfo[1].iterationVariables))))