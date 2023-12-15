using NonLinearSystemNeuralNetworkFMU
using BSON
using FMI
using FMIImport
import FMICore

#IEEE14 "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/temp-extendfmu/IEEE_14_Buses.fmu"
#       "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/profilingInfo.bson"

#simpleLoop "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/simpleLoop.interface.fmu"
#           "/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/profilingInfo.bson"

# eq 14
# /home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_110/simpleLoop.interface.fmu
# /home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_110/profilingInfo.bson

#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/simpleLoop.interface.fmu
#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/simpleLoop_1000/profilingInfo.bson

#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_112/simpleLoop.interface.fmu
fmu = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/temp-extendfmu/IEEE_14_Buses.fmu")
comp = FMI.fmiInstantiate!(fmu)
FMI.fmiSetupExperiment(comp)
FMI.fmiEnterInitializationMode(comp)
FMI.fmiExitInitializationMode(comp)

#/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_112/profilingInfo.bson
profilinginfo = getProfilingInfo("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_113/profilingInfo.bson")

vr = FMI.fmiStringToValueReference(fmu, profilinginfo[1].iterationVariables)

eq_num = profilinginfo[1].eqInfo.id
x = rand(length(profilinginfo[1].iterationVariables),)
status, jac = NonLinearSystemNeuralNetworkFMU.fmiEvaluateJacobian(comp, eq_num, vr, Float64.(x))

#IEEE14 [1:5,1:5]
display(reshape(jac, (length(profilinginfo[1].iterationVariables), length(profilinginfo[1].iterationVariables)))[1:5,1:5])


# when eqNumber == 0, then 1x1 output is okay, NxN output only [1,1] is okay other values are e-100 or so
# when eqNumber == eq_num, then crash