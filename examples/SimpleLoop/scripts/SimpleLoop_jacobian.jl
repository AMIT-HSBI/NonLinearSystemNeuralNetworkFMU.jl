
using FMI
using FMIImport
import FMICore
using Libdl


function fmiEvaluateJacobian(comp::FMICore.FMU2Component, eq::Integer, x::Array{Float64})::Tuple{FMI.fmi2Status, Array{Float64}}
    # wahrscheinlich braucht es eine c-Funktion, die nicht nur einen pointer auf die Jacobi-Matrix returned, sondern gleich die Auswertung an der Stelle x
    # diese Funktion muss auch ein Argument res nehmen welches dann die Evaluation enthält.?
  
    @assert eq>=0 "Residual index has to be non-negative!"
  
    # this is a pointer to Jacobian matrix in row-major-format or NULL in error case.
    fmiEvaluateJacobian = Libdl.dlsym(comp.fmu.libHandle, :myfmi2EvaluateJacobian)
  
    jac = Array{Float64}(undef, length(x)*length(x))
  
    eqCtype = Csize_t(eq)
  
    status = ccall(fmiEvaluateJacobian,
                   Cuint,
                   (Ptr{Nothing}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
                   comp.compAddr, eqCtype, x, jac)
  
    return status, jac
  end
  
  
  
  
  #-------------------------------
  # when using residual loss, load fmu
  #(status, res) = NonLinearSystemNeuralNetworkFMU.fmiEvaluateRes(fmu_comp, eq_num, rand(Float64, 110))
  #"/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/IEEE14/data/sims/IEEE_14_Buses_1000/IEEE_14_Buses.interface.fmu"
  fmu = FMI.fmiLoad("/home/fbrandt3/arbeit/NonLinearSystemNeuralNetworkFMU.jl/examples/SimpleLoop/data/sims/simpleLoop_100/unzipped_fmu_sl/simpleLoop.fmu")
  comp = FMI.fmiInstantiate!(fmu) # this or only load?
  FMI.fmiSetupExperiment(comp)
  FMI.fmiEnterInitializationMode(comp)
  FMI.fmiExitInitializationMode(comp)
  
  status, jac = fmiEvaluateJacobian(comp, 0, [1.,2.])
  
  jac