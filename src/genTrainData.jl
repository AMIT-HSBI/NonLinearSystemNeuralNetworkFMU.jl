include("genFMUs.jl")

function generateTrainingData(model_name::String, path_to_mo::String, path_to_omc::String, working_dir::String)

  omc_working_dir = abspath(joinpath(working_dir, model_name))
  path_to_fmu = generateFMU(model_name = model_name, path_to_mo=path_to_mo, path_to_omc=path_to_omc, tmpDir = omc_working_dir)

  #path_to_fmi_header = joinpath(@__DIR__, "..", "include", "fmi2")
  #addEqInterface2FMU(model_name = model_name,
  #                   path_to_fmu = path_to_fmu,
  #                   path_to_fmi_header = path_to_fmi_header,
  #                   eqIndices=eqIndices,
  #                   tempDir = "temp")
end

