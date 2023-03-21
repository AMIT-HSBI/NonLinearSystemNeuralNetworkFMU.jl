using DrWatson
@quickactivate "SimpleLoop"

# Generate FMUs
begin
  include("SimpleLoop_example.jl")
  @assert isfile(datadir("sims", "fmus", "simpleLoop.onnx_N100.fmu"))
  @assert isfile(datadir("sims", "fmus", "simpleLoop.onnx_N500.fmu"))
  @assert isfile(datadir("sims", "fmus", "simpleLoop.onnx_N750.fmu"))
  @assert isfile(datadir("sims", "fmus", "simpleLoop.onnx_N1000.fmu"))
end

# Generate gif
begin
  include("SimpleLoop_intersection.jl")
  @assert isfile(plotsdir("SimpleLoop_intersection.gif"))
end

# Plot data
begin
  include("SimpleLoop_dataGenPlots.jl")
  @assert isfile(plotsdir("SimpleLoop_data.svg"))
  @assert isfile(plotsdir("SimpleLoop_data_filtered.svg"))
end

# Plot simulation results
begin
  include("SimpleLoop_simresults.jl")
  @assert isfile(plotsdir("SimpleLoop_simresults_y.svg"))
end
