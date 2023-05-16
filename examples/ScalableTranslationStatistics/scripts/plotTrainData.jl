using DrWatson
@quickactivate "ScalableTranslationStatistics"

using CairoMakie
using ColorSchemes
using NonLinearSystemNeuralNetworkFMU
using CSV
using DataFrames

# Uncomment to change to Times New Roman like font
#update_theme!(fonts = (; regular = "Nimbus Roman No9 L", bold = "Nimbus Roman No9 L"))

include(srcdir("plotResult.jl"))

function plotAllTrainingData(sizes)
  for size in sizes
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
    shortName = split(modelName, ".")[end]

    profilingInfos = NonLinearSystemNeuralNetworkFMU.getProfilingInfo(datadir("sims", shortName, "profilingInfo.bson"))

    prof = profilingInfos[1]

    simDir = datadir("sims", shortName)

    csvFileData = joinpath(simDir, "data", "eq_$(prof.eqInfo.id).csv")
    df_data = CSV.read(csvFileData, DataFrames.DataFrame; ntasks=1)

    refCsvFile = joinpath(simDir, "temp-profiling", modelName*"_res.csv")
    df_ref = CSV.read(refCsvFile, DataFrames.DataFrame; ntasks=1)

    #resultCsvFile = datadir("exp_raw", shortName, "$(shortName)_res.onnx.csv")
    #df_surr = CSV.read(resultCsvFile, DataFrames.DataFrame; ntasks=1)

    fig = plotTrainArea(prof.iterationVariables, df_ref, df_trainData = df_data)
    mkpath(dirname(plotsdir(modelName, "$(shortName)_trainData.svg")))
    save(plotsdir(modelName, "$(shortName)_trainData.svg"), fig)

    fig = plotTrainDataHistogram(prof.usingVars, df_data, title = "Training Data Distribution - Equation $(prof.eqInfo.id)")
    save(plotsdir(modelName, "$(shortName)_trainData_hist.svg"), fig)
  end
end

function simulationTimes(sizes;
                         printAbsTime = true,
                         plotTimeLabels=true,
                         filename = plotsdir("simTimeOverview.pdf"),
                         title = "Simulation time of Examples.ScaledNLEquations.NLEquations_N")
  modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(sizes[01])"
  shortName = split(modelName, ".")[end]
  profilingInfos_01 = NonLinearSystemNeuralNetworkFMU.getProfilingInfo(datadir("sims", shortName, "profilingInfo.bson"))

  numEqs = length(profilingInfos_01)+1
  lenSizes = length(sizes)
  X = Array{Int}(undef, lenSizes*numEqs)
  Y_fraction = Array{Float64}(undef, lenSizes*numEqs)
  Y_total = zeros(Float64, lenSizes)
  Y_time = Array{Float64}(undef, lenSizes*(numEqs))
  eqMapping = Dict()
  for (i, prof) in enumerate(profilingInfos_01)
    push!(eqMapping, prof.eqInfo.id => i)
  end

  for (i,size) in enumerate(sizes)
    modelName = "ScalableTranslationStatistics.Examples.ScaledNLEquations.NLEquations_$(size)"
    shortName = split(modelName, ".")[end]
    profilingInfos = NonLinearSystemNeuralNetworkFMU.getProfilingInfo(datadir("sims", shortName, "profilingInfo.bson"))

    fraction_total = 0
    for (j, prof) in enumerate(profilingInfos)
      X[(i-1)*numEqs+j] = i
      Y_fraction[(i-1)*numEqs+j] = prof.eqInfo.fraction*100
      fraction_total += prof.eqInfo.fraction
      Y_time[(i-1)*numEqs+j] = prof.eqInfo.time
      Y_total[i] += prof.eqInfo.time
    end

    # Remaining equations
    Y_fraction[i*numEqs] = (1-fraction_total)*100
    X[i*numEqs] = i
    Y_time[i*numEqs] = Y_total[i] / fraction_total * (1-fraction_total)
  end

  # Plot relative simulation time
  stack = [j for _ in 1:lenSizes for j in 1:numEqs]
  fig = Figure(fontsize = 18)
  axis_frac = Axis(fig[1, 1],
                   xticks = (1:lenSizes, ["N=$size" for size in sizes]),
                   xgridvisible = false,
                   yticks = 0:10:100,
                   yminorticks = IntervalsBetween(2),
                   yminorticksvisible = true,
                   yminorgridvisible = true,
                   ylabel = "Simulation time [%]")
  barplot!(axis_frac,
           X,
           Y_fraction,
           stack = stack,
           color = stack)

  # Add bar labels
  totalTimes = [sum(Y_time[(i-1)*numEqs+1:i*numEqs]) for i=1:lenSizes]
  if plotTimeLabels
    text!(axis_frac,
          ["$(round(i,sigdigits=2)) [s]" for i in totalTimes],
          position = Point2f.(
              1:lenSizes,
              ones().*95
          ),
          color = :black,
          align = (:center, :center),
          tellwidth=false)
  end

  # Plot absolute simulation time
  if printAbsTime
    axis_time = Axis(fig[1, 2],
                     xticks = (1:lenSizes, ["N=$size" for size in sizes]),
                     xgridvisible = false,
                     yticks = 0:10:maximum(totalTimes),
                     yminorticks = IntervalsBetween(2),
                     yminorticksvisible = true,
                     yminorgridvisible = true,
                     ylabel = "Simulation time [s]")

    barplot!(axis_time,
             X,
             Y_time,
             stack = stack,
             color = stack)
  end

  # Title
  if title != ""
    Label(fig[0,:], text=title, fontsize=24, tellwidth=false, tellheight=true)
  end

  # Legend
  labels = vcat(["eq $(i)" for i in 1:numEqs-1], ["rest"])
  colors = ColorSchemes.viridis[range(0,1,length(labels))]
  elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
  Legend(fig[2,:], elements, labels,
         tellheight = true, tellwidth = false, margin = (10, 10, 10, 10),
         halign = :left, valign = :top, orientation = :horizontal)

  save(filename, fig)

  return filename
end
