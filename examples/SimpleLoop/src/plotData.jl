using CairoMakie
using DataFrames

"""
Plot expected and trained data from DataFrame
"""
function plotData(df::DataFrame; title = "SimpleLoop: Training Data")
  fig = Figure(fontsize = 21,
               resolution = (800, 800))

  axis = Axis(fig[1,1],
              title = title,
              xlabel = "x",
              ylabel = "y")

  limits!(axis, -2.5, 3.5, -2.5, 3.5)

  # Expected solutions
  X1 = 0.5 .* ( -sqrt.(-b^2 .- 2*b .* df.r .* df.s - (df.r).^2 .* ((df.s).^2 .- 2)) .+ b .+ df.r .* df.s)
  Y1 = df.r .* df.s .+ b .- X1
  s1 = CairoMakie.scatter!(axis, X1, Y1, color=(FH_Colors[1], 0.5), markersize = 21)

  X1 = 0.5 .* ( sqrt.(-b^2 .- 2*b .* df.r .* df.s - (df.r).^2 .* ((df.s).^2 .- 2)) .+ b .+ df.r .* df.s)
  Y1 = df.r .* df.s .+ b .- X1
  s2 = CairoMakie.scatter!(axis, X1, Y1, color=(FH_Colors[2], 0.5), markersize = 21)

  # Generated data
  X_train = df.r.*df.s .+ b .- df.y;
  s3 = CairoMakie.scatter!(axis, X_train, df.y, color=FH_Colors[3], markersize=12, marker=:xcross)

  marker = [MarkerElement(color=(FH_Colors[1], 0.5), marker = :circle, markersize = 21, points = Point2f[(0, 0.5)]),
            MarkerElement(color=(FH_Colors[2], 0.5), marker = :circle, markersize = 21, points = Point2f[(0.75, 0.5)])]

  Legend(fig[1,1],
         [marker, s3],
         ["Expected", "Generated"],
         tellheight = false,
         tellwidth = false,
         margin = (50, 50, 50, 50),
         halign = :left, valign = :bottom,
         labelsize = 21)

  return fig
end
