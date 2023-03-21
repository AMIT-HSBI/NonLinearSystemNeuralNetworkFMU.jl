using DrWatson
@quickactivate "SimpleLoop"

import Plots

t_start = 0
t_end = 2
intervals = 500
lim = 3.25

FH_Colors = ["#009BBB",
             "#E98300",
             "#C50084",
             "#722EA5",
             "#A2AD00"]

function plotFrame(time)
  b = -0.5
  r = 1 + time
  s = sqrt((2-time)*0.9)

  p = Plots.plot(xlim = (-lim,lim), ylim=(-lim,lim), aspect_ratio=:equal,
                 xlabel = "x", ylabel = "y",
                 legend=:topright,
                 title="SimpleLoop: r=$(rpad(round(r,digits=2), 4)), s=$(rpad(round(s,digits=2), 4))")

  # Circle x^2 + y^2 = r^2
  X = cos.(0:0.01:2*pi).*r
  Y = sin.(0:0.01:2*pi).*r
  p = Plots.plot(p, X, Y, label="x² + y² = r²", linewidth=2, linecolor=FH_Colors[1])

  # Line x+y = rs + b
  p1 = (-10, r*s + b + 10)
  p2 = (10, r*s + b - 10)
  p = Plots.plot(p, vcat(p1, p2), label = "x + y = rs - $(-b)", linewidth=2, linecolor=FH_Colors[4])

  # Solutions
  x1 = 0.5 * ( -sqrt(-b^2 - 2*b*r*s - r^2*(s^2-2))+b+r*s)
  y1 = r*s + b - x1
  s1 = (x1,y1)
  x2 = 0.5 * ( sqrt(-b^2 - 2*b*r*s - r^2*(s^2-2))+b+r*s)
  y2 =  r*s + b - x2
  s2 = (x2,y2)
  p = Plots.scatter(p, vcat(s1,s2), label = "solution", markercolor = FH_Colors[2])
end

# Animate Circle and Line intersecting
anim = Plots.@animate for time in range(t_start, t_end, intervals+1)
  plotFrame(time)
end

mkpath(dirname(plotsdir("SimpleLoop_intersection.gif")))
Plots.gif(anim, plotsdir("SimpleLoop_intersection.gif"), fps = 30)
