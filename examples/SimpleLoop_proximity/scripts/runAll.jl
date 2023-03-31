using DrWatson
@quickactivate "SimpleLoop_proximity"

#include(srcdir("genAllData.jl"))
#include(srcdir("trainFlux.jl"))
#include(srcdir("simulateFMU.jl"))
include(srcdir("plotStuff.jl"))

modelName = "simpleLoop"
N = 1_000

#genAllData(modelName, N)
#trainFlux(modelName, N; nepochs=500)
#simulateFMU(modelName, N)
#calcLoss(modelName, N)
plotStuff(modelName, N, fileType="png")
#for p in [1,2,3,4,5,10,20,30]
#  SurrogateRefRes(modelName, N; fileType="png", proximity=p)
#end
