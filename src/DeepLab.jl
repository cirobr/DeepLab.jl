module DeepLab


export DeepLabV3

import Flux
import Flux: Chain, Conv, BatchNorm, AdaptiveMeanPool, Dropout, relu
import Metalhead
import Metalhead:ResNet

include("aspp.jl")
include("deeplabv3.jl")


end