module DeepLab


export DeepLabV3
export resnet50_backbone, bb16, bb8

import Flux
import Flux: Chain, Conv, BatchNorm, AdaptiveMeanPool, Dropout, relu
import Metalhead
import Metalhead:ResNet

include("aspp.jl")
include("backbones.jl")
include("deeplabv3.jl")


end