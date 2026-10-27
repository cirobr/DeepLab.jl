using DeepLab
using Flux, Metalhead
import Metalhead: ResNet, DenseNet, EfficientNet
using Test

@testset "DeepLab.jl" begin
    include("backbones_tests.jl")
    include("deeplabv3_tests.jl")
end
