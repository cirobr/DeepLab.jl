using DeepLab
using Flux, Metalhead
import Metalhead: ResNet, DenseNet, EfficientNet
using Test

@testset "DeepLab.jl" begin
    include("deeplab_tests.jl")
end
