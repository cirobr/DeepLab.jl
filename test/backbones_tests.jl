x = randn(Float32, (256, 256, 3, 1))

@test size(bb32(x)) == (8, 8, 2048, 1) # pre-trained ResNet-50 backbone
@test size(bb16(x)) == (16, 16, 2048, 1)
@test size(bb8(x))  == (32, 32, 2048, 1)
