x = randn(Float32, (256, 256, 3, 1))

bb32 = resnet50_backbone(os=32, pretrain=false)
@test size(bb32(x)) == (8, 8, 2048, 1)

bb32 = resnet50_backbone(os=32, pretrain=true)
@test size(bb32(x)) == (8, 8, 2048, 1)

@test size(bb16(x)) == (16, 16, 2048, 1)

@test size(bb8(x))  == (32, 32, 2048, 1)
