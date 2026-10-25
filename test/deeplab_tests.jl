# Default: ResNet-50
model50 = DeepLabV3()   # DeepLabV3(3,19)
x = randn(Float32, 256, 256, 3, 1)
@test size(model50(x)) == (256, 256, 19, 1)

# ResNet-101 (pretrained optional)
model101 = DeepLabV3(backbone=ResNet(101; pretrain=false))
@test size(model101(x)) == (256, 256, 19, 1)

# DenseNet-121
model_dn = DeepLabV3(backbone=DenseNet(121))
@test size(model_dn(x)) == (256, 256, 19, 1)

# EfficientNet-B0
model_eff = DeepLabV3(backbone=EfficientNet(:b0)) # or :b0 - :b8
@test size(model_eff(x)) == (256, 256, 19, 1)

# You can also pass an already extracted backbone
bb = Metalhead.backbone(ResNet(50; pretrain=true))
model_custom = DeepLabV3(backbone=bb)
@test size(model_custom(x)) == (256, 256, 19, 1)