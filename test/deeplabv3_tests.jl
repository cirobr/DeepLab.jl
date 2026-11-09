x = randn(Float32, (256, 256, 3, 1))

# Pakage default: ResNet-50
model50 = DeepLabV3()   # DeepLabV3(3,19)
@test size(model50(x)) == (256, 256, 19, 1)

# Paper standard: ResNet-50, modified for cummulative output stride = 16
model50_std = DeepLabV3(backbone=bb16)
@test size(model50_std(x)) == (256, 256, 19, 1)

# ResNet-101 (pretrained optional)
model101 = DeepLabV3(backbone=ResNet(101; pretrain=false))
@test size(model101(x)) == (256, 256, 19, 1)

# DenseNet-121
model_dn = DeepLabV3(backbone=DenseNet(121))
@test size(model_dn(x)) == (256, 256, 19, 1)

# EfficientNet-B0
model_eff = DeepLabV3(backbone=EfficientNet(:b0)) # or :b0 - :b8
@test size(model_eff(x)) == (256, 256, 19, 1)

# Aux head
model_aux = DeepLabV3(backbone=ResNet(50; pretrain=false), use_aux=true)
yhat = model_aux(x)
@test size(yhat) == (256, 256, 19, 1)

yhat, aux = model_aux(x, Val(:aux))
@test size(yhat) == (256, 256, 19, 1)
@test size(aux) == (256, 256, 19, 1)