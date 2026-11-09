![alt text](https://github.com/cirobr/TinyMachines.jl/blob/main/images/logo-name-tm.png?raw=true)

# DeepLab.jl

[![Build Status](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml?query=branch%3Amain)

DeepLab family of models for semantic segmentation.


## DeepLabV3

### Reference
* "Rethinking Atrous Convolution for Semantic Image Segmentation" ([arXiv](https://arxiv.org/abs/1706.05587)). Authors: Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam

### Syntax

```julia
DeepLabV3(ch_in=3, ch_out=19;              # input/output channels
    backbone=ResNet(50; pretrain=false),   # backbone
    use_aux=false,                         # auxiliary head
    aux_index=4,                           # index for aux head extraction
    aspp_rates=(6,12,18),                  # aspp dilation rates
)
```

Generalized aux head loss function detection (example):

```julia
function lossfn(model, X, y)
      if isnothing(model.aux_classifier)
            yhat = model(X)
            return Flux.mse(yhat, y)
      else
            yhat, aux = model(X, Val(:aux))
            return Flux.mse(yhat, y) + 0.4f0 * Flux.mse(aux, y)
      end
end
```

Convenience backbones:

```julia
backbone=bb32  # ImageNet-pretrained ResNet-50 backbone (output stride=32, imported from Metalhead.jl)
backbone=bb16  # DeepLabv3 standard backbone, non-pretrained (OS=16)
backbone=bb8   # Non-pretrained ResNet-50 backbone, higher resolution (OS=8)
```


## Credits
Credits for the original architecture go to the references' authors, as aforementioned.

Credits for the implementation in Julia/Flux go to Ciro B Rosa.
* GitHub: https://github.com/cirobr
* LinkedIn: https://www.linkedin.com/in/cirobrosa/