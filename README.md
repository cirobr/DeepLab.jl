![alt text](https://github.com/cirobr/TinyMachines.jl/blob/main/images/logo-name-tm.png?raw=true)

# DeepLab.jl

[![Build Status](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml?query=branch%3Amain)

DeepLab family of models for semantic segmentation.


## DeepLabV3

### Reference
* "Rethinking Atrous Convolution for Semantic Image Segmentation" ([arXiv](https://arxiv.org/abs/1706.05587)). Authors: Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam

### Syntax

```
DeepLabV3(ch_in::Int=3, ch_out::Int=19; backbone=ResNet(50; pretrain=false))
```

* bb32: ImageNet-pretrained ResNet-50 backbone (output stride=32, imported from Metalhead.jl)
* bb16: DeepLabv3 standard backbone, non-pretrained (OS=16)
* bb8: Non-pretrained ResNet-50 backbone, higher resolution (OS=8)



## Credits
Credits for the original architectures go to the references' authors, as aforementioned.

Credits for the implementations in Julia/Flux go to Ciro B Rosa.
* GitHub: https://github.com/cirobr
* LinkedIn: https://www.linkedin.com/in/cirobrosa/