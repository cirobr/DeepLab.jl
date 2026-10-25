![alt text](./images/logo-name-tm.png)

# DeepLab.jl

[![Build Status](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/TinyMachines.jl/actions/workflows/CI.yml?query=branch%3Amain)

A collection of tiny machine learning models for semantic image segmentation in IoT devices, written in Julia/Flux.

Besides regular mask outputs, all models deliver their internal feature maps as additional outputs, which are useful for model compression through knowledge distillation.


## DeepLabV3

UNet5 is the classic U-Net architecture, with five encoder/decoder levels. UNet4 has four levels.

Reference:
* "U-Net: Convolutional Networks for Biomedical Image Segmentation" ([arXiv](https://arxiv.org/abs/1505.04597)). Credits: Ronnenberger, Olaf; Fischer, Philipp; and Brox, Thomas.
