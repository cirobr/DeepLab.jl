
"""
    resnet_stride_os(os::Int)

Return a `stride_fn` that produces the requested output stride.
Supported values: 32, 16, 8.

- OS=32 → standard ResNet (stage 4 downsamples)
- OS=16 → stage 4 has stride 1
- OS=8  → stages 3 and 4 have stride 1
"""
function resnet_stride_os(os::Int)
    os in (8, 16, 32) || error("OS must be 8, 16 or 32")

    function stride_fn(stage_idx::Integer, block_idx::Integer)
        # Default ResNet behaviour
        default = Metalhead.resnet_stride(stage_idx, block_idx)

        if os == 32
            return default
        elseif os == 16
            # keep stage 4 at stride 1
            return stage_idx == 4 ? 1 : default
        else  # os == 8
            # keep stages 3 and 4 at stride 1
            return stage_idx >= 3 ? 1 : default
        end
    end

    return stride_fn
end

"""
    resnet50_backbone(; os::Int=16, pretrain::Bool=false)

Build a ResNet-50 feature extractor with the given output stride.
"""
function resnet50_backbone(; os::Int=16, pretrain::Bool=false)
    stride_fn = resnet_stride_os(os)

    model = Metalhead.resnet(
        Metalhead.bottleneck,
        [3, 4, 6, 3];
        stride_fn = stride_fn,
        nclasses = 1000
    )

    # Extract backbone (drop classifier)
    bb = model.layers[1]

    # Optional: load pretrained weights only when OS=32
    if pretrain
        os == 32 || @warn "Pretrained weights are only valid for OS=32; ignoring pretrain=true"
        if os == 32
            # Load via the high-level constructor and copy (or just use Metalhead.backbone)
            pretrained = Metalhead.backbone(ResNet(50; pretrain=true))
            # Note: parameter copying would be needed if you want to transfer weights
            # into the custom model. For OS≠32 this is not straightforward.
            return pretrained
        end
    end

    return bb
end

# DeepLabv3 default (OS=16)
bb16 = resnet50_backbone(os=16)

# Higher resolution (OS=8)
bb8  = resnet50_backbone(os=8)
