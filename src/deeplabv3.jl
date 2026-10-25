# ---------- Flexible DeepLabV3 ----------
struct DeepLabV3
    backbone
    aspp::ASPP
    classifier::Conv
end

# function DeepLabV3(; num_classes::Int=19,
#                    backbone=ResNet(50; pretrain=false),
#                    in_channels::Int=3)
"""
    DeepLabV3(; num_classes=19, backbone=ResNet(50), in_channels=3)

Create a DeepLabv3-style model.
- `backbone`: any Metalhead model (or its backbone Chain). Default is ResNet(50).
  Examples: `ResNet(101)`, `DenseNet(121)`, `EfficientNet(0)`, `Xception()`, etc.
- Channels of the ASPP are detected automatically from the backbone.
"""
function DeepLabV3(ch_in::Int=3, ch_out::Int=19; backbone=ResNet(50; pretrain=false))
    # Extract backbone safely
    bb = if backbone isa Chain
        backbone
    else
        try
            Metalhead.backbone(backbone)
        catch
            # Fallback: many models store the feature extractor in .layers[1]
            hasproperty(backbone, :layers) ? backbone.layers[1] : backbone
        end
    end

    # Detect output channels
    dummy = randn(Float32, 64, 64, ch_in, 1)
    feat = bb(dummy)
    in_ch = size(feat, 3)

    aspp = ASPP(in_ch, 256)
    classifier = Conv((1,1), 256 => ch_out)

    DeepLabV3(bb, aspp, classifier)
end

function (m::DeepLabV3)(x)
    feat = m.backbone(x)
    aspp_out = m.aspp(feat)
    logits = m.classifier(aspp_out)
    Flux.upsample_bilinear(logits, size=size(x)[1:2])
end