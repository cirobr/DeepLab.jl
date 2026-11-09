# ---------- Flexible DeepLabV3 ----------
struct DeepLabV3
    backbone          # full backbone Chain
    aspp::ASPP
    classifier::Conv
    aux_classifier::Union{Chain, Nothing}   # optional
    aux_index::Int                          # which stage to attach aux (default 4 = after stage 3)
end

"""
    DeepLabV3(ch_in=3, ch_out=19; backbone=..., use_aux=false, aux_index=4, aspp_rates=(6,12,18))

Create a DeepLabv3-style model.
- `use_aux=true` adds an auxiliary classifier on intermediate features.
- `aux_index` selects which layer of the backbone Chain to tap
  (ResNet: 4 = after stage 3, which is the usual choice).
- `aspp_rates` controls the dilation rates of the ASPP module.
"""
function DeepLabV3(ch_in::Int=3, ch_out::Int=19;
                   backbone=ResNet(50; pretrain=false),
                   use_aux::Bool=false,
                   aux_index::Int=4,
                   aspp_rates::NTuple{3,Int}=(6, 12, 18))

    # Extract backbone safely
    bb = if backbone isa Chain
        backbone
    else
        try
            Metalhead.backbone(backbone)
        catch
            hasproperty(backbone, :layers) ? backbone.layers[1] : backbone
        end
    end

    # Detect final channels
    dummy = randn(Float32, 64, 64, ch_in, 1)
    feat = bb(dummy)
    in_ch = size(feat, 3)

    aspp = ASPP(in_ch, 256; rates=aspp_rates)
    classifier = Conv((1,1), 256 => ch_out)

    # Auxiliary head (optional)
    aux_classifier = nothing
    if use_aux
        # Run up to aux_index to discover intermediate channels
        x = dummy
        for i in 1:aux_index
            x = bb[i](x)
        end
        aux_ch = size(x, 3)

        aux_classifier = Chain(
            Conv((3,3), aux_ch => 256; pad=1, bias=false),
            BatchNorm(256),
            relu,
            Dropout(0.1),
            Conv((1,1), 256 => ch_out)
        )
    end

    DeepLabV3(bb, aspp, classifier, aux_classifier, aux_index)
end

# Inference / single-output forward (always returns main logits)
function (m::DeepLabV3)(x)
    feat = m.backbone(x)
    aspp_out = m.aspp(feat)
    logits = m.classifier(aspp_out)
    return Flux.upsample_bilinear(logits, size=size(x)[1:2])
end

# Training forward: returns (main_logits, aux_logits) when aux is present
function (m::DeepLabV3)(x, ::Val{:aux})
    isnothing(m.aux_classifier) && error("Model was constructed without an auxiliary head (use_aux=false)")

    # Run backbone up to aux_index
    h = x
    for i in 1:m.aux_index
        h = m.backbone[i](h)
    end
    aux_feat = h

    # Continue to the end for the main path
    for i in (m.aux_index+1):length(m.backbone)
        h = m.backbone[i](h)
    end
    feat = h

    # Main head
    aspp_out = m.aspp(feat)
    main_logits = m.classifier(aspp_out)
    main_logits = Flux.upsample_bilinear(main_logits, size=size(x)[1:2])

    # Aux head
    aux_logits = m.aux_classifier(aux_feat)
    aux_logits = Flux.upsample_bilinear(aux_logits, size=size(x)[1:2])

    return main_logits, aux_logits
end
