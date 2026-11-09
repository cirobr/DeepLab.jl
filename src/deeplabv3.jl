# ---------- Flexible DeepLabV3 ----------
struct DeepLabV3
    backbone_head          # Chain up to (and including) aux stage
    backbone_tail           # Chain after aux stage
    aspp::ASPP
    classifier::Conv
    aux_classifier::Union{Chain, Nothing}
end

"""
    DeepLabV3(ch_in=3, ch_out=19; backbone=..., use_aux=false, aux_index=4, aspp_rates=(6,12,18))

Create a DeepLabv3-style model.
- `use_aux=true` adds an auxiliary classifier on intermediate features.
- `aux_index` selects which layer of th@info "metrics OK"

e backbone Chain to tap
  (ResNet: 4 = after stage 3).
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

    # Split backbone
    backbone_head = Chain(bb[1:aux_index]...)
    backbone_tail = Chain(bb[aux_index+1:end]...)

    # Detect final channels
    dummy = randn(Float32, 64, 64, ch_in, 1)
    feat = backbone_tail(backbone_head(dummy))
    in_ch = size(feat, 3)

    aspp = ASPP(in_ch, 256; rates=aspp_rates)
    classifier = Conv((1,1), 256 => ch_out)

    # Auxiliary head (optional)
    aux_classifier = nothing
    if use_aux
        aux_feat = backbone_head(dummy)
        aux_ch = size(aux_feat, 3)

        aux_classifier = Chain(
            Conv((3,3), aux_ch => 256; pad=1, bias=false),
            BatchNorm(256),
            relu,
            Dropout(0.1),
            Conv((1,1), 256 => ch_out)
        )
    end

    DeepLabV3(backbone_head, backbone_tail, aspp, classifier, aux_classifier)
end

# Inference / single-output forward
function (m::DeepLabV3)(x)
    feat = m.backbone_tail(m.backbone_head(x))
    aspp_out = m.aspp(feat)
    logits = m.classifier(aspp_out)
    return Flux.upsample_bilinear(logits, size=size(x)[1:2])
end

# Training forward: returns (main_logits, aux_logits)
function (m::DeepLabV3)(x, ::Val{:aux})
    isnothing(m.aux_classifier) && error("Model was constructed without an auxiliary head (use_aux=false)")

    aux_feat = m.backbone_head(x)
    feat     = m.backbone_tail(aux_feat)

    # Main head
    aspp_out = m.aspp(feat)
    main_logits = m.classifier(aspp_out)
    main_logits = Flux.upsample_bilinear(main_logits, size=size(x)[1:2])

    # Aux head
    aux_logits = m.aux_classifier(aux_feat)
    aux_logits = Flux.upsample_bilinear(aux_logits, size=size(x)[1:2])

    return main_logits, aux_logits
end