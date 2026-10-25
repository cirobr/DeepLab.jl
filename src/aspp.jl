# ---------- ASPP (unchanged, just takes in_channels) ----------
struct ASPP
    branches::Vector{Chain}
    project::Chain
end

function ASPP(in_channels::Int, out_channels::Int=256;
              rates::NTuple{3,Int}=(6, 12, 18))
    branches = Chain[]

    # 1×1
    push!(branches, Chain(Conv((1,1), in_channels => out_channels; bias=false),
                          BatchNorm(out_channels), relu))

    # dilated
    for r in rates
        push!(branches, Chain(Conv((3,3), in_channels => out_channels; pad=r, dilation=r, bias=false),
                              BatchNorm(out_channels), relu))
    end

    # GAP
    push!(branches, Chain(AdaptiveMeanPool((1,1)),
                          Conv((1,1), in_channels => out_channels; bias=false),
                          BatchNorm(out_channels), relu))

    project = Chain(
        Conv((1,1), out_channels * 5 => out_channels; bias=false),
        BatchNorm(out_channels),
        relu,
        Dropout(0.1)
    )

    ASPP(branches, project)
end

function (m::ASPP)(x)
    H, W = size(x)[1:2]
    outs = map(m.branches) do b
        y = b(x)
        size(y)[1:2] == (H, W) ? y : Flux.NNlib.upsample_bilinear(y, size=(H, W))
    end
    cat(outs...; dims=3) |> m.project
end