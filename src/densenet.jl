using Flux, CuArrays, Metalhead

struct Bottleneck
  layer
end

Flux.treelike(Bottleneck)

Bottleneck(in_planes, growth_rate) = Bottleneck(
                                           Chain(BatchNorm(in_planes, relu),
                                           Conv((1, 1), in_planes=>4growth_rate),
                                           BatchNorm(4growth_rate, relu),
                                           Conv((3, 3), 4growth_rate=>growth_rate, pad = (1, 1))))

(b::Bottleneck)(x) = cat(3, b.layer(x), x)

Transition(chs::Pair{<:Int, <:Int}) = Chain(BatchNorm(chs[1], relu),
                                            Conv((1, 1), chs),
                                            x -> meanpool(x, (2, 2)))

function _make_dense_layers(block, in_planes, growth_rate, nblock)
  local layers = []
  for i in 1:nblock
    push!(layers, block(in_planes, growth_rate))
    in_planes += growth_rate
  end
  Chain(layers...)
end

function _DenseNet(nblocks; in_chs = 1, block = Bottleneck, growth_rate = 12, reduction = 0.5, num_classes = 1)
  num_planes = 2growth_rate
  layers = []
  push!(layers, Conv((7, 7), in_chs=>num_planes, stride = (2, 2), pad = (3, 3)))
  push!(layers, BatchNorm(num_planes, relu))
  push!(layers, x -> maxpool(x, (3, 3), stride = (2, 2), pad = (1, 1)))

  for i in 1:3
    push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[i]))
    num_planes += nblocks[i] * growth_rate
    out_planes = Int(floor(num_planes * reduction))
    push!(layers, Transition(num_planes=>out_planes))
    num_planes = out_planes
  end

  push!(layers, _make_dense_layers(block, num_planes, growth_rate, nblocks[4]))
  num_planes += nblocks[4] * growth_rate
  push!(layers, BatchNorm(num_planes, relu))

  Chain(layers..., x -> meanpool(x, (7, 7)),
        x -> reshape(x, :, size(x, 4)),
        Dense(num_planes, num_classes), x -> σ.(x))
end

"""
  get_densenet_model(depth, pretrained::Bool = false)

Creates the Densenet model as per the arguments passed.

Arguments:
1. `depth`: The depth of the Densenet Model. The current allowed
            values are `121`, `169`, `201` and `264`.
2. `pretrained`: Gets the network with pretrained Imagenet weights.
                 `Metalhead.jl` currently has only the weights for
                 Densenet 121 so sending `depth` anything other than
                 `121` results in an **ERROR**.
"""

function get_densenet_model(depth, pretrained::Bool = false)
  if(pretrained && depth != 121)
    error("Pretrained weights are only available for Densenet 121")
  end
  if depth == 121
    if pretrained
      DenseNet() |> gpu
    else
      _DenseNet([6, 12, 24, 16]) |> gpu
    end
  elseif depth == 169
    _DenseNet([6, 12, 32, 32]) |> gpu
  elseif depth == 201
    _DenseNet([6, 12, 48, 32]) |> gpu
  elseif depth == 264
    _DenseNet([6, 12, 64, 48]) |> gpu
  else
    error("No such model is available")
  end
end
