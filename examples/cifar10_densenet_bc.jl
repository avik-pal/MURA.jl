using Flux, CuArrays, Metalhead, Images
using Metalhead: trainimgs
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using Base.Iterators: partition
using BSON: @save

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

function DenseNet(nblocks; block = Bottleneck, growth_rate = 12, reduction = 0.5, num_classes = 10)
    num_planes = 2growth_rate
    layers = []
    push!(layers, Conv((3, 3), 3=>num_planes, pad = (1, 1)))

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

    Chain(layers..., x -> meanpool(x, (4, 4)),
          x -> reshape(x, :, size(x, 4)),
          Dense(num_planes, num_classes), softmax)
end

densenet = DenseNet([6, 12, 24, 16]) |> gpu

im_mean = reshape([0.5071, 0.4867, 0.4408], 1, 1, 3)
im_std = reshape([0.2675, 0.2565, 0.2761], 1, 1, 3)

im_to_arr(x) = (permutedims(float.(channelview(x)), [3, 2, 1]) .- im_mean) ./ im_std

X = trainimgs(CIFAR10)
imgs = [im_to_arr(X[i].img) for i in 1:50000]
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
train = [(cat(4, imgs[i]...), labels[:,i]) for i in partition(1:50000, 32)]

loss(x, y) = crossentropy(x, y)

accuracy(x, y) = mean(argmax(x, 1:10) .== argmax(y, 1:10))

opt = ADAM(params(densenet))

@epochs 5 begin
    for data in train
        forward = (densenet(data[1] |> gpu), data[2] |> gpu)
        l = loss(forward[1], forward[2])
        println("Loss = $l and Accuracy = $(accuracy(forward[1], forward[2]))")
        Flux.back!(l)
        opt()
    end
    densenet = densenet |> cpu
    @save "cifar10_checkpoint.bson" densenet
    densenet = densenet |> gpu
end
