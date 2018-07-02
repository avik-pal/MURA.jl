using Flux: Tracker
using Flux.Tracker: data

#--------------------General Utilities---------------------

function get_topk(probs, k = 5)
  T = eltype(probs)
  prob = Array{T, 1}()
  idx = Array{Int, 1}()
  while(k!=0)
    push!(idx, indmax(probs))
    push!(prob, probs[idx[end]])
    probs[idx[end]] = 0.0
    k -= 1
  end
  (prob, idx)
end

function one_hot_encode(preds, idx)
  one_hot = zeros(eltype(preds.data), size(preds)[1], 1)
  one_hot[idx ,1] = 255.0
  one_hot |> gpu
end

im2arr_rgb(img) = permutedims(float.(channelview(imresize(img, (224, 224)))), (3, 2, 1))

# Just a hack to add the guided relu functions and replace the relu
function change_activation(model::Chain, activation)
  updated_model = []
  for l in model.layers
    if typeof(l) <: Dense
      push!(updated_model, Dense(l.W, l.b, identity))
      push!(updated_model, activation)
    elseif typeof(l) <: Conv
      push!(updated_model, Conv(identity, l.weight, l.bias, l.stride, l.pad, l.dilation))
      push!(updated_model, activation)
    elseif typeof(l) <: BatchNorm
      push!(updated_model, BatchNorm(identity, l.β, l.γ, l.μ, l.σ, l.ϵ, l.momentum, l.active))
      push!(updated_model, activation)
    else
      push!(updated_model, l)
    end
  end
  Chain(updated_model...)
end

#--------------------BackPropagation----------------------

struct Backprop
  model::Chain
end

function (m::Backprop)(img, top = 1)
  if !Tracker.istracked(img)
    img = param(img) |> gpu
  end
  preds = m.model(img)
  probs = softmax(preds)
  prob, inds = get_topk(probs.data, top)
  grads = []
  for (i, idx) in enumerate(inds)
    Flux.back!(preds, one_hot_encode(preds, idx))
    push!(grads, (clamp.(img.grad |> cpu, 0.0, 1.0), prob[i], idx))
    img.grad .= 0.0
  end
  grads
end

#-----------------Vanilla BackPropagation-----------------

VanillaBackprop(model::Chain) = Backprop(model[1:end-1] |> gpu)

#------------------Guided BackPropagation-----------------

guided_relu1(x) = max.(zero(x), x)

guided_relu1(x::TrackedArray) = Tracker.track(guided_relu1, x)

Tracker.back(::typeof(guided_relu1), Δ, x) = Tracker.@back(x, Int.(x .> zero(x)) .* max.(zero(Δ), Δ))

function GuidedBackprop(model::Chain)
  model = change_activation(model, guided_relu1)
  Backprop(model[1:end-1] |> gpu)
end

#---------------------Deconvolution-----------------------

guided_relu2(x) = max.(zero(x), x)

guided_relu2(x::TrackedArray) = Tracker.track(guided_relu2, x)

Tracker.back(::typeof(guided_relu2), Δ, x) = Tracker.@back(x, max(zero(Δ), Δ))

function Deconvolution(model::Chain)
  model = change_activation(model, guided_relu2)
  Backprop(model[1:end-1] |> gpu)
end

#------------------------Grad CAM--------------------------

#--------------------Guided Grad CAM-----------------------

#----------------------Usage Notes-------------------------

# img = reshape(normalize(im2arr(load("cat_dog.png"))) * 255, 224, 224, 3, 1);
# m = VanillaBackprop(model.layers)
# m(img)
# x = permutedims(reshape(m[1][1], 224, 224, 3), (3, 2, 1));
# colorview(RGB{eltype(x)}, x)
# Gray.(colorview(RGB{eltype(x)}, x))
