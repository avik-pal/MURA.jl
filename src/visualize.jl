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
  one_hot = zeros(eltype(data(preds)), size(preds)[1], 1)
  one_hot[idx ,1] = 1000.0
  one_hot |> gpu
end

im2arr_rgb(img) = permutedims(float.(channelview(imresize(img, (224, 224)))), (3, 2, 1))

#-----------------Vanilla BackPropagation-----------------

module VanillaBackprop

using ..get_topk, ..one_hot_encode
using ..Tracker, ..data, ..Flux
using Flux: relu

struct Backprop
  model::Chain
end

create(model) = Backprop((model.layers[1:end-1]) |> gpu)

function (m::Backprop)(img, top = 1)
  if !Tracker.istracked(img)
    img = param(img) |> gpu
  end
  preds = m.model(img)
  probs = softmax(preds)
  prob, inds = get_topk(data(probs), top)
  grads = []
  for (i, idx) in enumerate(inds)
    Flux.back!(preds, one_hot_encode(preds, idx))
    push!(grads, (clamp.(img.grad |> cpu, 0.0, 1.0), prob[i], idx))
    img.grad .= 0.0
  end
  grads
end

end

#------------------Guided BackPropagation-----------------

module GuidedBackprop

using ..get_topk, ..one_hot_encode
using ..Tracker, ..data, ..Flux
using Flux: relu

relu(x::TrackedReal) = Tracker.track(relu, x)

Tracker.back(::typeof(relu), Δ, x) = Tracker.@back(x, Int(x > zero(x)) * max(zero(Δ), Δ))

struct Backprop
  model::Chain
end

create(model) = Backprop((model.layers[1:end-1]) |> gpu)

function (m::Backprop)(img, top = 1)
  if !Tracker.istracked(img)
    img = param(img) |> gpu
  end
  preds = m.model(img)
  probs = softmax(preds)
  prob, inds = get_topk(data(probs), top)
  grads = []
  for (i, idx) in enumerate(inds)
    Flux.back!(preds, one_hot_encode(preds, idx))
    push!(grads, (clamp.(img.grad |> cpu, 0.0, 1.0), prob[i], idx))
    img.grad .= 0.0
  end
  grads
end

end

#---------------------Deconvolution-----------------------

module Deconv

using ..get_topk, ..one_hot_encode
using ..Tracker, ..data, ..Flux
using Flux: relu

relu(x::TrackedReal) = Tracker.track(relu, x)

Tracker.back(::typeof(relu), Δ, x) = Tracker.@back(x, max(zero(Δ), Δ))

struct Backprop
  model::Chain
end

create(model) = Backprop((model.layers[1:end-1]) |> gpu)

function (m::Backprop)(img, top = 1)
  if !Tracker.istracked(img)
    img = param(img) |> gpu
  end
  preds = m.model(img)
  probs = softmax(preds)
  prob, inds = get_topk(data(probs), top)
  grads = []
  for (i, idx) in enumerate(inds)
    Flux.back!(preds, one_hot_encode(preds, idx))
    push!(grads, (clamp.(img.grad |> cpu, 0.0, 1.0), prob[i], idx))
    img.grad .= 0.0
  end
  grads
end

end

#------------------------Grad CAM--------------------------

#--------------------Guided Grad CAM-----------------------

#----------------------Usage Notes-------------------------

# img = reshape(normalize(im2arr(load("cat_dog.png"))) * 255, 224, 224, 3, 1);
# img = param(img) |> gpu;
# model = model.layers[1:end-1]
# preds = backpropagation(img);
# probs = softmax(preds);
# prob, inds = get_topk(probs.data)
# ∇backpropagation(preds, inds[1])
# x = permutedims(reshape(clamp.(img.grad, 0.0, 1.0), 224, 224, 3), (3, 2, 1)) |> cpu;
# colorview(RGB{eltype(x)}, x)
