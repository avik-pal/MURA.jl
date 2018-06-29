using BSON: @save, @load

"""
  train_model()

This function takes no arguments. All the variables used in here
are to be `globally initialized`.

This function trains the `model` using the `data_dict["train"]`
dataset. I call to the function essentially signifies `1 epoch`.

Apart from these variable it expects `save_interval` and
`overwrite_save_interval` to be set prior to the call. THese essentially
help to automate the checkpointing the model.

At the end of the epoch the total loss and accuracy is printed but
if more detailed printing is needed then simply set `verbose` o `1`.

Also the `cost_metric` and `accuracy_metric` dictionaries are filled
up at the end of 1 epoch.
"""

function train_model()
  global model
  info("Starting to train model")
  start_time = time()
  i = 1
  local costs = []
  local accs = []
  for d in data_dict["train"]
    forward = model(d[1] |> gpu)
    res = d[2] |> gpu
    l = loss(res, forward, "train")
    a = accuracy(res, forward)
    if verbose == 1
      @show l
      @show a
    end
    if isnan(l) || isinf(l)
      continue # Donot backprop through nan or inf loss instead neglect it
      # This happens since the network is absolutely sertain about its prediction
      # So the loss becomes NaN as it doesn't handle this case
      # error("Model Weights have diverged. Change the Learning Rate")
    end
    push!(costs, Tracker.data(l))
    push!(accs, Tracker.data(a))
    Flux.back!(l)
    opt()
    end_time = time()
    if(end_time - start_time > save_interval)
      model = model |> cpu
      i == overwrite_save_interval ? i = 1 : i += 1
      save_path = model_save_path * "_i.bson"
      @save save_path model
      info("Model Saved at $(save_path)")
      model = model |> gpu
      start_time = time()
    end
  end
  push!(cost_metric["train"], sum(costs)/length(data_dict["train"]))
  push!(accuracy_metric["train"], sum(accs)/length(data_dict["train"]))
  info("Training Loss is $(cost_metric["train"][end]) Training Accuracy is $(accuracy_metric["train"][end])")
end

"""
  validate_model()

This function takes no arguments. All the variables used in here
are to be `globally initialized`.

This function tests the `model` using the `data_dict["valid"]`
dataset.

Also the `cost_metric` and `accuracy_metric` dictionaries are filled
up at the end. Additionally it displays the validation loss and
accuracy.
"""

function validate_model()
  global model
  info("Validating Model")
  Flux.testmode!(model)
  local costs = []
  local accs = []
  for d in data_dict["valid"]
    forward = model(d[1] |> gpu)
    res = d[2] |> gpu
    l = loss(res, forward, "train")
    a = accuracy(res, forward)
    if isnan(l) || isinf(l)
      continue # Donot backprop through nan or inf loss instead neglect it
    end
    push!(costs, Tracker.data(l))
    push!(accs, Tracker.data(a))
  end
  push!(cost_metric["valid"], sum(costs)/length(data_dict["valid"]))
  push!(accuracy_metric["valid"], sum(accs)/length(data_dict["valid"]))
  Flux.testmode!(model, false)
  info("Validation Loss is $(cost_metric["valid"][end]) Validation Accuracy is $(accuracy_metric["valid"][end])")
end
