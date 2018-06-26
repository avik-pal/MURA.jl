using BSON: @save, @load

function train_model()
  info("Starting to train model")
  start_time = time()
  i = 1
  local costs = []
  local accs = []
  for d in data["train"]
    forward = model(d[1] |> gpu)
    res = d[2] |> gpu
    l = loss(res, forward, "train")
    push!(costs, l)
    a = accuracy(res, forward)
    push!(accs, a)
    if verbose == 1
      @show l
      @show a
    end
    Flux.back!(l)
    opt()
    end_time = time()
    if(end_time - start_time > save_interval)
      model = model |> cpu
      i == overwrite_save_interval ? i = 1 : i += 1
      save_path = model_save_path * "_i.bson"
      @save model save_path
      info("Model Saved at $(save_path)")
      model = model |> gpu
      start_time = time()
    end
  end
  push!(cost_metric["train"], sum(costs)/length(data))
  push!(accuracy_metric["train"], sum(accs)/length(data))
  info("Training Loss is $(cost_metric["train"][end]) Training Accuracy is $(accuracy_metric["train"][end])")
end

function validate_model()
  info("Validating Model")
  local costs = []
  local accs = []
  for d in data["valid"]
    forward = model(d[1] |> gpu)
    res = d[2] |> gpu
    l = loss(res, forward, "train")
    push!(costs, l)
    a = accuracy(res, forward)
    push!(accs, a)
  end
  push!(cost_metric["valid"], sum(costs)/length(data))
  push!(accuracy_metric["valid"], sum(accs)/length(data))
  info("Validation Loss is $(cost_metric["valid"][end]) Validation Accuracy is $(accuracy_metric["valid"][end])")
end
