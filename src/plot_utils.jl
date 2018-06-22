using Plots

function plot_training(acc, cost)
  plotly()
  acc_train = acc["train"]
  acc_valid = acc["valid"]
  cost_train = acc["train"]
  cost_valid = acc["valid"]
  x = length(acc)

  p1 = plot(x, acc_train, xlabel = "Epochs", ylabel = "Accuracy", title = "Training Accuracy")

  p2 = plot(x, acc_valid, xlabel = "Epochs", ylabel = "Accuracy", title = "Validation Accuracy")

  p3 = plot(x, cost_train, xlabel = "Epochs", ylabel = "Cost", title = "Training Cost")

  p4 = plot(x, cost_valid, xlabel = "Epochs", ylabel = "Cost", title = "Validation Cost")

  plot(p1, p2, p3, p4, layout = (2, 2))
end
