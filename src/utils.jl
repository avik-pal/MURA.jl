using Plots

"""
  plot_training(acc, cost)

Plot the `Cost vs Epochs` and `Accuracy vs Epochs` graphs
for both the Training and Validation Set.

Arguments:
1. `acc`: A dictionary with keys `train` and `valid` and stores an
          array corresponding to the accuracy during training.
2. `cost`: A dictionary with keys `train` and `valid` and stores an
           array corresponding to the cost during training.
"""

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

"""
  get_count(dataset)

Counts the number of abnormal images and the number of normal
images in the dataset.

Arguments:
1. `dataset`: For the input format see the output of `get_batched_images`.
              Any other input will not give proper results. Ideally pass
              the output from the `get_batched_images` function.
"""

function get_count(dataset)
  local categories = ["train", "valid"]
  total_abnormal_imgs = Dict("train" => 0, "valid" => 0)
  total_normal_imgs = Dict("train" => 0, "valid" => 0)
  for cat in categories
    for tup in dataset[cat]
      for i in tup[2]
        if i == 1
          total_abnormal_imgs[cat] += 1
        else
          total_normal_imgs[cat] += 1
        end
      end
    end
  end
  (total_abnormal_imgs, total_normal_imgs)
end

"""
  confusion_matrix(dataset)

Generates the confusion matrix based on the predicted values of the
model and the ground truth. The `x-axis` corresponds to the `ground truth`
and `y-axis` is the predicted value.

Arguments:
1. `dataset`: It must be an `array of tuples` or `array of arrays`
              where the first element of the `tuple` or `array` must
              be the `data` and the second element must be the `label`.
"""

function confusion_matrix(dataset)
  conf_mat = zeros(Int64, 2, 2)
  Flux.testmode!(model)
  for data in dataset
    pred = model(data[1] |> gpu)
    output = Int.(pred .>= threshold)
    for i in 1:length(output)
      conf_mat[output[i] + 1, Int(data[2][j] + 1)] += 1
    end
  end
  Flux.testmode!(model, false)
  conf_mat
end

"""
  Κ_score(conf_mat)

Return the kappa score from the confusion matrix

Arguments:
1. `conf_mat`: Confusion matrix. The orientation must be same as
               the output of `confusion_matrix` function.
"""

function Κ_score(conf_mat)
  tot_instances = sum(conf_mat)
  pₒ = (conf_mat[1, 1] + conf_mat[2, 2])/tot_instances
  pₑ = (sum(conf_mat[1, :]) * sum(conf_mat[:, 1]) + sum(conf_mat[:, 2]) * sum(conf_mat[2, :]))/(tot_instances ^ 2)
  (pₒ - pₑ)/(1 - pₑ)
end
