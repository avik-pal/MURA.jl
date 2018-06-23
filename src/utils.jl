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
