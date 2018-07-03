include("datapipeline.jl")
include("densenet.jl")
include("utils.jl")
include("train.jl")
using CUDAnative

#--------------Hyperparameters-------------
model = get_densenet_model(169)
threshold = 0.5
lr = 0.0001
batch_size = 16
which_cat = "XR_HAND"
save_interval = 300
overwrite_save_interval = 3 # Overwrite the model after `x` saves. If `never` overwrite set it to `0`
model_save_path = "../saved_models/model_$(which_cat)" # Donot put the extension it is automatically managed
verbose = 1
epochs = 1
plot_save_path = "../saved_plots/plot_$(which_cat).png"
#------------------------------------------

data_dict = get_batched_images(which_cat, batch_size, path_t1 = "../MURA-v1.1/train_labeled_studies.csv", path_t2 = "../MURA-v1.1/train_image_paths.csv", path_v1 = "../MURA-v1.1/valid_labeled_studies.csv", path_v2 = "../MURA-v1.1/valid_image_paths.csv")

counts = get_count(data_dict)

weight_1 = Dict(i => counts[1][i]/(counts[1][i] + counts[2][i]) for i in ["train", "valid"])

weight_2 = Dict(i => counts[2][i]/(counts[1][i] + counts[2][i]) for i in ["train", "valid"])

loss(x_true, x_pred, cat) = mean(- weight_2[cat] * x_true .* CUDAnative.log.(x_pred) - weight_1[cat] * (1.0 - x_true) .* CUDAnative.log.(1.0 - x_pred))

accuracy(x_true, x_pred, threshold = threshold) = mean((x_pred .>= threshold) .== x_true)

preci(mat) = mat[2, 2] / (mat[2, 2] + mat[1, 2])

recall(mat) = mat[2, 2] / (mat[2, 2] + mat[2, 1])

function f1_score(mat)
  p = preci(mat)
  r = recall(mat)
  2 * p * r / (p + r)
end

cost_metric = Dict("train" => [], "valid" => [])
accuracy_metric = Dict("train" => [], "valid" => [])

opt = ADAM(params(model), lr)

for i in 1:epochs
  info("Starting Epoch $i")
  Flux.Optimise.@interrupts train_model()
  @save model_save_path*"_end_epoch.bson" model
  Flux.Optimise.@interrupts validate_model()
  info("Epoch $i Complete")
end

plot = plot_training(accuracy_metric, cost_metric)

savefig(plot, plot_save_path)
