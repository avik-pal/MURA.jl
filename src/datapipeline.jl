using DataFrames, CSV, Images
using Base.Iterators: partition

data_cat = ["train", "valid"]

function get_count_df(path_1, path_2, category)
  df_1 = CSV.read(path_1, header = ["path", "label"])
  df_2 = CSV.read(path_2, header = ["path"])
  df_1[:count] = 0
  i = 1
  for (j, paths) in enumerate(df_1[:path])
    while(paths == df_2[:path][i][1:end-10])
      df_1[:count][j] += 1
      i += 1
    end
  end
  if(category == "train")
    global df_train = df_1
  else
    global df_valid = df_1
  end
end

function get_indices(study_type, category)
  start_ind = 0
  end_ind = 0
  if category == "train"
    df = df_train
  else
    df = df_valid
  end
  for (i, path) in enumerate(df[:path])
    if split(path, "/")[3] == study_type
      if start_ind == 0
        start_ind = i
      end
    else
      if start_ind != 0 && end_ind == 0
        end_ind = i - 1
        break
      end
    end
  end
  if end_ind == 0
    end_ind = size(df)[1]
  end
  (start_ind, end_ind)
end

function get_study_data(study_type)
  a1, b1 = get_indices(study_type, "train")
  a2, b2 = get_indices(study_type, "valid")
  Dict("train" => df_train[a1:b1, :], "valid" => df_valid[a2:b2, :])
end

im_mean = reshape([0.485, 0.456, 0.406], 1, 1, 3)
im_std = reshape([0.229, 0.224, 0.225], 1, 1, 3)

normalize(img) = (img .- im_mean) ./ im_std

denormalize(img) = img .* im_std .+ im_std

im2arr(img) = permutedims(float.(channelview(imresize(img, (224, 224)))), [3, 2, 1])

function get_batched_images(study_type, batch_size; path_t1 = "",
                            path_t2 = "", path_v1 = "", path_v2 = "")
  try
    df_train[1]
  catch
    warn("df_train was not loaded")
    get_count_df(path_t1, path_t2, "train")
  end
  try
    df_valid[1]
  catch
    warn("df_valid was not loaded")
    get_count_df(path_v1, path_v2, "valid")
  end
  dict = get_study_data(study_type)
  images = Dict("train_imgs" => [], "train_labs" => [],
                "valid_imgs" => [], "valid_labs" => [])
  for cate in data_cat
    for (i, path) in enumerate(dict[cate][:path])
      for j in 1:dict[cate][:count][i]
        push!(images[cate * "_imgs"], normalize(im2arr(load(dict[cate][:path] * "_image$(j).png"))))
        push!(images[cate * "_labs"], dict[cate][:count][i])
      end
    end
  end
  Dict("train" => [(cat(4, images["train_imgs"][i]...), images["train_labs"][i])
        for i in partition(1:length(images["train"]), batch_size)],
       "valid" => [(cat(4, images["valid_imgs"][i]...), images["valid_labs"][i])
        for i in partition(1:length(images["valid"]), batch_size)])
end