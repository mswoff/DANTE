Here are pretrained models for some of the datasets.

Under each dataset are folders (i.e. "pair_predictions_[model_num]") which correspond to a specific model trained on one of the folds and tested on the appropriate testing and validation data.
You can tell which fold it was tested on based on the name of the folder "val_fold_[fold_num]". We store the .h5 model that performed best on validation in the "val_fold_[fold_num]" folder.

The inputs to each networks should be [group_inputs, pair_inputs] where \
  group_inputs = np.array of shape (1, max_people, d)\
  pair_inputs = np.array of shape (1, 2, d))

where d is the size of each person's feature vector (typically 4 to represent (x, y, sin(orientation), cos(orientation))

max_people depends on the dataset the model was trained for, but is the number of people in the context representation (e.g. 4 for cocktail_party)
