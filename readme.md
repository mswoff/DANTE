# How to Implement

Please use python 3.7 and install the requirements.txt file.

### Overview
The "datasets" folder contains both the various group detection datasets as well as functions for reformatting the data to be accepted as an input by the model.

The "deep_fformation" folder contains functions to train and evaluate a the model.

Pre-trained models are stored in "deep_fformation/models".

# Steps
(1-3 have already been done for the publicly available datasets)
1. In the datsets folder, create a folder for a given dataset (e.g. CocktailParty)
2. Under the dataset, create a subdirectory called DS_utils (e.g. CocktailParty/DS_utils)
3. In the DS_utils subdirectory, put in the properly formatted features.txt and group_names.txt files (see [Data File Formats](#data-file-formats) for details). These should already be there for the datasets included.
4. Run reformat_data.py with the proper flags for your dataset. (e.g. reformat_data.py -d cocktail_party). See [reformat_data.py](#data-building-scripts) for more details on flags. This should generate "coordinates.txt" "affinities.txt" and "timechanges.txt"
5. Run build_dataset.py in order to convert the data from text datasets to pickle arrays for each fold
6. Run run_models.py to automatically train many models
7. Run model_search.py to find the best model per fold according to the validation data

Steps 1-3 should already be completed for the datasets in this folder. Step 3 might need to be redone for features.txt to append more dummy/fake people to each row, for example to train on multiple datasets and keep the matrix dimensions consistent. See [features.txt](#data-file-formats) for more details

# Datasets
One of the most difficult practical aspects of the group detection is the variety of datasets that have been developed. Many of the raw datasets are formatted slightly differently and even have often slightly different features (some don't contain angles, some contain both head and body angles, etc).

An additional complication of training a NN on such data is that often not all scenes accross datasets or within a given dataset have the same # of people, making matrix formulations of the data more difficult. The datasets folder contains functions to assist with formatting the data for our model.

## Data Building Scripts

### reformat_data.py
This file contains useful functions to build the coordinates.txt, affinities.txt, and timechanges.txt files used by "build_dataset.py". It takes as input data files "group_names.txt" and "features.txt".

We have found that first converting the data to the form in "group_names.txt" and "features.txt" and then using the functions in reformat_data.py was simpler than attempting to build the matrices in "coordinates.txt" and "affinities.txt" from scratch. Often the data comes in a form similar to group_names.txt or features.txt

Flags:\
-d --dataset (required) Name of the dataset being used \
-p --n_people (optional, default = 15) Number of people per row in features.txt (also number of rows used in the network). Must correspond to the number of people per row in features.txt\
-f --n_features (default = 4) Number of features per person (including name), before reformatting. In the standard case (like cocktail party, this is 4 (name, x, y, theta) \
-a --n_aug_features (default = 4) Number of features per person (not including name), after reforamtting. In the standard case (like cocktail party, this is 4 (x, y, cos(theta), sin(theta))\
-v --use_velocities (default = False) Whether or not to use the velocities as a feature (as crude angle estimates or otherwise)  

Examples: 
Note: For the most part, you should be using the same numbers with each dataset that I use. In order to do synthetic augmentation of the cocktail party dataset (which we will get to shortly), you must use CocktailParty14, which is augmented with dummy people to have 14 people per row. If you do this, than -p is now 14 instead of 6.

(standard case) (CocktailParty/CoffeeBreak with only 1 angle (either head or body orientation))
(We set the max number of people depending on how we constructed features.txt. In this case, since the CocktailParty dataset always has 6 people, and we aren't combining it with other datasets during training, we didn't add dummy (fake) people to the features.txt file and had just 6 real people per row)
```shell
python reformat_data.py -d cocktail_party -p 6
```
or for example, CoffeeBreak with 14 people
```shell
python reformat_data.py -d CoffeeBreak -p 14
```
or the Synthetic dataset with 14 people, some of whom are dummy in features.txt

```shell
python reformat_data.py -d Synth -p 14
```

(non-standard case, meaning features other than x,y,theta) (Friends Meet (no angles, use velocities, and max of 10 people per row))
```shell
python reformat_data.py -d FM_Synth -p 10 -f 3 -a 4 -v True
```

(non-standard case) (Salsa file with head and body angles, and there are up to 18 people in each frame)
```shell
python reformat_data.py -d SALSA_all -p 18 -f 5 -a 6
```

The non-standard cases of velocities and 2 angles are covered in the function reformat_data.py/shift_all_data_nonstandard.py.

With other non-standard feature inputs one may need to use additional flags and even edit this function.


### shift_data.py
This file contains helper functions for "reformat_data.py" used in centering people around 2 people i and j, and also for flipping the frames over the horizontal axis.

### build_dataset.py
This file builds the X, Y tensors for use in training, validation, and testing. It also splits the data into folds. For a given dataset, we must already have a "coordinates.txt" file, a "timechanges.txt" file, and an "affinities.txt" file. This will create directories for the folds and fill them with "train.p", "val.p", and "test.p".

Flags:\
-p --path (required): path to the dataset being used\
-s --augment_synthetic (optional): Augments your training data with the synthetic data

```shell
python build_dataset.py -p [path_to_dataset]
```

Example for cocktail party and no augmentation (if called from datasets directory):
```shell
python build_dataset.py -p cocktail_party
```

### run_models.py 
This script automates the model training process. Simply run the script and it will train and save many models.

Flags:\
-d --dataset (required): Which datset to use. The name should match the name of the folder exactly.\
-p --no_pointnet (optional): Removes the global context vector.\
-s --symmetric (optional): Forces the network to make symmetric predictions for each pair of two people. Does this by making the Dyad use a symmetric (max) function.\
-e --epochs (optional): Max number of epochs to run for. Early stopping may halt training before this is reached.


```shell
python run_models.py -d [dataset_name]
```

Example for cocktail party.
```shell
python run_models.py -d cocktail_party
```

### model_search.py
After training many models, this script will find the best one for each fold based on validation metrics.

The script will output 2 sections of text. The sections report results based on different metrics (F1 with T=1 and F1 with T=2/3). The outputs correspond to the best validation results for the different metrics, on a fold-by-fold basis. The first line in the section is the metric's result on the validation data. The second line is both the T=2/3 and T=1 F1 result on the test data. And the final line in the folder index which corresponds to that model. 

Note: if there are no results for a given fold, the results will be -inf and -1.


For example, if we only trained models on the first 3 folds, the output might be:

best val f1 one:  [0.453125, 0.3201219512195122, 0.40625, -inf, -inf]\
best test f1s:  [['0.7135321624087592', '0.3730366492146597'], ['0.7869882133995038', '0.4296875'], ['0.7340425531914894', '0.43736049107142855'], [-inf, -inf], [-inf, -inf]]\
best idx:  [2, 6, 4, -1, -1]

best val f1 2/3:  [0.7421669407894737, 0.6188387223974764, 0.6006493506493507, -inf, -inf]\
best test f1s  [['0.7154657643312102', '0.3730366492146597'], ['0.7869882133995038', '0.4296875'], ['0.7340425531914894', '0.43736049107142855'], [-inf, -inf], [-inf, -inf]]\
best idx:  [5, 3, 4, -1, -1]


Looking at the first metric (T=1), we see that the best models for folds 1, 2, and 3 can be found in pair_predictions_2, pair_predictions_6, and pair_predictions_4, respectively. Examining fold 1, we see that the best model achieved a T=1 F1 score of .71 on the validation data. We can now look at the second line to see how that model performed on the test data. We see it had a T=1 F1 score of .44 and T=2/3 F1 score of .73 on the test data for that fold.

Flags:\
-p --path (required): Path to directory with models to compare


```shell
python model_search.py -p [path_to_models_directory]
```

Example for cocktail_party
```shell
python model_search.py -p models/cocktail_party
```

### run_baseline_models.py
This file is used to run the basic ML algorithms described in "Recognizing F-Formations in the Open World" (https://ieeexplore.ieee.org/abstract/document/8673233).

Flags:\
-d --dataset (required): Which datset to use. The name should match the name of the folder exactly.\
-n --naive_grouping (optional): If set, uses the grouping algorithm described in the paper. If not set, we use Dominant Sets for the clustering.

Example for cocktail_party
```shell
python run_baseline_models.py -d cocktail_party -n
```


## Data File Formats

#### coordinates.txt
Each row of this file contains the coordinates of every person in the room, shifted relative to the room centered between two people, i and j, for a certain timestamp and vertical orientation (000 for normal, 001 for orientation flipped vertically). The rows are indicated in the form time:i:j:orientation.

e.g. 123153:0:1:001 ID003 1.4 3.4 1.3 ID004 -3.3 4.1 0.0 ... ID001 3.4 0.0 0.1 ID002 -3.4 0.0 .2

Note that in this file i:j are 0 indexed, and also that the last two people in a row are i and j (ID00[i+1] ID00[j+1]). The rows should have all examples of one time in order for all ij permutations, and then the same time flipped vertically, and then moving on to the next time.

e.g (for 6 people)  
123153:0:1:000 ID003 x y .. ID004 x y .. ... ID001 x y .. ID002 x y ..  
123153:0:2:000 ID002 x y .. ID004 x y .. ... ID001 x y .. ID003 x y ..  
...  
123153:5:4:000 ID001 x y .. ID002 x y .. ... ID006 x y .. ID005 x y ..  
123153:0:1:001 ID003 x y .. ID004 x y .. ... ID001 x y .. ID002 x y ..  
...  
123153:5:4:001 ID001 x y .. ID002 x y .. ... ID006 x y .. ID005 x y ..  
123154:0:1:000 ID003 x y .. ID004 x y .. ... ID001 x y .. ID002 x y ..  

#### affinities.txt
Each row of this file contains the ground truth affinity for people i and j at a given time (0 if they are in a group, 1 if not)

e.g. 123412:3:4:001 0        (this would mean ID004 and ID005 are not in a group)

The times should follow the same order as coordinates.txt

#### timechanges.txt
A file which contains the indicies of the rows of affinities.txt or coordinates.txt where the time changed (i.e. new frame). Using this file allows for easier appropriate splitting of the data into train/test without splitting in the middle of a timestamp.

---
DSutils

Note: features and group_names are modeled almost exactly after the raw data format of CocktailParty and CoffeeBreak. Getting other data into this format is up to the user.

#### features.txt
A file with basic feature information. The standard case is that the features are simply x, y, theta. Each row should contain a timestamp followed by ID feature1 feature2 etc. We denote the name of a person by ID00[number]. We ended up not removing 0s for larger numbers, so person 12 is ID0012

e.g. 432134.1234 ID001 12 13 .4 ID002 49 -3 .1 ID003 34 3 4

One complication is that different datasets may have different amounts of features. For example, if there are two angles, the features would be x y theta_head theta_body. If there are no angles they would simply be x y. This is okay, it just means that later functions will need additional flags set.

This is often very close to the format of most raw datsets. One addition constraint we put on features.txt for ease of matrix creation is the condition that the rows must all be of the same length. Thus, if there are fewer people in a frame, they are filled with dummy data indicated by the word "fake". 

e.g. If there are a max of 6 people in our network, but only 4 in this frame, and also let's say in this case there are no angles, just x y coordinates:
432134.1234 ID001 3 4 ID002 -3 3 ID003 31 43 ID004 93 -1 fake fake fake fake fake fake

#### group_names.txt
A file that contains ground truth groups in each timestamp, indicated by a row. Each row should look as follows

e.g. (person ID003 is in no group) 432134.1234 < ID001 ID004 > < ID003 > < ID002 ID005 ID006 >

