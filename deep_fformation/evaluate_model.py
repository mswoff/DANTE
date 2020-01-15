import pickle
import argparse
import matplotlib.pyplot as plt

from utils import load_data

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

import numpy as np
import tensorflow as tf
import keras
import os


"""
Makes predictions on the training and test sets using a model and then
saves the results to a .txt file. Can also be used to instead calculate the F1
score on the test set with the --F1 flag.

Example usage:
python evaluate_model.py -k 0 -m models/cocktail_party/pair_predicitons_8 -d cocktail_party
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k_fold', type=str, default='0', 
        help="the fold being considered")
    parser.add_argument('-m', '--model_path', type=str, 
        help="path to the desired model directory (e.g. models/cocktail_party/pair_predictions_1/)", required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True,
        help="which dataset to use (e.g. cocktail_party)")
    parser.add_argument('-f', '--F1', action='store_true', default=False, 
        help="calculates the F1 score on the test set, otherwise saves predictions to an output file") 
    parser.add_argument('--non_reusable', action='store_true', default=False, 
        help="doesn't reuse the same sets in GDSR calc")  

    return parser.parse_args()

# returns a prediction matrix for the training, test, and concatenated data
def build_predictions(model, fold, X_train, X_test):
    fold = int(fold)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    fold_len = test_preds.shape[0]
    test_idx = fold_len * fold

    preds = np.concatenate((train_preds[:test_idx], test_preds, train_preds[test_idx:]))

    return train_preds, test_preds, preds

# saves the predictions to the output file. Preds should be
# of length equal to the entire dataset
def save_predictions(preds, output_file_name, group_lines):
    print(preds.shape, len(group_lines)) # currently not including val data or something
    if preds.shape[0] != len(group_lines):
        throw("ERROR: prediction not for full data")

    print("saving predictions to " + output_file_name)
    output = open(output_file_name, 'w+')
    num_examples, output_len = preds.shape

    for i in range(num_examples):
        line = group_lines[i]
        split = line.split()
        timestamp = split[0]

        output.write(timestamp)
        for j in range(output_len):
            output.write(" ")
            output.write(str(preds[i][j]))

        output.write("\n")
    output.close()


def dump(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    args = get_args()

    test, train, val = load_data("../datasets/" + args.dataset + "/fold_" + str(args.k_fold))
    X, y, timestamps = test
    num_test, _, max_people, d = X[0].shape

    model = keras.models.load_model(args.model_path + "/val_fold_" + str(args.k_fold) 
        + "/best_val_model.h5", custom_objects={'tf':tf , 'max_people':max_people})

    preds = model.predict(X)

    if args.F1: # calculate F1
        from gcdata_importer import import_gc_data, add_time
        if "CoffeeBreak" in args.dataset:
            positions, groups = import_gc_data("CoffeeBreak")
            groups_at_time = add_time(groups)
            from gcdata_importer import F1_calc
            n_people = 14
            n_features = 4
            group_percent = False
        elif "SALSA_all" in args.dataset:
            positions, groups = import_gc_data("SALSA_all")
            groups_at_time = add_time(groups)
            gcdata_importer import F1_calc
            n_people = 18
            n_features = 5
            group_percent = False
        elif "FM_Synth" in args.dataset:
            positions, groups = import_gc_data("FM_Synth")
            groups_at_time = add_time(groups)
            gcdata_importer import F1_calc
            n_people = 10
            n_features = 3
            # calculate GDSR instead of F1
            group_percent = True
        elif "ocktail" in args.dataset:
            from ds_for_valid import create_groups_at_time, F1_calc
            groups_at_time = create_groups_at_time()
            positions = []
        else:
            throw("unrecognized dataset")


        f_2_3, _, _, f_1, _, _ = F1_calc(preds, timestamps, groups_at_time, positions, 
                                    n_people=n_people, thres=1e-5, n_features=n_features, 
                                    group_percent=group_percent, non_reusable=args.non_reusable)
        print(f_2_3, f_1)

    else: # save predictions
        path = args.model_path + "/preds"
        if not os.path.isdir(path):
            os.makedirs(path)

        dump(path + '/preds', preds)
        dump(path + '/timestamps', timestamps)
