import numpy as np
import pickle
import os
import shutil

import argparse

"""
Converts the datasets from .txt files into pickle objects which will be used by the models.
Splits the data into folds sequentially, with the validation data sandwiching the test fold.
Use --augment_synthetic to add the synthetic data to the training set.

Example usage:
python build_dataset.py -p cocktail_party
"""

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', required=True,
        help="where to load and save from (e.g. if calling from 'datasets', can use 'cocktail_party'")
    parser.add_argument('-s', '--augment_synthetic', help="adds synthetic data as well", action="store_true")
    # synthetic must have be made to have the same max number of people as the original dataset
    return parser.parse_args()

# builds X_group tensor with (num_examples, 1, max_people, d) dimensions
# uses "people_lines" (of length num_examples) to read in each example
# each person is represented by (x, y, sine(theta), cosing(theta)).
# Also outputs X_pairs tensor of shape (num_examples, 1, 2, d)
def build_X(people_lines, max_people, d, convert_units=False):
    num_examples = len(people_lines)
    X_group = np.zeros((num_examples, 1, max_people, d))
    X_pairs = np.zeros((num_examples, 1, 2, d))

    for i in range(num_examples):
        line = people_lines[i]
        if type(line) == str:
	    split = line.split()
	else:
	    split = line
        timestamp = split[0]

        # Build X
        j = 1
        while j < len(split):
            representation = split[j:j+d]
            vect = np.array([float(x) for x in representation])
            person_idx = int((j-1)/d)

            if convert_units:
		# "ocktail" not in args.path and "SALSA" not in args.path and "FM" not in args.path:
                # convert to meters from cm
                vect[:2] /= 100
            if  j < len(split) - 2 * d:
                X_group[i, 0, person_idx, :] = vect
            else:
                X_pairs[i, 0, person_idx - max_people, :] = vect

            j += d

    return X_group, X_pairs


# builds Y tensor with (num_examples, output_size) dimensions
# uses "group_lines" (of length num_examples) to read in each
# list of affinities. Also returns the timestamps as a list of strings
def build_Y(group_lines):
    output_size = len(group_lines[0].split()) - 1
    num_examples = len(group_lines)

    Y = np.zeros((num_examples, output_size))
    timestamps = []

    for i in range(num_examples):
        line = group_lines[i]
        split = line.split()
        timestamp = split[0]
        timestamps.append(timestamp)

        vect = np.asarray(split[1:], dtype=np.float32)
        Y[i] = vect
    return Y, timestamps

# splits the dataset into training, validation, and testing set
# based on the indexes provided
def split_test_train_val(start_test, end_test, val_start_ends, data):
    val = []
    for start, end in val_start_ends:
        val.append(data[start:end])
    val = np.concatenate(val)

    test = data[start_test:end_test]
    start = min([start for start,end in val_start_ends] + [start_test])
    end = max([end for start,end in val_start_ends] + [end_test])
    train = np.concatenate((data[:start], data[end:]))

    return test, train, val

def diffs_from_start(idxs):
    start = idxs[0]
    diffs = [idx - start for idx in idxs]
    return diffs

def get_start_end_timechange(fold, num_folds, X,  path):
    timechanges = open(path + '/timechanges.txt', 'r')
    timechanges = timechanges.readlines()
    timechanges = [int(val) for val in timechanges]

    timechanges.append(X.shape[0])
    num_times = len(timechanges) - 1

    start_test_idx = int(num_times / num_folds * fold)
    end_test_idx = int(num_times / num_folds * (fold+1))

    val_fold_idx_diff = int(num_times / num_folds / 2)

    start_test = timechanges[start_test_idx]
    end_test = timechanges[end_test_idx]

    # one fold before test
    if start_test == 0:
        # creates larger fold after test
        val_start = end_test
        val_end_idx = end_test_idx + 2 * val_fold_idx_diff
        val_end = timechanges[val_end_idx]
        val_start_ends = [(val_start, val_end)]

    elif fold == num_folds - 1:
        # creates larger fold before test
        val_start_idx = start_test_idx - 2 * val_fold_idx_diff
        val_start = timechanges[val_start_idx]
        val_end = start_test
        val_start_ends = [(val_start, val_end)]

    else:
        # creates fold before test
        val_start_idx = start_test_idx - val_fold_idx_diff
        val_start = timechanges[val_start_idx]
        val_end = start_test
        val_start_ends = [(val_start, val_end)]

        # creates fold after test
        val_start = end_test
        val_end_idx = end_test_idx + val_fold_idx_diff
        val_end = timechanges[val_end_idx]
        val_start_ends.append((val_start, val_end))


    return start_test, end_test, val_start_ends

# repeats num_add of the group people
def repeat_people(people_lines, num_add):
    new_lines = []
    for line in people_lines:
        split = line.split()
        timestamp = split[0]
        new_split = []
        new_split.append(timestamp)
        new_split += split[1:1+4*num_add]
        new_split += split[1:]
        new_lines.append(" ".join(new_split))
    return new_lines

def augment_synthetic(X_group, X_pairs, Y, max_people, d):
    path = "Synth"
    people_file = open(os.path.join(path, 'coordinates.txt'), 'r')
        
    people_lines = people_file.readlines()

    # hack for salsa
    if 'SALSA' in args.path:
        people_lines = repeat_people(people_lines, 2)
    
    convert_units = ("ocktail" not in args.path and "SALSA" not in args.path and "FM" not in args.path)
    X_group_aug, X_pairs_aug = build_X(people_lines, max_people, d, convert_units=convert_units)

    group_file = open(os.path.join(path, 'affinities.txt'), 'r')
    group_lines = group_file.readlines()

    Y_aug, _ = build_Y(group_lines)

    X_group = np.concatenate([X_group_aug, X_group], axis=0)
    X_pairs = np.concatenate([X_pairs_aug, X_pairs], axis=0)
    Y = np.concatenate([Y_aug, Y], axis=0)

    return X_group, X_pairs, Y


def dump(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)



if __name__ == "__main__":
    args = get_args()

    num_folds = 5
    path = args.path

    # Build X
    people_file = open(os.path.join(path, 'coordinates.txt'), 'r')
    if "SALSA" in args.path:
        d = 6 # uses both head and body orientations
    else:
        d = 4
        
    people_lines = people_file.readlines()

    # hack for salsa to get synth
    if args.augment_synthetic and "SALSA" in args.path:
        new_lines = []
        for line in people_lines:
            split = line.split()
            timestamp = split[0]
            new_split = []
            new_split.append(timestamp)
            i = 1
            while i < len(split):
                new_split += split[i:i+4]
                i += 6
            new_lines.append(" ".join(new_split))
        people_lines = new_lines
        d = 4

    # hack for Friends Meet to get synth
    if 'FM' in args.path and args.augment_synthetic:
        people_lines = repeat_people(people_lines, 4)

    # subtract 2 b/c people of interest not included in context
    max_people =  int((len(people_lines[0].split()) - 1) / d) - 2

    X_group, X_pairs = build_X(people_lines, max_people, d)

    # Build Y
    group_file = open(os.path.join(path, 'affinities.txt'), 'r')
    group_lines = group_file.readlines()

    Y, timestamps = build_Y(group_lines)

    total_len = len(group_lines)
    fold_size = int(total_len / num_folds)
    val_fold_sz = int(fold_size / 2)

    for k in range(num_folds):
        start_test = k * fold_size
        end_test = (k+1) * fold_size

        val_start_ends = []

        # one fold before test
        if start_test == 0:
            # creates fold after next fold
            val_start = end_test + val_fold_sz
            val_end = end_test + 2 * val_fold_sz
        else:
            val_start = start_test - val_fold_sz
            val_end = start_test
        val_start_ends.append((val_start, val_end))

        # one fold after test
        if k == num_folds - 1:
            # creates fold before previous fold
            val_start = start_test - 2 * val_fold_sz
            val_end = start_test - val_fold_sz
        else:
            val_start = end_test
            val_end = end_test + val_fold_sz
        val_start_ends.append((val_start, val_end))

        if os.path.isfile(path + '/timechanges.txt'):
        # adjusts for different numbers of people per frame
            start_test, end_test, val_start_ends = get_start_end_timechange(k, num_folds, X_group, path)

        X_group_test, X_group_train, X_group_val = split_test_train_val(start_test, end_test, val_start_ends, X_group)
        X_pairs_test, X_pairs_train, X_pairs_val = split_test_train_val(start_test, end_test, val_start_ends, X_pairs)
        Y_test, Y_train, Y_val = split_test_train_val(start_test, end_test, val_start_ends, Y)
        timestamps_test, timestamps_train, timestamps_val = split_test_train_val(start_test, end_test, val_start_ends, timestamps)

        temp_path = args.path

        if args.augment_synthetic:
            if temp_path[-1] == '/':
                temp_path = temp_path[:-1]
            temp_path += "_synth"
            if not os.path.isdir(temp_path):
                os.makedirs(temp_path)
                ds_path = args.path + '/DS_utils'
                if os.path.isdir(ds_path):
                    shutil.copytree(ds_path, temp_path + '/DS_utils')

            X_group_train, X_pairs_train, Y_train = augment_synthetic(
                X_group_train, X_pairs_train, Y_train, max_people, d)

        temp_path += '/fold_' + str(k)
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)

        dump(temp_path + '/test.p', ([X_group_test, X_pairs_test], Y_test, timestamps_test))
        dump(temp_path + '/train.p', ([X_group_train, X_pairs_train], Y_train, timestamps_train))
        dump(temp_path + '/val.p', ([X_group_val, X_pairs_val], Y_val, timestamps_val))
                    

        
