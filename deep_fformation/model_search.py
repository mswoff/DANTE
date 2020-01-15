import os
import argparse

"""
Run this to find the models with the best validation loss, T=1 F1, and T=2/3 F1 scores
accross the folds. Prints the corresponding test T=2/3 and T=1 scores.

Example usage:
python model_search.py -p models/cocktail_party
"""

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, 
        help="path to directory with models to consider", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    path = args.path + '/pair_predictions_'

    # initialize best models to -inf scores and -1 indexes
    best_val_f1_one = []
    for i in range(5):
        best_val_f1_one.append([float("-inf"), float("-inf"), float('-inf')])
    best_f1_one_idx = [-1] * 5

    best_val_f1_two_thirds = []
    for i in range(5):
        best_val_f1_two_thirds.append([float("-inf"), float("-inf"), float('-inf')])
    best_f1_two_thirds_idx = [-1] * 5

    # search through all model subdirectories
    for i in range(1000):
        if os.path.isdir(path + str(i)):
            tmp = path + str(i)
            # determine which fold we are looking at
            for fold in range(5):
                if os.path.isdir(tmp + '/val_fold_' + str(fold)):
                    name = tmp + '/val_fold_' + str(fold)

                    file = open(name + "/results.txt")
                    lines = file.readlines()
                    beat_val_one = False
                    beat_val_two_thirds = False
                    for line in lines:
                        if "best_val_f1_1" in line:
                            f1 = float(line.split()[-1])
                            if f1 > best_val_f1_one[fold][0]:
                                best_val_f1_one[fold][0] = f1
                                best_f1_one_idx[fold] = i
                                beat_val_one = True
                        if "best_val_f1_2/3" in line:
                            f1 = float(line.split()[-1])
                            if f1 > best_val_f1_two_thirds[fold][0]:
                                best_val_f1_two_thirds[fold][0] = f1
                                best_f1_two_thirds_idx[fold] = i
                                beat_val_two_thirds = True
                        if 'test_f1s' in line and beat_val_one:                            
                            split = line.split()
                            best_f1_one_idx[fold] = i # delete
                            best_val_f1_one[fold][-1] = split[-1]
                            best_val_f1_one[fold][-2] = split[-2]
                            beat_val_one = False
                        if 'test_f1s' in line and beat_val_two_thirds: 
                            split = line.split()
                            best_f1_two_thirds_idx[fold] = i # delete
                            best_val_f1_two_thirds[fold][-1] = split[-1]
                            best_val_f1_two_thirds[fold][-2] = split[-2]
                            beat_val_two_thirds = False

                    file.close()

    print("best val f1 one: ", [tup[0] for tup in best_val_f1_one])
    print("best test f1s: ", [tup[1:] for tup in best_val_f1_one])
    print("best idx: ", best_f1_one_idx)
    print()

    print("best val f1 2/3: ", [tup[0] for tup in best_val_f1_two_thirds])
    print("best test f1s ", [tup[1:] for tup in best_val_f1_two_thirds])
    print("best idx: ", best_f1_two_thirds_idx)
