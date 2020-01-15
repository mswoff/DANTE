import numpy as np
from collections import defaultdict

from utils import load_data

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

import argparse

from F1_calc import F1_calc
from reformat_data import add_time, import_data




# Trains and evaluates models from "Recognizing F-Formations in the Open World"
# https://ieeexplore.ieee.org/abstract/document/8673233


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help="which dataset to use", required=True)
    parser.add_argument('-n', '--naive_grouping', action="store_true", 
        help="uses the original grouping algorithm from the paper instead of dominant sets")
    return parser.parse_args()


# changes from (x, y, sin(theta), cosine(theta)) to (dist, effort angle)
def convert_to_effort_angle(X_pairs):

    X_new = np.zeros((X_pairs.shape[0], 2))

    for i in range(X_pairs.shape[0]):
        p1 = X_pairs[i, 0, 0]
        p2 = X_pairs[i, 0, 1]

        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # convert from sin and cos to angle
        s1 = np.arcsin(p1[2])
        c1 = np.arccos(p1[3])
        s2 = np.arcsin(p2[2])
        c2 = np.arccos(p2[3])

        effort_angle = np.abs(s1 - s2) + np.abs(c1 - c2)
        effort_angle *= 2 # to make it between 0-2pi

        X_new[i, 0] = dist
        X_new[i, 1] = effort_angle

    return X_new

def get_learner(name, val):
    if name in "logistic regression":
        model = learner(C=val, solver="lbfgs")
    elif name in "Weighted KNN":
        model = learner(n_neighbors=val, weights="distance")
    elif name in "Bagged Trees":
        model = learner(n_estimators=val)
    return model

def get_results(preds, timestamps, groups_at_time, positions, n_people, n_features, naive):
    return F1_calc(2/3, preds, timestamps, groups_at_time, positions,
        n_people, 1e-5, n_features, dominant_sets=not naive)[0], F1_calc(1, preds, timestamps, groups_at_time, positions,
        n_people, 1e-5, n_features, dominant_sets=not naive)[0]



if __name__ == "__main__":
    args = get_args()
    
    folds = [0, 1, 2, 3, 4]
    classifiers = {"logistic regression": LogisticRegression, 
        "Weighted KNN": KNeighborsClassifier,
        "Bagged Trees" : BaggingClassifier}
    normalize = False
    epochs = 600
    val_folds = 1
    model_str = "two_people_at_a_time"
    f_1_results = defaultdict(list)
    f_2_3_results = defaultdict(list)

    if "CoffeeBreak" in args.dataset:
        positions, groups = import_data("CoffeeBreak")
        groups_at_time = add_time(groups)
        n_people = 14
        n_features = 4

    elif "SALSA_all" in args.dataset:
        positions, groups = import_data("SALSA_all")
        groups_at_time = add_time(groups)
        n_people = 18
        n_features = 5

    elif "cocktail_party" in args.dataset:
        positions, groups = import_data("cocktail_party")
        groups_at_time = add_time(groups)
        n_people = 6
        n_features = 4

    else:
        throw("unknown dataset")



    params = {"logistic regression": ("C", [10**v for v in range(-4, 8)]),
        "Weighted KNN": ("n_neighbors", range(1, 10)), 
        "Bagged Trees" : ("n_estimators", range(5, 40, 3))}

    for fold in folds:
        # load and convert data into (dists, angle) form
        test, train, val = load_data("../datasets/" + args.dataset + "/fold_" + str(fold))

        [_, X_pairs_train], Y_train, timestamps_train = train
        X_train = convert_to_effort_angle(X_pairs_train)

        [_, X_pairs_val], Y_val, timestamps_val = val
        X_val = convert_to_effort_angle(X_pairs_val)

        [_, X_pairs_test], Y_test, timestamps_test = test
        X_test = convert_to_effort_angle(X_pairs_test)

        for name, learner in classifiers.items():

            param, vals = params[name]
            best_f_1 = []
            best_f_2_3 = []
            for val in vals:
                model = get_learner(name, val)

                model.fit(X_train, Y_train.flatten())

                preds = model.predict_proba(X_val)
                preds = preds[:, 1] # preds has prob 0 and prob 1, we only want prob 1

                f_2_3, f_1 = get_results(preds, timestamps_val, groups_at_time, positions, n_people, n_features, args.naive_grouping)

                best_f_1.append(f_1)
                best_f_2_3.append(f_2_3)
            
            # run best hyper params
            best_f_1_param = vals[np.argmax(best_f_1)]
            model = get_learner(name, best_f_1_param)
            model.fit(X_train, Y_train.flatten())
            preds = model.predict_proba(X_test)
            preds = preds[:, 1]

            f_2_3, f_1 = get_results(preds, timestamps_test, groups_at_time, positions, n_people, n_features, args.naive_grouping)

            f_1_results[name].append(f_1)
            f_2_3_results[name].append(f_2_3)


    print(f_1_results)
    print()
    print(f_2_3_results)










