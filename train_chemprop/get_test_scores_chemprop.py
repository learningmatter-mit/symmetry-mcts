"""Trains a chemprop model on a dataset by doing domain classification followed by regression with re-weighting."""

import os, sys

filepath = os.path.realpath(__file__)
exepath = os.path.split(os.path.realpath(filepath))[0]
paths = [exepath, "chemprop"]
print(paths)
sys.path.insert(0, os.path.join(*paths))
sys.path.append(os.path.join(*paths))

import chemprop
import argparse
import subprocess
import itertools
import pandas as pd
import numpy as np

import os, csv
import sys
import math
import pprint
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import argparse, glob
import pickle, sys, os, glob, json, re


def predict(arguments):
    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
    return preds


if __name__ == "__main__":
    # retrieve index of training folder
    parser = argparse.ArgumentParser(
        description="driver code for posthoc filtering pipeline"
    )
    parser.add_argument(
        "--sampled_filepath", type=str, help="path to csv file to run inference on"
    )
    # parser.add_argument("--preds_filepath", type=str, help="path to preds output file")
    parser.add_argument(
        "--weights_dir", type=str, help="trained weights folder", required=True
    )
    parser.add_argument(
        "--preds_path", type=str, help="path to save preds", required=True
    ) 

    parser.add_argument
    args = parser.parse_args()

    arguments = [
        "--test_path",
        args.sampled_filepath,
        "--preds_path",
        args.preds_path,
        "--checkpoint_dir",
        args.weights_dir,
        "--uncertainty_method",
        "ensemble",
        "--evaluation_methods",
        "spearman",
        "--evaluation_scores_path",
        "uncertainty_metrics.csv",
    ]

    # if not os.path.isdir(os.path.dirname(args.preds_filepath)):
    #     os.makedirs(os.path.dirname(args.preds_filepath))
    preds = predict(arguments)
    print(preds)

    # df_preds = pd.read_csv(args.preds_filepath)
    df_gt = pd.read_csv(args.sampled_filepath)

    preds_homo = [p[0] for p in preds]
    preds_gap = [p[1] for p in preds]
    preds_lumo = [p[2] for p in preds]

    # # Assuming df_preds and df_gt are already loaded DataFrames
    # # Compute RMSE for each property
    rmse_homo = np.sqrt(mean_squared_error(df_gt['homo'], preds_homo))
    rmse_gap = np.sqrt(mean_squared_error(df_gt['gap'], preds_gap))
    rmse_lumo = np.sqrt(mean_squared_error(df_gt['lumo'], preds_lumo))

    print(f'RMSE for HOMO: {rmse_homo}')
    print(f'RMSE for GAP: {rmse_gap}')
    print(f'RMSE for LUMO: {rmse_lumo}') 
