import os, sys
import chemprop

from utils import set_all_seeds

exepath = "~/experiments"

paths = [exepath, "chemprop"]
print(paths)
sys.path.insert(0, exepath)
set_all_seeds(9999)


arguments = [
    "--data_path",
    "data/patent_AL_iter_2_train.csv",
    "--separate_val_path",
    "data/patent_AL_iter_2_val.csv",
    "--separate_test_path",
    "data/patent_AL_iter_2_test.csv",
    "--target_columns",
    "homo",
    "gap",
    "lumo",
    "--dataset_type",
    "regression",
    "--save_dir",
    "checkpoints/chemprop_weights_patent_AL_iter_2_ensemble",
    "--ensemble_size",
    "5",
    "--epochs",
    "500",
    "--save_preds",
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(
    args=args, train_func=chemprop.train.run_training
)
