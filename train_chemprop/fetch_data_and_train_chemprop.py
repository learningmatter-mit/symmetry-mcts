# import os
# import django

# import sys

# sys.path.insert(0, "/home/gridsan/sakshay/experiments/htvs")
# sys.path.insert(0, '/home/gridsan/sakshay/experiments/htvs/djangochem')

# os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.orgel"
# # os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.toy"

# os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"


# django.setup()

# # Shell Plus Model Imports
# from features.models import AtomDescriptor, BondDescriptor, ConnectivityMatrix, DistanceMatrix, ProximityMatrix, SpeciesDescriptor, TrainingSet, Transformation
# from guardian.models import GroupObjectPermission, UserObjectPermission
# from django.contrib.contenttypes.models import ContentType
# from neuralnet.models import ActiveLearningLoop, NetArchitecture, NetCommunity, NetFamily, NeuralNetwork, NnPotential, NnPotentialStats
# from jobs.models import Job, JobConfig, WorkBatch
# from django.contrib.admin.models import LogEntry
# from django.contrib.auth.models import Group, Permission, User
# from django.contrib.sessions.models import Session
# from pgmols.models import (AtomBasis, BasisSet, Batch, Calc, Cluster,
#                            Geom, Hessian, Jacobian, MDFrame, Mechanism, Method, Mol, MolGroupObjectPermission,
#                            MolSet, MolUserObjectPermission, PathImage, ProductLink, ReactantLink, Reaction,
#                            ReactionPath, ReactionType, SinglePoint, Species, Stoichiometry, Trajectory)
# import experiments
# # Shell Plus Django Imports
# from django.core.cache import cache
# from django.db import transaction
# from django.utils import timezone
# from django.contrib.auth import get_user_model
# from django.urls import reverse
# from django.conf import settings
# from django.db.models import Avg, Case, Count, F, Max, Min, Prefetch, Q, Sum, When, Exists, OuterRef, Subquery

# from experiments.models import Substance
# import pandas as pd
# from rdkit.Chem import AllChem, Descriptors
# import numpy as np
# from pathlib import Path

# import os, sys
# ms = 'MCTS_y6_random_sample_v3'

# if not Path(ms + '.csv').is_file():

#     # for ms in molsets:
#     results = Calc.objects.filter(species__group__name='opd',
#                                     parentjob__config__name='wb97xd_def2svpd_tda_tddft_orca',
#                                     method__name='tddft_tda_hyb_wb97xd3',
#                                     jacobian__isnull=True,
#                                     species__mol__sets__name=ms).values_list('species__smiles',
#                                                                             'props__homo',
#                                                                             'props__excitedstates',
#                                                                             'props__lumo')

#     # This takes the value of the SMILES, HOMO, and LUMO as is, and extracts the value of the optical gap
#     # (lambda_max) from props__excitedstates. This takes the "reddest peak", or the vertical excitation energy
#     # with the lowest energy (does not take into account oscillator strength)
#     results = [[data if i != 2 else data[0]['energy'] for i, data in enumerate(result)] for result in results]

#     df = pd.DataFrame(results, columns=['smiles','homo','gap','lumo'])


#     df.to_csv(ms + '.csv', index=False)

# # Prepare combined train test dataset
# df_patent = pd.read_csv('patent_opd_labeled.csv').sample(frac=1)
# df_mcts = pd.read_csv('MCTS_y6_random_sample_v3.csv').sample(frac=1)
# df_test = df_mcts.iloc[-50:].sample(frac=1)
# df_val = df_mcts.iloc[-100:-50].sample(frac=1)
# df_train = pd.concat([df_patent, df_mcts.head(len(df_mcts) - 100)], ignore_index=True, axis=0).sample(frac=1)
# df_train.to_csv('patent_MCTS_train.csv', index=False)
# df_val.to_csv('patent_MCTS_val.csv', index=False)
# df_test.to_csv('patent_MCTS_test.csv', index=False)

import os, sys
import chemprop

# from chemprop_inference import predict
# from chemprop.predict_one import predict_one
from utils import set_all_seeds

# filepath = os.path.realpath(__file__)
# exepath = os.path.split(os.path.realpath(filepath))[0]
exepath = "~/experiments"

paths = [exepath, "chemprop"]
print(paths)
sys.path.insert(0, exepath)
# sys.path.insert(0, os.path.join(*paths))
# sys.path.append(os.path.join(*paths))
set_all_seeds(9999)


arguments = [
    "--data_path",
    "patent_MCTS_train.csv",
    "--separate_val_path",
    "patent_MCTS_val.csv",
    "--separate_test_path",
    "patent_MCTS_test.csv",
    "--target_columns",
    "homo",
    "gap",
    "lumo",
    "--dataset_type",
    "regression",
    "--save_dir",
    "chemprop_weights",
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


# arguments = [
#     '--test_path', 'chemprop_patent+mcts_train_smiles.csv',
#     '--preds_path', 'chemprop_patent+mcts_train_smiles_preds.csv',
#     '--uncertainty_method', 'ensemble',
#     '--checkpoint_dir', 'models/patent_MCTS_checkpoints_ensemble'
# ]

# # arguments = [
# #     '--test_path', '/dev/null',
# #     '--preds_path', '/dev/null',
# #     '--uncertainty_method', 'ensemble',
# #     '--checkpoint_dir', 'models/patent_MCTS_checkpoints_ensemble',
# # ]

# # smiles = [['Cc1c(-c2ccc(-c3ccc(C=CC=CC=C4C(=O)c5cc(Br)c(Cl)cc5C4=C(C#N)C#N)c4nsnc34)s2)sc2c1oc1c2sc2c3sc4c(oc5c(C)c(-c6ccc(-c7ccc(C=CC=CC=C8C(=O)c9cc(Br)c(Cl)cc9C8=C(C#N)C#N)c8nsnc78)s6)sc54)c3c(F)c(F)c12']]
# args = chemprop.args.PredictArgs().parse_args(arguments)
# model_objects = chemprop.train.load_model(args=args)
# print(chemprop.train.make_predictions(args=args, model_objects=model_objects, return_uncertainty=True))
# # print(chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects, return_uncertainty=True))
# # print(predict_one('models/patent_MCTS_checkpoints', [['Cc1c(-c2ccc(-c3ccc(C=CC=CC=C4C(=O)c5cc(Br)c(Cl)cc5C4=C(C#N)C#N)c4nsnc34)s2)sc2c1oc1c2sc2c3sc4c(oc5c(C)c(-c6ccc(-c7ccc(C=CC=CC=C8C(=O)c9cc(Br)c(Cl)cc9C8=C(C#N)C#N)c8nsnc78)s6)sc54)c3c(F)c(F)c12']]))
# # print(predict_one('models/patent_MCTS_checkpoints', [['Cc1c(-c2ccc(C=C3C(=O)c4c(csc4Cl)C3=C(C#N)C#N)c3nsnc23)n(C)c2c1oc1c2sc2c3sc4c(oc5c(C)c(-c6ccc(C=C7C(=O)c8c(csc8Cl)C7=C(C#N)C#N)c7nsnc67)n(C)c54)c3c3nccnc3c12']]))
