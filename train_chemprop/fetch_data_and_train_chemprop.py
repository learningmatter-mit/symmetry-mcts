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

# # iter1: patent
# # iter2: patent + ['MCTS_patent_random_sample', 'MCTS_patent_random_sample_with_bridges', 'MCTS_patent_random_sample_with_bridges_kevin','MCTS_patent_random_sample_with_bridges_akshays', 'MCTS_patent_random_sample_with_bridges_akshaye']
# # iter3: patent + ['MCTS_patent_random_sample', 'MCTS_patent_random_sample_with_bridges', 'MCTS_patent_random_sample_with_bridges_kevin','MCTS_patent_random_sample_with_bridges_akshays', 'MCTS_patent_random_sample_with_bridges_akshaye'] + ['num_atoms_cutoff_fragments_AL_iter1', 'num_atoms_cutoff_fragments_AL_EI_iter1_500', 'best_EI', 'lowest_bandgap']

# molsets = ['MCTS_patent_random_sample', 'MCTS_patent_random_sample_with_bridges', 'MCTS_patent_random_sample_with_bridges_kevin',
#            'MCTS_patent_random_sample_with_bridges_akshays', 'MCTS_patent_random_sample_with_bridges_akshaye', 'num_atoms_cutoff_fragments_AL_iter1', 'num_atoms_cutoff_fragments_AL_EI_iter1_500', 'best_EI', 'lowest_bandgap', 'diverse_lowest_bandgap']

# combined_df = pd.DataFrame(columns=['smiles', 'homo', 'gap', 'lumo'])
# for ms in molsets:
#     results = Calc.objects.filter(species__group__name='opd',
#                                     parentjob__config__name='wb97xd_def2svpd_tda_tddft_orca',
#                                     method__name='tddft_tda_hyb_wb97xd3',
#                                     jacobian__isnull=True,
#                                     species__mol__sets__name=ms).values_list('species__smiles',
#                                                                             'props__homo',
#                                                                             'props__excitedstates',
#                                                                             'props__lumo')

#     results = [[data if i != 2 else data[0]['energy'] for i, data in enumerate(result)] for result in results]

#     df = pd.DataFrame(results, columns=['smiles', 'homo', 'gap', 'lumo'])
#     print(ms, len(df))

#     combined_df = pd.concat([combined_df, df], ignore_index=True)

# combined_df = combined_df.sample(frac=1)
# # Drop duplicates based on the 'smiles' column
# combined_df = combined_df.drop_duplicates(subset=['smiles'])
# # Reset the index after dropping duplicates
# combined_df = combined_df.reset_index(drop=True)

# # Prepare combined train test dataset
# df_patent = pd.read_csv('patent_opd_labeled.csv').sample(frac=1)
# df_test = combined_df.tail(50)
# df_val = combined_df.iloc[-100:-50]
# df_train = pd.concat([df_patent, combined_df.iloc[:-100]], ignore_index=True, axis=0).sample(frac=1)
# df_train.to_csv('patent_MCTS_frag_decomp_train_AL_iter3.csv', index=False)
# df_val.to_csv('patent_MCTS_frag_decomp_val_AL_iter3.csv', index=False)
# df_test.to_csv('patent_MCTS_frag_decomp_test_AL_iter3.csv', index=False)

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
    "patent_MCTS_frag_decomp_train_AL_iter3.csv",
    "--separate_val_path",
    "patent_MCTS_frag_decomp_val_AL_iter3.csv",
    "--separate_test_path",
    "patent_MCTS_frag_decomp_test_AL_iter3.csv",
    "--target_columns",
    "homo",
    "gap",
    "lumo",
    "--dataset_type",
    "regression",
    "--save_dir",
    "chemprop_weights_frag_decomp_AL_iter3_ensemble",
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
