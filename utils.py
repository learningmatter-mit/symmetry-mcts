import math
import os
import re
import json
import random
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

# from MCTS_y6 import normalization_params


def set_all_seeds(seed):
    random.seed(seed)
    #   os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def find_isotope_mass_from_string(smi):
    return [int(mass) for mass in re.findall(r"\[(\d+)He\]", smi)]


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def compute_molecular_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.ExactMolWt(mol)


def get_num_atoms(smiles):
    return Chem.MolFromSmiles(smiles).GetNumAtoms()


def get_num_atoms_by_id(smiles, id):
    num_carbons = 0
    for atom in Chem.MolFromSmiles(smiles).GetAtoms():
        if atom.GetAtomicNum() == id:
            num_carbons += 1
    return num_carbons


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_identity_reward(reduction):
    if reduction == "sum":
        return 0
    elif reduction == "product":
        return 1
    elif reduction == "min":
        return float("inf")


# def get_unnorm_rewards(gap_reward_norm, sim_reward_norm):
#     gap_reward_unnorm = gap_reward_norm * normalization_params['bandgap']['std_dev'] + normalization_params['bandgap']['mean']
#     sim_reward_unnorm = sim_reward_norm * normalization_params['similarity']['std_dev'] + normalization_params['similarity']['mean']
#     return gap_reward_unnorm, sim_reward_unnorm


def get_normalized_rewards(gap_reward, sim_reward, normalization_params):
    std_gap_reward = (
        gap_reward - normalization_params["bandgap"]["mean"]
    ) / normalization_params["bandgap"]["std_dev"]
    std_sim_reward = (
        sim_reward - normalization_params["similarity"]["mean"]
    ) / normalization_params["similarity"]["std_dev"]
    # return sigmoid(std_gap_reward), sigmoid(std_sim_reward)
    return std_gap_reward, std_sim_reward


def get_total_reward(gap_reward, sim_reward, train_params, reduction="sum"):
    if reduction == "sum":
        return -1 * (
            train_params["sum_weights"]["bandgap"] * gap_reward
            + train_params["sum_weights"]["similarity"] * sim_reward
        )
    elif reduction == "product":
        return -1 * gap_reward * sim_reward


# def get_total_reward(gap_reward, sim_reward, train_params, normalization_params, reduction='sum'):
#     normalized_gap_reward, normalized_sim_reward = get_normalized_rewards(gap_reward, sim_reward, normalization_params)
#     if reduction == 'sum':
#         return train_params['sum_weights']['bandgap'] * (1 - normalized_gap_reward) + train_params['sum_weights']['similarity'] * (1 - normalized_sim_reward)
#     elif reduction == 'product':
#         return (1 - normalized_gap_reward) * (1 - normalized_sim_reward)
#     elif reduction == 'min':
#         return min(1 - normalized_gap_reward, 1 - normalized_sim_reward)
