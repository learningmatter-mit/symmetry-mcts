import math
import os
import re
import json
import random
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


def set_all_seeds(seed):
    random.seed(seed)
    #   os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_smiles_validity(smi):
    if Chem.MolFromSmiles(smi):
        return True
    return False


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
    """
    Returns the identity element for a given reduction operation.

    Parameters:
    reduction (str): The type of reduction operation. 
                     Supported values are "sum", "product", and "min".

    Returns:
    int or float: The identity element for the specified reduction operation.
                  - For "sum", returns 0.
                  - For "product", returns 1.
                  - For "min", returns positive infinity.
    """
    if reduction == "sum":
        return 0
    elif reduction == "product":
        return 1
    elif reduction == "min":
        return float("inf")


def get_total_reward(gap_reward, sim_reward, train_params, reduction="sum"):
    """
    Calculate the total reward based on the given gap reward, similarity reward, 
    and training parameters.

    Args:
        gap_reward (float): The reward associated with the bandgap.
        sim_reward (float): The reward associated with the similarity.
        train_params (dict): A dictionary containing training parameters, 
                             specifically the weights for summing the rewards.
        reduction (str, optional): The method to reduce the rewards. 
                                   Can be "sum" or "product". Defaults to "sum".

    Returns:
        float: The total reward calculated based on the specified reduction method.
    """
    if reduction == "sum":
        return -1 * (
            train_params["sum_weights"]["bandgap"] * gap_reward
            + train_params["sum_weights"]["similarity"] * sim_reward
        )
    elif reduction == "product":
        return -1 * gap_reward * sim_reward
