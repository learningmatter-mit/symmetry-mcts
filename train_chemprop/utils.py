import os
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


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


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
