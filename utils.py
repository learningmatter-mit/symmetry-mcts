import os
from rdkit import Chem
from rdkit.Chem import Descriptors


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def compute_molecular_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.ExactMolWt(mol)
