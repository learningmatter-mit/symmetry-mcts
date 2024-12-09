import sys

sys.path.insert(0, "/home/gridsan/sakshay/molMCTS")
from sklearn.cluster import KMeans
import numpy as np
import json
import copy
from rdkit import Chem
import pandas as pd
from rdkit.Chem import rdMolDescriptors as rdmd
from utils import find_isotope_mass_from_string


df = pd.read_csv("fragments_patents.csv")


def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def is_within_size_limit_core(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    num_atoms = mol.GetNumHeavyAtoms()
    return num_atoms <= 35


def is_within_size_limit_bridge(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    num_atoms = mol.GetNumHeavyAtoms()
    return num_atoms <= 10


def is_within_size_limit_end_group(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    num_atoms = mol.GetNumHeavyAtoms()
    return num_atoms <= 20


def satisfy_charge_limit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    charge = Chem.GetFormalCharge(mol)
    if charge != 0:
        return False
    return True


def fill_inert_positions(smi):
    for isotope_num in list(set(find_isotope_mass_from_string(smi))):
        smi = smi.replace(str(isotope_num) + "He", "H")
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi)))


def satisfy_reactive_position_identity(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "He":
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
            if any([n not in ["C", "Si"] for n in neighbors]):
                return False
    return True


def num_unique_positions(smi):
    return len(set(find_isotope_mass_from_string(smi))) == 2


# Filter out rows with invalid SMILES strings
df = df[df["fragments"].apply(is_valid_smiles)]
df = df[df["fragments"].apply(satisfy_charge_limit)]

filled_smiles = [fill_inert_positions(smi) for smi in df["fragments"]]
df["filled"] = filled_smiles

cores_df = df.loc[(df["num_positions"] >= 1) & (df["num_positions"] <= 4)]
cores_df = cores_df[cores_df["fragments"].apply(is_within_size_limit_core)]

bridges_df = df.loc[(df["num_positions"] == 2)]
bridges_df = bridges_df[bridges_df["fragments"].apply(num_unique_positions)]
bridges_df = bridges_df[
    bridges_df["fragments"].apply(satisfy_reactive_position_identity)
]
bridges_df = bridges_df[bridges_df["fragments"].apply(is_within_size_limit_bridge)]

end_groups_df = df.loc[df["num_positions"] == 1]
end_groups_df = end_groups_df[
    end_groups_df["fragments"].apply(satisfy_reactive_position_identity)
]
end_groups_df = end_groups_df[
    end_groups_df["fragments"].apply(is_within_size_limit_end_group)
]

cores_df.to_csv("cores.csv", index=False)
bridges_df.to_csv("bridges.csv", index=False)
end_groups_df.to_csv("end_groups.csv", index=False)


for tp in ["cores", "bridges", "end_groups"]:
    cores_df = pd.read_csv("{}.csv".format(tp))

    fps = [
        rdmd.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(core), 2, nBits=2048)
        for core in cores_df["fragments"]
    ]
    print("Len(fps) ", len(fps))

    kmeans = KMeans(n_clusters=100, random_state=0).fit_predict(fps)
    print("Len(kmeans) ", len(kmeans))

    cluster_numbers = []
    for i, cluster in enumerate(kmeans):
        cluster_numbers.append(int(cluster))

    cores_df["cluster"] = cluster_numbers

    cores_df = cores_df.sort_values(by="cluster", ascending=True)

    cores_df.to_csv("{}_with_clusters.csv".format(tp), index=False)
