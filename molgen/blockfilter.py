
from rdkit.Chem import AllChem

from blocks import blockset, block


def smilestomolweight(smile):
    molsmile = AllChem.MolFromSmiles(str(smile))
    molweight = AllChem.CalcExactMolWt(molsmile)
    return molweight


def smiles_contains_any_bonds(smiles, bond_list):
    molsmile = AllChem.MolFromSmiles(str(smiles))
    for bond in bond_list:
        if molsmile.HasSubstructMatch(AllChem.MolFromSmiles(bond)):
            return True
    return False

