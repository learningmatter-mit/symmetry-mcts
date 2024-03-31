# import pandas as pd
# from rdkit import Chem


# df = pd.read_csv('patent_opd.csv')
# df_smiles = list(df.smiles)
# valid_smiles = []
# for smi in df_smiles:
#     mol = Chem.MolFromSmiles(smi)
#     if mol != None:
#         Chem.RemoveStereochemistry(mol)
#         valid_smiles.append(Chem.MolToSmiles(mol))
#     else:
#         print("Skipping")
# df_new = pd.DataFrame({'smiles': valid_smiles})
# df_new.to_csv('valid_smiles_patent_opd.csv', index=False)
import pandas as pd
from rdkit import Chem

df = pd.read_csv("patent_opd.csv")
df_smiles = list(df.smiles)
valid_smiles = []


def filter_molecules(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Check for atoms other than C, O, N, H, Cl, Br, S, F, I
        allowed_atoms = set(["C", "O", "N", "H", "Cl", "Br", "S", "F", "I", "Si"])
        atoms = set([atom.GetSymbol() for atom in mol.GetAtoms()])
        if atoms.issubset(allowed_atoms):
            # Check for problematic atom representations
            if "[O]" not in smiles:
                # Check for aromaticity
                if any([atom.GetIsAromatic() for atom in mol.GetAtoms()]):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    # Append valid SMILES to the list
                    return Chem.MolToSmiles(mol)
                else:
                    print(f"Removed molecule without aromatic ring: {smiles}")
            else:
                print(
                    f"Removed molecule with problematic atom representation: {smiles}"
                )
        else:
            print(f"Removed molecule with forbidden atoms: {smiles}")
    else:
        print(f"Invalid SMILES string: {smiles}")

    return None


# Apply the filtering function to each SMILES string
valid_smiles = [filter_molecules(smi) for smi in df_smiles]

# Create a new DataFrame with the valid SMILES
df_new = pd.DataFrame({"smiles": [smi for smi in valid_smiles if smi is not None]})

# Save the DataFrame to a new CSV file
df_new.to_csv("valid_smiles_patent_opd_with_Si.csv", index=False)
