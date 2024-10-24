import pandas as pd
from rdkit import Chem

df = pd.read_csv("patent_opd.csv")
df_smiles = list(df.smiles)
valid_smiles = []


def filter_molecules(smiles):
    """
    Filters molecules based on specific criteria and returns a valid SMILES string if all criteria are met.

    Args:
        smiles (str): The SMILES string representing the molecule to be filtered.

    Returns:
        str or None: The filtered SMILES string if the molecule meets all criteria, otherwise None.

    Criteria:
        - The molecule must contain only the following atoms: C, O, N, H, Cl, Br, S, F, I, Si.
        - The molecule must not contain the problematic atom representation "[O]".
        - The molecule must contain at least one aromatic ring.
        - Stereochemistry information is removed from the molecule.

    Notes:
        - If the molecule does not meet any of the criteria, a message is printed indicating the reason for removal.
        - If the SMILES string is invalid, a message is printed indicating the invalidity.
    """
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
