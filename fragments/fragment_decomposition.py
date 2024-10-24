import os
import sys
import copy
import re

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import find_isotope_mass_from_string


class FragmentDecomp:
    """
    A class used to decompose chemical structures into fragments using SMILES notation.

    Attributes
    ----------
    fragments_set : set
        A set to store unique fragments.
    smiles : str
        The SMILES string of the molecule to be decomposed.
    reactant_smarts : str
        The SMARTS pattern used for the decomposition reaction.
    mapper_dict : dict
        A dictionary mapping canonical SMILES to unique isotope masses.
    mapper_dict_inverse : dict
        A dictionary mapping unique isotope masses to canonical SMILES with isotopes.

    Methods
    -------
    fill_inert_positions(smi, fragments_tuple_list)
        Fills inert positions in the molecule with specified fragments.
    run_decomposition_reaction(smiles, dummy1="Ne", dummy2="Ne")
        Runs the decomposition reaction on the given SMILES string.
    map_unique_fragments(smiles)
        Maps unique fragments from the decomposition reaction to isotope masses.
    uniquify(original_list)
        Returns a list of unique items from the original list.
    get_fragments()
        Returns a set of canonical SMILES strings of the fragments.
    _get_fragments(smiles, memo={})
        Recursively decomposes the molecule and returns a set of fragments.
    """
    def __init__(self, smiles):
        self.fragments_set = set()
        self.smiles = smiles
        self.reactant_smarts = "[*&R1:1]!@;-[*&!He&R0,*&R1:2]"
        # self.reactant_smarts = "[*&R1:1]!@;-[*&R1:2]"
        self.mapper_dict, self.mapper_dict_inverse = self.map_unique_fragments(
            self.smiles
        )

    def fill_inert_positions(self, smi, fragments_tuple_list):
        """
        Fills inert positions in a molecule with fragments based on the provided isotope numbers.

        Args:
            smi (str): The SMILES string of the molecule.
            fragments_tuple_list (list of tuples): A list of tuples where each tuple contains:
                - frag_smi (str): The SMILES string of the fragment.
                - isotope_number (int): The isotope number used to identify the inert positions.

        Returns:
            str: The SMILES string of the modified molecule with inert positions filled.
        """
        mol = Chem.MolFromSmiles(smi)
        for frag_smi, isotope_number in fragments_tuple_list:
            reaction_smarts = "[*:1][{}He].[*:2][{}He]>>[*:1]-[*:2]".format(
                isotope_number, isotope_number
            )
            mol2 = Chem.MolFromSmiles(frag_smi)
            rxn = AllChem.ReactionFromSmarts(reaction_smarts)
            products = rxn.RunReactants((mol, mol2))
            mol = products[0][0]
        return Chem.MolToSmiles(mol)

    def run_decomposition_reaction(self, smiles, dummy1="Ne", dummy2="Ne"):
        """
        Executes a decomposition reaction on a given SMILES string using specified dummy atoms.

        Args:
            smiles (str): The SMILES string of the molecule to decompose.
            dummy1 (str, optional): The first dummy atom to use in the reaction. Defaults to "Ne".
            dummy2 (str, optional): The second dummy atom to use in the reaction. Defaults to "Ne".

        Returns:
            tuple: A tuple containing the products of the reaction as RDKit molecule objects.
        """
        reaction_smarts = "{}>>[*:1][{}].[*:2][{}]".format(
            self.reactant_smarts, dummy1, dummy2
        )
        mol = Chem.MolFromSmiles(smiles)
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        products = rxn.RunReactants((mol,))
        return products

    def map_unique_fragments(self, smiles):
        """
        Maps unique fragments from the given SMILES string to unique isotope masses.

        This method performs a decomposition reaction on the input SMILES string and
        assigns unique isotope masses to each unique fragment produced. It returns
        two dictionaries: one mapping the canonical SMILES of each fragment to its
        assigned isotope mass, and another mapping the isotope mass back to a modified
        canonical SMILES string where "Ne" is replaced with the isotope mass followed
        by "He".

        Args:
            smiles (str): The input SMILES string representing the molecule to be decomposed.

        Returns:
            tuple: A tuple containing two dictionaries:
                - mapper_dict (dict): A dictionary mapping canonical SMILES strings of fragments to unique isotope masses.
                - mapper_dict_inverse (dict): A dictionary mapping isotope masses to modified canonical SMILES strings.
        """
        products = self.run_decomposition_reaction(smiles)
        mapper_dict = {}
        mapper_dict_inverse = {}
        isotope_mass = 100
        for prod_pair in products:
            for prod in prod_pair:
                canonical_smiles = Chem.MolToSmiles(prod)
                if canonical_smiles not in mapper_dict:
                    mapper_dict[canonical_smiles] = isotope_mass
                    mapper_dict_inverse[isotope_mass] = canonical_smiles.replace(
                        "Ne", "{}He".format(isotope_mass)
                    )
                    isotope_mass += 1
        return mapper_dict, mapper_dict_inverse

    def uniquify(self, original_list):
        """
        Remove duplicates from the original list while preserving the order.

        Args:
            original_list (list): The list from which duplicates need to be removed.

        Returns:
            list: A new list with duplicates removed, preserving the original order.
        """
        unique_list = []
        seen = set()

        for item in original_list:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)
        return unique_list

    def get_fragments(self):
        """
        Extracts and canonicalizes fragments from a SMILES string.

        This method retrieves fragments from the SMILES string associated with the instance,
        canonicalizes them, and then replaces isotope masses with a new range of masses.

        Returns:
            set: A set of canonicalized fragment SMILES strings with updated isotope masses.
        """
        fragments = list(self._get_fragments(self.smiles))
        canon_fragments = []
        for frag in fragments:
            canon_fragments.append(Chem.CanonSmiles(frag))

        for i, frag in enumerate(canon_fragments):
            masses = self.uniquify(find_isotope_mass_from_string(frag))
            new_masses = list(range(100, 100 + len(masses)))
            assert len(masses) == len(new_masses)
            for j, mass in enumerate(masses):
                canon_fragments[i] = canon_fragments[i].replace(
                    str(mass), str(new_masses[j])
                )
        return set(canon_fragments)

    def _get_fragments(self, smiles, memo={}):
        """
        Decompose a given SMILES string into its fragments and store them in a set.

        This method performs a decomposition reaction on the input SMILES string and processes
        the resulting fragments. It handles isotopic masses and fills inert positions in the
        fragments. The fragments are then checked against a reactant SMARTS pattern, and if
        they match, the method is called recursively on the new fragments. The resulting
        fragments are stored in a set and memoized for future use.

        Args:
            smiles (str): The SMILES string to be decomposed.
            memo (dict, optional): A dictionary for memoization to store previously computed
                                   fragments. Defaults to an empty dictionary.

        Returns:
            set: A set of decomposed fragment SMILES strings.
        """
        if smiles in memo:
            return memo[smiles]

        products = self.run_decomposition_reaction(smiles, dummy1="Ne", dummy2="Ne")
        product_smiles_list = [Chem.MolToSmiles(prod) for prod in products[0]]
        filled_smiles_list = copy.deepcopy(product_smiles_list)
        for i, prod in enumerate(products[0]):
            prod_smiles = Chem.MolToSmiles(prod)
            if "He" in prod_smiles:
                isotope_masses = find_isotope_mass_from_string(prod_smiles)
                fragments_tuple_list = [
                    (self.mapper_dict_inverse[isotope_mass], isotope_mass)
                    for isotope_mass in isotope_masses
                ]
                filled_smiles = self.fill_inert_positions(
                    prod_smiles, fragments_tuple_list
                )
                filled_smiles_list[i] = filled_smiles

        new_smiles_list = [
            smi.replace(
                "Ne", "{}He".format(self.mapper_dict[filled_smiles_list[1 - i]])
            )
            for i, smi in enumerate(product_smiles_list)
        ]
        new_mols_list = [Chem.MolFromSmiles(smi) for smi in new_smiles_list]
        for i, prod in enumerate(new_mols_list):
            if prod == None:
                print("Prod is none: ", new_smiles_list[i])
                print(self.smiles)
            if prod.HasSubstructMatch(Chem.MolFromSmarts(self.reactant_smarts)):
                self._get_fragments(new_smiles_list[i], memo)
            else:
                self.fragments_set.add(new_smiles_list[i])
        memo[smiles] = self.fragments_set
        return self.fragments_set


def driver(smi):
    """
    Decomposes a given SMILES string into its constituent fragments.

    Args:
        smi (str): A SMILES (Simplified Molecular Input Line Entry System) string representing a molecule.

    Returns:
        list or None: A list of fragments if decomposition is successful, otherwise None.
    """
    canon_smi = Chem.CanonSmiles(smi)
    frag_obj = FragmentDecomp(canon_smi)
    try:
        return frag_obj.get_fragments()
    except:
        return None


if __name__ == "__main__":
    # smi1 = 'c1(C)c(c2ccccc2)sc(c3ccccc3)c1'
    # smi1 = 'c1(c2ccccc2)c(C)sc(c3ccccc3)c1'
    # smiles = 'c1(C)cc(C)cc(CC)c1'
    # # smiles = 'N(c1ccc(CC)cc1)(c2ccc(CC)cc2)(CC)'
    # smiles = 'Cc1c(-c2ccc(C=C3C(=O)c4c(Cl)cc(Br)cc4C3=C(C#N)C#N)[nH]2)oc2c1oc1c2oc2c3oc4c5oc(-c6ccc(C=C7C(=O)c8c(Cl)cc(Br)cc8C7=C(C#N)C#N)[nH]6)c(C)c5oc4c3c3nn(C)nc3c12'
    # smi1 = 'c1(C)cc(CC)cc(C)c1'
    # smi2 = 'c1([Br])cc(c2cccc2)cc(Br)c1'

    # frag_obj1 = FragmentDecomp(Chem.CanonSmiles(smi1))
    # frag_obj2 = FragmentDecomp(Chem.CanonSmiles(smi2))
    # print(frag_obj1.get_fragments())
    # print(frag_obj2.get_fragments())

    fragments = set()

    patent_smiles = list(pd.read_csv("valid_smiles_patent_opd_with_Si.csv")["smiles"])
    all_data = Parallel(n_jobs=48)(delayed(driver)(smile) for smile in patent_smiles)
    all_data = [d for d in all_data if d != None]
    fragments = set()
    for datum in all_data:
        fragments = fragments.union(datum)

    valid_fragments = []
    for frag in fragments:
        if Chem.MolFromSmiles(frag) != None:
            valid_fragments.append(frag)
        else:
            print("Invalid fragment, skipping: ", frag)
    num_positions = []
    for frag in fragments:
        num_positions.append(len(find_isotope_mass_from_string(frag)))

    df_frags = pd.DataFrame(
        {"fragments": list(fragments), "num_positions": list(num_positions)}
    )
    df_frags.to_csv("fragments_patents_with_Si.csv", index=False)
