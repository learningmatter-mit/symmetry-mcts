import copy
import re
from rdkit import Chem
from rdkit.Chem import AllChem

def find_isotope_mass_from_string(smi):
    return [int(mass) for mass in re.findall(r'\[(\d+)He\]', smi)]

def fill_inert_positions(smi, fragments_tuple_list):
    mol = Chem.MolFromSmiles(smi)
    for frag_smi, isotope_number in fragments_tuple_list:
        reaction_smarts = "[*:1][{}He].[*:2][{}He]>>[*:1]-[*:2]".format(isotope_number, isotope_number)
        mol2 = Chem.MolFromSmiles(frag_smi)
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        products = rxn.RunReactants((mol, mol2))
        mol = products[0][0]
    return Chem.MolToSmiles(mol)

def run_decomposition_reaction(smiles, dummy1='Ne', dummy2='Ne'):
    reaction_smarts = "{}>>[*:1][{}].[*:2][{}]".format(reactant_smarts, dummy1, dummy2)
    mol = Chem.MolFromSmiles(smiles)
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    products = rxn.RunReactants((mol,))
    return products

def map_unique_fragments(smiles):
    products = run_decomposition_reaction(smiles)
    mapper_dict = {}
    mapper_dict_inverse = {}
    isotope_mass = 100
    for prod_pair in products:
        for prod in prod_pair:
            canonical_smiles = Chem.MolToSmiles(prod)
            if canonical_smiles not in mapper_dict:
                mapper_dict[canonical_smiles] = isotope_mass
                mapper_dict_inverse[isotope_mass] = canonical_smiles.replace('Ne', '{}He'.format(isotope_mass))
                isotope_mass += 1
    return mapper_dict, mapper_dict_inverse

def get_fragments(smiles, memo={}):
    if smiles in memo:
        return memo[smiles]

    products = run_decomposition_reaction(smiles, dummy1='Ne', dummy2='Ne')
    product_smiles_list = [Chem.MolToSmiles(prod) for prod in products[0]]
    filled_smiles_list = copy.deepcopy(product_smiles_list)
    for i, prod in enumerate(products[0]):
        prod_smiles = Chem.MolToSmiles(prod)
        if 'He' in prod_smiles:
            isotope_masses = find_isotope_mass_from_string(prod_smiles)
            fragments_tuple_list = [(mapper_dict_inverse[isotope_mass], isotope_mass) for isotope_mass in isotope_masses]
            filled_smiles = fill_inert_positions(prod_smiles, fragments_tuple_list)
            filled_smiles_list[i] = filled_smiles

    new_smiles_list = [smi.replace('Ne', '{}He'.format(mapper_dict[filled_smiles_list[1-i]])) for i, smi in enumerate(product_smiles_list)]
    new_mols_list = [Chem.MolFromSmiles(smi) for smi in new_smiles_list]
    for i, prod in enumerate(new_mols_list):
        if prod.HasSubstructMatch(Chem.MolFromSmarts(reactant_smarts)):
            get_fragments(new_smiles_list[i], memo)
        else:
            fragments_set.add(new_smiles_list[i])
    memo[smiles] = fragments_set
    return fragments_set
 
if __name__ == '__main__':
    smiles = 'c1(C)cc(C)cc(CC)c1'
    smiles = 'N(c1ccc(CC)cc1)(c2ccc(CC)cc2)(CC)'
    smiles = 'Cc1c(-c2ccc(C=C3C(=O)c4c(Cl)cc(Br)cc4C3=C(C#N)C#N)[nH]2)oc2c1oc1c2oc2c3oc4c5oc(-c6ccc(C=C7C(=O)c8c(Cl)cc(Br)cc8C7=C(C#N)C#N)[nH]6)c(C)c5oc4c3c3nn(C)nc3c12'
    reactant_smarts = '[a:1]-[A;!He;R0:2]'
    fragments_set = set()
    mapper_dict, mapper_dict_inverse = map_unique_fragments(smiles)
    print(mapper_dict)
    print(get_fragments(smiles))
