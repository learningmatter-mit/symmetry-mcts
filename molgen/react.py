import os
import re
import sys
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import find_isotope_mass_from_string
# cores: > 1 reactive position
# pi bridges: 2 different reactive positions
# end groups: 1 reactive position

def num_sites_from_pos(smi, pos):
    return len(re.findall(rf'\[({pos})He\]', smi))

def uniquify(original_list):
    unique_list = []
    seen = set()

    for item in original_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list

def reduce_to_lowest_form(smi):
    canon_smi = Chem.CanonSmiles(smi)
    masses = uniquify(find_isotope_mass_from_string(canon_smi))
    new_masses = list(range(100, 100 + len(masses)))
    assert len(masses) == len(new_masses)
    for j, mass in enumerate(masses):
        canon_smi = canon_smi.replace(str(mass), str(new_masses[j]))
    return canon_smi

def run(smi1, smi2, pos1, pos2):
    reaction_smarts = "[*:1][{}He].[*:2][{}He]>>[*:1]-[*:2]".format(pos1, pos2)
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    mol2 = Chem.MolFromSmiles(smi2)
    
    num_sites = num_sites_from_pos(smi1, pos1)
    mol1 = Chem.MolFromSmiles(smi1)
    for i in range(num_sites):
        products = rxn.RunReactants((mol1, mol2))
        # try:
        mol1 = products[0][0]
        # except:
        #     import pdb; pdb.set_trace()

    return Chem.MolToSmiles(mol1) 

if __name__ == '__main__':
    smi1 = 'c1([100He])cc([100He])c([101He])cc1'
    smi2 = 'c1([100He])ccc([100He])cc1'
    # smi2 = '[100He]Br'

    print(run(smi1, smi2, 100, 100))
