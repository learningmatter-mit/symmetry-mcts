import copy
import re

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from joblib import Parallel, delayed


def find_isotope_mass_from_string(smi):
    return [int(mass) for mass in re.findall(r'\[(\d+)He\]', smi)]

class FragmentDecomp:
    def __init__(self, smiles):
        self.fragments_set = set()
        self.smiles = smiles
        self.reactant_smarts = '[*&R1:1]!@;-[*&!He&R0,*&R1:2]'
        # self.reactant_smarts = '[a:1]!@;-[A&!He&R0,a:2]'
        # self.reactant_smarts = '[a:1]!@[A&!He&R0,a:2]'
        # self.reactant_smarts = '[a:1]-[A&!He&R0,a:2]'
        self.mapper_dict, self.mapper_dict_inverse = self.map_unique_fragments(self.smiles)

    def fill_inert_positions(self, smi, fragments_tuple_list):
        mol = Chem.MolFromSmiles(smi)
        # if mol == None:
        #     print(smi)
        for frag_smi, isotope_number in fragments_tuple_list:
            reaction_smarts = "[*:1][{}He].[*:2][{}He]>>[*:1]-[*:2]".format(isotope_number, isotope_number)
            mol2 = Chem.MolFromSmiles(frag_smi)
            rxn = AllChem.ReactionFromSmarts(reaction_smarts)
            products = rxn.RunReactants((mol, mol2))
            mol = products[0][0]
        return Chem.MolToSmiles(mol)

    def run_decomposition_reaction(self, smiles, dummy1='Ne', dummy2='Ne'):
        reaction_smarts = "{}>>[*:1][{}].[*:2][{}]".format(self.reactant_smarts, dummy1, dummy2)
        mol = Chem.MolFromSmiles(smiles)
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        products = rxn.RunReactants((mol,))
        return products

    def map_unique_fragments(self, smiles):
        products = self.run_decomposition_reaction(smiles)
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

    def uniquify(self, original_list):
        unique_list = []
        seen = set()

        for item in original_list:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)
        return unique_list

    def get_fragments(self):
        fragments = list(self._get_fragments(self.smiles))
        canon_fragments = []
        for frag in fragments:
            # canon_fragments.append(frag)
            canon_fragments.append(Chem.CanonSmiles(frag))

        for i, frag in enumerate(canon_fragments):
            masses = self.uniquify(find_isotope_mass_from_string(frag))
            new_masses = list(range(100, 100 + len(masses)))
            assert len(masses) == len(new_masses)
            for j, mass in enumerate(masses):
                canon_fragments[i] = canon_fragments[i].replace(str(mass), str(new_masses[j]))
        return set(canon_fragments)

    def _get_fragments(self, smiles, memo={}):
        if smiles in memo:
            return memo[smiles]

        products = self.run_decomposition_reaction(smiles, dummy1='Ne', dummy2='Ne')
        product_smiles_list = [Chem.MolToSmiles(prod) for prod in products[0]]
        filled_smiles_list = copy.deepcopy(product_smiles_list)
        for i, prod in enumerate(products[0]):
            prod_smiles = Chem.MolToSmiles(prod)
            if 'He' in prod_smiles:
                isotope_masses = find_isotope_mass_from_string(prod_smiles)
                fragments_tuple_list = [(self.mapper_dict_inverse[isotope_mass], isotope_mass) for isotope_mass in isotope_masses]
                filled_smiles = self.fill_inert_positions(prod_smiles, fragments_tuple_list)
                filled_smiles_list[i] = filled_smiles

        new_smiles_list = [smi.replace('Ne', '{}He'.format(self.mapper_dict[filled_smiles_list[1-i]])) for i, smi in enumerate(product_smiles_list)]
        new_mols_list = [Chem.MolFromSmiles(smi) for smi in new_smiles_list]
        for i, prod in enumerate(new_mols_list):
            if prod == None:
                print("Prod is none: ",  new_smiles_list[i])
                print(self.smiles)
            if prod.HasSubstructMatch(Chem.MolFromSmarts(self.reactant_smarts)):
                self._get_fragments(new_smiles_list[i], memo)
            else:
                self.fragments_set.add(new_smiles_list[i])
        memo[smiles] = self.fragments_set
        return self.fragments_set

def driver(smi):
    canon_smi = Chem.CanonSmiles(smi)
    frag_obj = FragmentDecomp(canon_smi)
    try:
        return frag_obj.get_fragments()
    except:
        return None

if __name__ == '__main__':
    # smiles = 'c1(C)c(c2ccccc2)sc(c3ccccc3)c1'
    # smiles = 'c1(C)cc(C)cc(CC)c1'
    # # smiles = 'N(c1ccc(CC)cc1)(c2ccc(CC)cc2)(CC)'
    # smiles = 'Cc1c(-c2ccc(C=C3C(=O)c4c(Cl)cc(Br)cc4C3=C(C#N)C#N)[nH]2)oc2c1oc1c2oc2c3oc4c5oc(-c6ccc(C=C7C(=O)c8c(Cl)cc(Br)cc8C7=C(C#N)C#N)[nH]6)c(C)c5oc4c3c3nn(C)nc3c12'
    # smi1 = 'c1(C)cc(CC)cc(C)c1'
    # smi2 = 'c1([Br])cc(c2cccc2)cc(Br)c1'
    
    # frag_obj1 = FragmentDecomp(Chem.CanonSmiles(smi1))
    # frag_obj2 = FragmentDecomp(Chem.CanonSmiles(smi2))
    # print(frag_obj1.get_fragments())
    # print(frag_obj2.get_fragments())
    
    # fragments = set()

    patent_smiles = list(pd.read_csv('valid_smiles_patent_opd.csv')['smiles'])
    all_data = Parallel(n_jobs=48)(delayed(driver)(smile) for smile in patent_smiles) 
    all_data = [d for d in all_data if d != None]
    fragments = set()
    for datum in all_data:
        fragments = fragments.union(datum)

    num_positions = []
    for frag in fragments:
        num_positions.append(len(find_isotope_mass_from_string(frag)))

    
    # for i, smi in tqdm(enumerate(patent_smiles), total=len(patent_smiles)):
    #     canon_smi = Chem.CanonSmiles(smi)
    #     frag_obj = FragmentDecomp(canon_smi)
    #     try:
    #         frags = frag_obj.get_fragments()
    #     except:
    #         print("Skipping: ", smi)
    #         # import pdb; pdb.set_trace()
    #         continue
    #     fragments = fragments.union(frags)
    
    df_frags = pd.DataFrame({'fragments': list(fragments), 'num_positions': list(num_positions)})
    df_frags.to_csv('fragments_patents.csv', index=False)
