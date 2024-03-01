import re
from rdkit import Chem
from rdkit.Chem import AllChem

sample_smiles = "Cc1ccc2c(c1)C(C)(C)CC(C)(C)c1cc3c(cc1-2)C(C)(C)C(C)(C)c1cc(N)ccc1-3"
sample_smiles = "O=c1oc2ccccc2c2ccc(-c3ccc(-c4ccc5sc6ccccc6c5c4)cc3)cc12"
mol = Chem.MolFromSmiles(sample_smiles)
# reactant_smarts = '[a:1]-[A;!He;R0:2]'
# reactant_smarts = '[a:1]!@;-[A&!He&R0,a:2]'
reactant_smarts = "[*&R1:1]!@;-[*&!He&R0,*&R1:2]"

reaction_smarts = "{}>>[*:1][{}].[*:2][{}]".format(reactant_smarts, "Ne", "Ne")

rxn = AllChem.ReactionFromSmarts(reaction_smarts)
products = rxn.RunReactants((mol,))
for i, prod_set in enumerate(products):
    print("product: ", i)
    for prod in prod_set:
        print(Chem.MolToSmiles(prod))

print(mol.HasSubstructMatch(Chem.MolFromSmarts(reactant_smarts)))


# def find_isotope_mass_from_string(smi):
#     return [int(mass) for mass in re.findall(r'\[(\d+)He\]', smi)]

# def uniquify(original_list):
#     unique_list = []
#     seen = set()

#     for item in original_list:
#         if item not in seen:
#             unique_list.append(item)
#             seen.add(item)
#     return unique_list

# def get_fragments():
#     smi1 = 'c1([100He])nc([101He])cc([100He])c1'
#     smi2 = 'c1([102He])cc([105He])nc([102He])c1'
#     fragments = [smi1, smi2]
#     print("Unfixed: ", fragments)
#     canon_fragments = []
#     for frag in fragments:
#         # canon_fragments.append(frag)
#         canon_fragments.append(Chem.CanonSmiles(frag))

#     for i, frag in enumerate(canon_fragments):
#         masses = uniquify(find_isotope_mass_from_string(frag))
#         new_masses = list(range(100, 100 + len(masses)))
#         assert len(masses) == len(new_masses)
#         for j, mass in enumerate(masses):
#             canon_fragments[i] = canon_fragments[i].replace(str(mass), str(new_masses[j]))
#     return set(canon_fragments)

# if __name__ == '__main__':
#     print(get_fragments())
