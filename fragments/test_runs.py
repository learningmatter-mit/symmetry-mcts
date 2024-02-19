from rdkit import Chem
from rdkit.Chem import AllChem

sample_smiles = 'c1c(C)c(c2ccccc2)ccc1'
mol = Chem.MolFromSmiles(sample_smiles)
reactant_smarts = '[a:1]-[A&!He&R0,a:2]'

reaction_smarts = "{}>>[*:1][{}].[*:2][{}]".format(reactant_smarts, 'He', 'He')

rxn = AllChem.ReactionFromSmarts(reaction_smarts)
products = rxn.RunReactants((mol,))
for i, prod_set in enumerate(products):
    print("product: ", i)
    for prod in prod_set:
        print(Chem.MolToSmiles(prod))

print(mol.HasSubstructMatch(Chem.MolFromSmarts(reactant_smarts)))