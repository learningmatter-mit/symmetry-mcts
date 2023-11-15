from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List

class MolFragments:
    def __init__(self, smiles):
        self.inert_atoms = ['He', 'Ne', 'Ar', 'Xe', 'Kr']
        self.smiles = smiles
        self.moving_core_smiles = smiles
        self.products = {'core': '', 'functional_grps': {}}

    def run_reaction(self, recursive_ctr) -> List[List]:
        # Define your reaction SMARTS
        if recursive_ctr > 4:
            self.products['core'] = self.moving_core_smiles
            return self.moving_core_smiles
        # reaction_smarts = "[a:1]-[A:2]>>[*:1][{}].[*:2][{}]".format(self.inert_atoms[recursive_ctr], '*')
        # reaction_smarts = "[a:1]-[!$([A]=[He,Ne,Ar,Xe,Kr]):2]>>[*:1][{}].[*:2][{}]".format(self.inert_atoms[recursive_ctr], '*')
        reaction_smarts = "[a:1]-[A;!He;!Ne;!Ar;!Xe;!Kr;R0:2]>>[*:1][{}].[*:2][{}]".format(self.inert_atoms[recursive_ctr], '*')
        mol = Chem.MolFromSmiles(self.moving_core_smiles)
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)

        products = rxn.RunReactants((mol,))
        if len(products) == 0:
            self.products['core'] = self.moving_core_smiles
            return self.moving_core_smiles

        # for i in range(len(products)):
        #     for j in range(len(products[i])):
        #         print(Chem.MolToSmiles(products[i][j]))
        #     print('---------')
        # print('xxxxxxxxxxxxx')
        self.moving_core_smiles = Chem.MolToSmiles(products[0][0])
        fxn_grp_smiles = Chem.MolToSmiles(products[0][1])
        if fxn_grp_smiles in self.products['functional_grps'].keys():
            self.moving_core_smiles = self.moving_core_smiles.replace(self.inert_atoms[recursive_ctr], self.products['functional_grps'][fxn_grp_smiles])
            recursive_ctr -= 1
        else:
            self.products['functional_grps'][fxn_grp_smiles] = self.inert_atoms[recursive_ctr]

        return self.run_reaction(recursive_ctr+1)


if __name__ == '__main__':
    # Define your input SMILES string
    AAA = MolFragments("c1(C)cc(C)cc(C)c1")
    AAB = MolFragments("c1(C)cc(C)cc(CC)c1")
    ABC = MolFragments("c1(C)cc(CC)cc(CCC)c1")
    input_smiles = MolFragments("Cc1nc2c(nc1C)c1c3c(c4oc(C=CC=CC=CC=CC=C5C(=O)c6cc(F)c(F)cc6C5=C(C#N)C#N)c(C)c4n3C)n(C)c1c1c2c2c(c3oc(C=CC=CC=CC=CC=C4C(=O)c5cc(F)c(F)cc5C4=C(C#N)C#N)c(C)c3n2C)n1C")
    patent_mol1 = MolFragments("Cc1c2ccccc2c(-c2cccc(-n3c(-c4ccccc4)nc4ccccc43)c2)c2ccccc12")
    patent_mol2 = MolFragments("CCSc1c(CCc2ccccc2)ccc2c(SCC)c(NCc3ccccc3)ccc12")
    patent_mol3 = MolFragments("CCSc1c(CCc2ccccc2)ccc2c(SCC)c(CCc3ccccc3)ccc12")
    patent_mol4 = MolFragments("Cc1ccc2c(c1)C(C)(C)CC(C)(C)c1cc3c(cc1-2)C(C)(C)C(C)(C)c1cc(N)ccc1-3")
    patent_mol5 = MolFragments("Cc1c(-c2ccc(C=C3C(=O)c4c(Cl)cc(Br)cc4C3=C(C#N)C#N)[nH]2)oc2c1oc1c2oc2c3oc4c5oc(-c6ccc(C=C7C(=O)c8c(Cl)cc(Br)cc8C7=C(C#N)C#N)[nH]6)c(C)c5oc4c3c3nn(C)nc3c12")

    AAA.run_reaction(0)
    print(AAA.products)

    AAB.run_reaction(0)
    print(AAB.products)

    ABC.run_reaction(0)
    print(ABC.products)

    input_smiles.run_reaction(0)
    print(input_smiles.products)

    patent_mol1.run_reaction(0)
    print(patent_mol1.products)

    patent_mol2.run_reaction(0)
    print(patent_mol2.products)

    patent_mol3.run_reaction(0)
    print(patent_mol3.products)

    patent_mol4.run_reaction(0)
    print(patent_mol4.products)

    patent_mol5.run_reaction(0)
    print(patent_mol5.products)
