"""
unittests for rdkit

these aren't actually testing rdkit in the way that it already has
its own tests

Instead, these are tests to confirm the behavior of rdkit in the ways
we need it to.


@author Tim Hirzel <hirzel@chemistry.harvard.edu>
"""

import unittest
from rdkit.Chem import AllChem


def rdkit_react(smile_left, smile_right, smarts, sanitize=False):
    """
    return list of tuples (Left, Right) of products from rdkit
    """
    mol_left = AllChem.MolFromSmiles(str(smile_left))
    mol_right = AllChem.MolFromSmiles(str(smile_right))
    if sanitize:
        AllChem.SanitizeMol(mol_left)
        AllChem.SanitizeMol(mol_right)
    rxn = AllChem.ReactionFromSmarts(str(smarts))
    products = rxn.RunReactants((mol_left, mol_right))

    # by pulling out p[0], we are ignoring the "reaction remnants"
    # ie. [Mg][Mg],  [Fe][Be] etc. on the right side of the product

    product_smiles = [(AllChem.MolToSmiles(p[0]), AllChem.MolToSmiles(p[1])) for p in products]
    # list of set ensure no duplicate results in return list
    return product_smiles

# SMARTS
L_REACTANT = "[C,c:1]:,=[C,c:2]:,-[C,c:3]([H:12]):,=[C,c:4][Mg:5]"
R_REACTANT = "[C,c:6]:,=[C,c:7]:,-[C,c:8]([H:13]):,=[C,c:9]([Mg:11]):,-[C,c:10]"
L_PROD = "[C,c:1]:,=[C,c:2]:,-[C,c:3](:,-[C,c:10]):,=[C,c:4]:,-[C,c:7]:,=[C,c:6]"
R_PROD = "[*:12][*:5][*:8][*:13][*:9][*:11]"
FUSION_SMARTS = L_REACTANT + "." + R_REACTANT + ">>" + L_PROD + "." + R_PROD

LINK_SMARTS = "[*:1][Mg:3].[*:2][Mg:4]>>[*:1][*:2].[Mg:3][Mg:4]"


# REACTANTS

BENZENE_A = u"c1([Mg])ccccc1"
PYRIDINE = "c1cc([Mg])ncc1"

BENZENE_B = "C1([Mg])=CC=CC=C1"
NAPHTHALENE = "[Mg]C1=CC2=C(C=CC=C2)C=C1"

SMILE_D = "[Mg][Cl]"



class ReactionTests(unittest.TestCase):
    def setUp(self):
        """
        any setup code to run the test methods
        """
        pass

    def test_reaction(self):
        prods = rdkit_react(BENZENE_A, PYRIDINE, LINK_SMARTS)
        self.assertEquals(prods[0], ('c1ccc(cc1)-c1ccccn1', '[Mg][Mg]'))

    def test_rdkit_fusion(self):
        """
        test fusion reaction in rdkit.  Notice how it's different
        that the chemaxon version of the exact same test
        """
        prods = rdkit_react(BENZENE_B, NAPHTHALENE, FUSION_SMARTS)
        expect = []
        self.assertEquals(prods, expect)
        prods = rdkit_react(NAPHTHALENE, BENZENE_B, FUSION_SMARTS)
        expect = []
        self.assertEquals(prods, expect)

    def test_mark(self):
        BENZENE_2 = "c1([Mg])cccc([Mg])c1"
        b2 = AllChem.MolFromSmiles(BENZENE_2)

        atoms = [b2.GetAtomWithIdx(i) for i in range(b2.GetNumAtoms())]
        first = AllChem.MolToSmiles(b2)
        for a in atoms:
            a.SetProp("molAtomMapNumber", str(10))
        expect = "[Mg:10][c:10]1[cH:10][cH:10][cH:10][c:10]([Mg:10])[cH:10]1"
        self.assertEquals(AllChem.MolToSmiles(b2), expect)
        for a in atoms:
            a.ClearProp("molAtomMapNumber")
        self.assertEquals(AllChem.MolToSmiles(b2), first)

    def test_react_all(self):
        PYRIDINE_2 = "c1nc([Mg])cc([Mg])c1"
        BENZENE_2 = "c1([Mg])cccc([Mg])c1"

        b2 = AllChem.MolFromSmiles(BENZENE_2)
        p2 = AllChem.MolFromSmiles(PYRIDINE_2)

        rxn = AllChem.ReactionFromSmarts(LINK_SMARTS)

        print("BEFORE PROTECT")
        prods = rxn.RunReactants((b2, p2))
        self.assertEquals(len(prods), 4)


        reactants = [rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())]
        print([AllChem.MolToSmiles(r) for r in reactants])

        for match in p2.GetSubstructMatches(reactants[1]):
            print(match)
            for m in match:
                p2.GetAtomWithIdx(m).SetProp('_protected', '1')
            try:
                print("AFTER PROTECT")
                prods = rxn.RunReactants((b2, p2))
                self.assertEquals(len(prods), 2)

                for psets in prods:
                    print([AllChem.MolToSmiles(p) for p in psets])

                result = prods[0][0]
                print("result")
                print(AllChem.MolToSmiles(result))

                atoms = [result.GetAtomWithIdx(i) for i in range(result.GetNumAtoms())]
                print([a for a in atoms if a.HasProp("_protected")])
                prods = rxn.RunReactants((result, p2))
                print("complete")
                for psets in prods:
                    print([AllChem.MolToSmiles(p) for p in psets])
            finally:
                for m in match:
                    p2.GetAtomWithIdx(m).ClearProp('_protected')



