import unittest
import itertools
from rdkit.Chem.AllChem import MolFromSmarts, CalcExactMolWt

from blocks.reactionenvironment import RDKitReactionEnvironment
from blocks.protect import ExposeOnly
from blocks.blockset import BlockSet
from ..singlereactant_rxnenv import SingleReactantRxnEnv
from ..blockreactor import make_cleaner_func
from blocks.block import Block

class TestCleaner(unittest.TestCase):

    def setUp(self):
        REACTIVE_MOL = """[At:1]C(=O)c1ccc(-c2cc(-c3ccc(C([At:1])=O)cc3)cc(-c3cc4oc5cc(-c6cc(-c7ccc(C([At:1])
        =O)cc7)cc(-c7ccc(C([At:1])=O)cc7)c6)c(-c6cc(-c7ccc(C([At:1])=O)cc7)cc(-c7ccc(C([At:1])=O)cc7)c6)
        cc5c4cc3-c3cc(-c4ccc(C([At:1])=O)cc4)cc(-c4ccc(C([At:1])=O)cc4)c3)c2)cc1"""

        CLEANER = "[C:1](=O)[At,Be,Cd,Dy]>>[*:1](=O)c1ccccc1"

        self.renv = SingleReactantRxnEnv(CLEANER, handles=["At", "Be", "Cd", "Dy"])

        self.block_a = Block("".join(REACTIVE_MOL.split()))

    @unittest.skip
    def test_complex_cleaner(self):
        with self.renv:
            for prod in self.block_a.react_fill():
                print(prod.smiles())

    @unittest.skip
    def test_long_smiles(self):
        block_b = Block(smiles="O=C([At])[c]1[cH][cH][c](-[c]2[cH][c](-[c]3[cH][cH][c]([C](=[O])[At])[cH][cH]3)[cH][c](-[c]3[cH][c]4[o][c]5[cH][c](-[c]6[cH][c](-[c]7[cH][cH][c]([C](=[O])[c]8[cH][cH][cH][cH][cH]8)[cH][cH]7)[cH][c](-[c]7[cH][cH][c]([C](=[O])[c]8[cH][cH][cH][cH][cH]8)[cH][cH]7)[cH]6)[c](-[c]6[cH][c](-[c]7[cH][cH][c]([C](=[O])[c]8[cH][cH][cH][cH][cH]8)[cH][cH]7)[cH][c](-[c]7[cH][cH][c]([C](=[O])[c]8[cH][cH][cH][cH][cH]8)[cH][cH]7)[cH]6)[cH][c]5[c]4[cH][c]3-[c]3[cH][c](-[c]4[cH][cH][c]([C](=[O])[c]5[cH][cH][cH][cH][cH]5)[cH][cH]4)[cH][c](-[c]4[cH][cH][c]([C](=[O])[c]5[cH][cH][cH][cH][cH]5)[cH][cH]4)[cH]3)[cH]2)[cH][cH]1")
        print(block_b.smiles(show_markers=False))
        with self.renv:
            print(block_b.sites(reactant_index=0))
            for prod in block_b.react_fill():
                print(prod.smiles())

    def test_samsung_cleaner(self):
        smarts = ["[#7:1]([At,Be,Cd,Dy])[At,Be,Cd,Dy]>>[*:1](c1ccccc1)c1ccccc1",
                  "[#7:1][At,Be,Cd,Dy]>>[*:1]c1ccccc1",
                  "[#6:1][At,Be,Cd,Dy]>>[*:1]c1ccccc1",
                  "[S:1][At,Be,Cd,Dy]>>[*:1]c1ccccc1"]

        clean = make_cleaner_func(smarts, "1")
        b = Block("[Be:1]c1nc([Be:1])nc(-n2cnc3cc([At])ccc32)n1")
        with self.renv:
            self.assertEquals(clean(b).cleaned().smiles(), "c1ccc(-c2nc(-c3ccccc3)nc(-n3cnc4ccccc43)n2)cc1")
