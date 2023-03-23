import unittest

from rdkit import RDLogger

from blocks.block import Block
from ..multismarts_rxnenv import MultiSmartsRxnEnv, MultiSmartsSingleRxnEnv

# hide RDKit warnings by setting level to ERROR
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

MALE_ATOM = "Mg"
FEMALE_ATOM = "Fe"
BI_ATOM = "Be"


def smarts(left_gender_atom, right_gender_atom):
    base = "[*:1][{lga}:3].[*:2][{rga}:4]" + \
           ">>[*:1][*:2].[{lga}:3][{rga}:4]"
    return base.format(lga=left_gender_atom, rga=right_gender_atom)


def test_smarts():
    atom_pairs = [(BI_ATOM, BI_ATOM),
                  (MALE_ATOM, MALE_ATOM),
                  ]
    return [smarts(l, r) for l, r in atom_pairs]


def link_donor_smarts():
    atom_pairs = [(BI_ATOM, BI_ATOM),
                  (BI_ATOM, MALE_ATOM),
                  ]
    return [smarts(l, r) for l, r in atom_pairs]


class TestMultiSmarts(unittest.TestCase):
    def setUp(self):
        self.rxn_env = MultiSmartsRxnEnv(test_smarts(), [BI_ATOM, MALE_ATOM])

    def test_rxn(self):
        SMILE = "c1nc([Be])cc([Mg])c1"
        SMILE_B = "[Be][Cl]"
        SMILE_M = "[Mg][Cl]"

        prod1 = self.rxn_env.react(Block(SMILE), Block(SMILE_B))
        self.assertEquals(len(prod1), 1)
        self.assertEquals(prod1[0], Block("c1nc([Cl])cc([Mg])c1"))

        prod2 = self.rxn_env.react(Block(SMILE), Block(SMILE_M))
        self.assertEquals(len(prod2), 1)
        self.assertEquals(prod2[0], Block("c1nc([Be])cc([Cl])c1"))


class TestRealSmarts(unittest.TestCase):
    def setUp(self):
        self.rxn_env = MultiSmartsRxnEnv(link_donor_smarts(), [BI_ATOM, MALE_ATOM])

    def test_carbazole(self):
        CARB_SMILES = "[Mg]N1C2=CC=C([Be])C=C2C2=C1C=CC([Be])=C2"
        carb = Block(CARB_SMILES)
        prod1 = set(self.rxn_env.react(carb, carb))
        expect = """[Be]c1ccc2c(c1)c1cc(-c3ccc4c(c3)c3cc([Be])ccc3n4[Mg])ccc1n2[Mg]
                    [Be]c1ccc2c(c1)c1cc(-n3c4ccc([Be])cc4c4cc([Be])ccc43)ccc1n2[Mg]"""
        self.assertEquals(prod1, set([Block(s) for s in expect.split()]))

    def test_carbazole_2(self):
        CARB_SMILES = "[Mg]N1C2=CC=C([Be])C=C2C2=C1C=CC([Be])=C2"
        carb = Block(CARB_SMILES)
        with self.rxn_env:
            prod1 = set(carb.react(carb))
        expect = """[Be]c1ccc2c(c1)c1cc(-c3ccc4c(c3)c3cc([Be])ccc3n4[Mg])ccc1n2[Mg]
                    [Be]c1ccc2c(c1)c1cc(-n3c4ccc([Be])cc4c4cc([Be])ccc43)ccc1n2[Mg]"""
        self.assertEquals(prod1, set([Block(s) for s in expect.split()]))

    def test_carbazole_react_fill(self):
        CARB_SMILES = "[Mg]N1C2=CC=C([Be])C=C2C2=C1C=CC([Be])=C2"
        carb = Block(CARB_SMILES)
        # import ipdb; ipdb.set_trace()
        with self.rxn_env:
            prod1 = set(carb.react_fill(carb))

        expect = """
            [Be]c1ccc2c(c1)c1cc(-c3ccc4c(c3)c3cc(-c5ccc6c(c5)c5cc([Be])ccc5n6[Mg])ccc3n4[Mg])ccc1n2[Mg]
            [Be]c1ccc2c(c1)c1cc(-c3ccc4c(c3)c3cc(-n5c6ccc([Be])cc6c6cc([Be])ccc65)ccc3n4[Mg])ccc1n2[Mg]
            [Be]c1ccc2c(c1)c1cc([Be])ccc1n2-c1ccc2c(c1)c1cc(-n3c4ccc([Be])cc4c4cc([Be])ccc43)ccc1n2[Mg]"""
        self.assertEquals(prod1, set([Block(s) for s in expect.split()]))


class TestSpecialCleaner(unittest.TestCase):
    def setUp(self):
        self.special_cleaner = MultiSmartsSingleRxnEnv(["[#7,O,S:1][Mg,Be,Fe]>>[*:1]C", "[C:1](=O)[Be]>>[*:1](=O)C"],
                                                       [BI_ATOM, MALE_ATOM, FEMALE_ATOM])

    def test_clean1(self):
        dirty_smiles = "[Be]c1ccc(N([Be])c2nnc(-n3c4ccccc4c(=O)c4ccccc43)n2N2c3ccccc3N([Be])c3ccccc32)cc1"
        dirty_block = Block(dirty_smiles)
        with self.special_cleaner:
            prods = list(dirty_block.react_fill())

        expect = "[Be]c1ccc(N(C)c2nnc(-n3c4ccccc4c(=O)c4ccccc43)n2N2c3ccccc3N(C)c3ccccc32)cc1"
        self.assertEquals(len(prods), 1)
        self.assertEquals(prods[0].smiles(),  expect)
