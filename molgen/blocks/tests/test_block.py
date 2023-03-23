import unittest

from rdkit import RDLogger
from rdkit.Chem.AllChem import CalcExactMolWt
from rdkit.Chem import (MolToSmiles, MolFromSmiles, SanitizeMol,
                        MolFromInchi, MolToInchi, InchiToInchiKey, Mol)

from ..block import Block
from ..blockset import BlockSet
from ..reactivesite import ReactiveSite
from ..reactionenvironment import RDKitReactionEnvironment


LINK_SMARTS = "[*:1][Mg:3].[*:2][Mg:4]>>[*:1][*:2].[Mg:3][Mg:4]"

SMILE_A = u"c1([Mg])ccccc1"
CLEAN_A = "c1ccccc1"
SMILE_B = "c1cc([Mg])ncc1"

BAD_MATCH = "c1cc([Fe])ncc1"

SMILE_C = "c1nc([Mg])cc([Mg])c1"
SMILE_D = "[Mg][Cl]"

# hide RDKit warnings by setting level to ERROR
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class TestBlock(unittest.TestCase):
    def setUp(self):
        self.rxn_env = RDKitReactionEnvironment(LINK_SMARTS, handles=["Mg"])

    def test_react(self):
        expect_list = ["c1ccc(-c2ccccn2)cc1"]
        block_a = Block(smiles=SMILE_A)
        block_b = Block(smiles=SMILE_B)
        with self.rxn_env:
            results = [b.smiles() for b in block_a.react(block_b)]
            self.assertEquals(results, expect_list)

    def test_parents(self):
        block_a = Block(smiles=SMILE_A)
        block_b = Block(smiles=SMILE_B)
        with self.rxn_env:
            for prod_block in block_a.react(block_b):
                self.assertIn(block_a, prod_block.parents)
                self.assertIn(block_b, prod_block.parents)

    def test_empty(self):
        """
        when nothing can be made, assert that we get a
        StopIteration from the react generator
        """
        block_a = Block(smiles=SMILE_A)
        block_b = Block(smiles=BAD_MATCH)

        with self.assertRaises(StopIteration):
            with self.rxn_env:
                next(block_a.react(block_b))

    def test_equality(self):
        block_a = Block(smiles=SMILE_A)
        block_a2 = Block(smiles=SMILE_A)
        self.assertEquals(block_a, block_a2)

    def test_no_rxn_env(self):
        """
        run a reaction and then leave the environment
        makes sure the reaction does not work
        """
        block_a = Block(smiles=SMILE_A)
        block_b = Block(smiles=SMILE_B)
        with self.rxn_env:
            list(block_a.react(block_b))
        with self.assertRaises(Exception):
            next(block_a.react(block_b))

    def test_clean(self):
        block_a = Block(smiles=SMILE_A)
        with self.rxn_env:
            clean_a = block_a.cleaned()
        self.assertEquals(CLEAN_A, clean_a.smiles())

    def test_clean2(self):
        block_a = Block(smiles="c1([Mg:100])ccccc1")
        with self.rxn_env:
            clean_a = block_a.cleaned()
        self.assertEquals(CLEAN_A, clean_a.smiles())

    def test_remove_markers(self):
        block_a = Block(smiles="c1([Mg:100])ccccc1")
        self.assertEquals(MolToSmiles(MolFromSmiles("[Mg:100]c1ccccc1")), block_a.smiles())
        self.assertEquals(MolToSmiles(MolFromSmiles("[Mg]c1ccccc1")), block_a.smiles(show_markers=False))

    def test_sites(self):
        block_2 = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            sites = block_2.sites()
        expect = [ReactiveSite(block_2, (2, 3)),
                  ReactiveSite(block_2, (5, 6))]
        self.assertEquals(expect, sites)

    def test_sites_2(self):
        # version with two diff reactant sites
        LINK_SMARTS_2 = "[*:1][Mg:3].[*:2][Be:4]>>[*:1][*:2].[Mg:3][Be:4]"
        rxn_env = RDKitReactionEnvironment(LINK_SMARTS_2, handles=["Mg","Be"])
        block_2 = Block(smiles="c1nc([Mg])cc([Be])c1")
        with rxn_env:
            sites_0 = block_2.sites(0)
            sites_1 = block_2.sites(1)
        expect_0 = [ReactiveSite(block_2, (2, 3))]
        expect_1 = [ReactiveSite(block_2, (5, 6))]
        self.assertEquals(sites_0, expect_0)
        self.assertEquals(sites_1, expect_1)



    def test_react_all_not_left_marked(self):
        TEST_SMARTS = "[*:1][Be:3].[*:2][Be:4]>>[*:1][*:2].[Be:3][Be:4]"

        carb = Block("[Mg]N1C2=CC=C([Be])C=C2C2=C1C=CC([Be])=C2")
        benz = Block("[Be]c1cc([Be])cc([Be])c1")

        with RDKitReactionEnvironment(TEST_SMARTS, handles=["Be"]):
            for p in carb.react_all(benz):
                self.assertEquals(p.smiles(show_markers=False), p.smiles(show_markers=True))

    def test_react_fill_not_left_marked(self):
        TEST_SMARTS = "[*:1][Be:3].[*:2][Be:4]>>[*:1][*:2].[Be:3][Be:4]"

        carb = Block("[Mg]N1C2=CC=C([Be])C=C2C2=C1C=CC([Be])=C2")
        benz = Block("[Be]c1cc([Be])cc([Be])c1")

        with RDKitReactionEnvironment(TEST_SMARTS, handles=["Be"]):
            for p in carb.react_fill(benz):
                self.assertEquals(p.smiles(show_markers=False), p.smiles(show_markers=True))


    def test_react_sym(self):
        """
        Test symmetrical reaction with a 2-site core and a 1-site
        block added to it
        """
        block_2 = Block(smiles="c1nc([Mg])cc([Mg])c1")
        block_1 = Block(smiles="[Mg][Cl]")
        with self.rxn_env:
            prods = list(block_2.react_sym(block_1))
        self.assertEquals(len(prods), 1)
        self.assertEquals(prods[0].smiles(), "Clc1ccnc(Cl)c1")

    def test_react_benzene(self):
        """
        react_sym a benzene against itself expect two equal answers.
        plain react would result in 4 answers, but this will
        only permute on the two sites of the right-side benzene
        """
        block_2 = Block(smiles="c1([Mg])cccc([Mg])c1")
        with self.rxn_env:
            prods = list(block_2.react_sym(block_2))
        self.assertEquals(len(prods), 2)
        expect = "[Mg]c1cccc(-c2cccc(-c3cccc([Mg])c3)c2)c1"
        self.assertEquals(prods[0].smiles(), expect)
        self.assertEquals(prods[0], prods[1])

    def test_react_benzene_pyridine(self):
        """
        react_fill a 2-site pyridine onto a 2-site benzene.
        We expect three products, all with both benzene sites
        filled, and with three permutations of the pyridine
        oriented with nitrogen in position: (close, close),
        (far, far), and (close, far)  (see expect smiles if this
        makes no sense :) )
        """
        benzene = Block(smiles="c1([Mg:1])cccc([Mg:2])c1")
        pyridine = Block(smiles="c1nc([Mg:3])cc([Mg:4])c1")
        prods = []
        with self.rxn_env:
            for p in benzene.react(pyridine):
                #print(p.smiles())
                prods.append(p)
        self.assertEquals(len(prods), 2)

        expect = """[Mg]c1cccc(-c2cc([Mg])ccn2)c1
                    [Mg]c1cccc(-c2ccnc([Mg])c2)c1"""
        expect = set(MolToSmiles(MolFromSmiles(i)) for i in expect.split())
        self.assertEquals(set([p.smiles(show_markers=False) for p in prods]), expect)

    def test_react_sym_benzene_pyridine(self):
        """
        react_sym a 2-site benzene and 2-site pyridine,
        we expect two different results, one with the
        the pyridine nitrogen close to benzene, and one
        with nitrogen in the far position
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            prods = list(benzene.react_sym(pyridine))
        self.assertEquals(len(prods), 2)
        expect = set([Block(smiles="[Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1"),
                      Block(smiles="[Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1")])
        self.assertEquals(set(prods), expect)

    def test_react_sym_benzene3_pyridine2(self):
        """
        react_sym a 3-site benzene and 2-site pyridine,
        we expect two different results, one with the
        the pyridine nitrogen close to benzene, and one
        with nitrogen in the far position.

        Three sites on the benzene tests makes this test
        very important that we are indeed forcing reactions
        in the correct sites
        """
        benzene = Block(smiles="c1([Mg])cc([Mg])cc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            prods = list(benzene.react_sym(pyridine))
        self.assertEquals(len(prods), 2)
        expect = set([Block(smiles="[Mg]c1cc(-c2cc(-c3ccnc([Mg])c3)cc(-c3ccnc([Mg])c3)c2)ccn1"),
                      Block(smiles="[Mg]c1ccnc(-c2cc(-c3cc([Mg])ccn3)cc(-c3cc([Mg])ccn3)c2)c1")])

        self.assertEquals(set(prods), expect)

    def test_react_normal_benzene_pyridine_twice(self):
        """
        React once normally a 2-site benzene and 2-site pyridine.
        React the products again with the pyridine.
        This is an example to contrast with react_sym and react_fill.
        We end up with 7 products.
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            prods = BlockSet(benzene.react(pyridine))
            prods = BlockSet(prods.react(BlockSet([pyridine])))

        self.assertEquals(len(prods), 7)
        expect = """[Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                    [Mg]c1cccc(-c2ccnc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                    [Mg]c1cccc(-c2ccnc(-c3ccnc([Mg])c3)c2)c1
                    [Mg]c1cccc(-c2cc(-c3ccnc([Mg])c3)ccn2)c1
                    [Mg]c1cccc(-c2cc(-c3cc([Mg])ccn3)ccn2)c1"""
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_fill_benzene_pyridine(self):
        """
        react_fill a 2-site pyridine onto a 2-site benzene.
        We expect three products, all with both benzene sites
        filled, and with three permutations of the pyridine
        oriented with nitrogen in position: (close, close),
        (far, far), and (close, far)  (see expect smiles if this
        makes no sense :) )
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        prods = []
        with self.rxn_env:
            for p in benzene.react_fill(pyridine):
                # print(p.smiles())
                prods.append(p)
        self.assertEquals(len(prods), 4)

        expect = """[Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                    [Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1"""
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_fill_benzene1(self):
        benzene = Block(smiles="c1cccc([Mg])c1")
        pyridine = Block(smiles="[Mg]C#N")
        prods = []
        with self.rxn_env:
            for p in benzene.react_fill(pyridine):
                #print(p.smiles())
                prods.append(p)
        self.assertEquals(len(prods), 1)

        expect = 'N#Cc1ccccc1'
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_fill_benzene1to2(self):
        benzene = Block(smiles="c1([Mg])ccccc1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        prods = []
        with self.rxn_env:
            for p in benzene.react_fill(pyridine):
                # print(p.smiles())
                prods.append(p)
        self.assertEquals(len(prods), 2)

        expect = """[Mg]c1ccnc(-c2ccccc2)c1
                    [Mg]c1cc(-c2ccccc2)ccn1"""
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_fill_benzene2to1(self):
        benzene = Block(smiles="c1c([Mg])ccc([Mg])c1")
        pyridine = Block(smiles="c1nccc([Mg])c1")
        prods = []
        with self.rxn_env:
            for p in benzene.react_fill(pyridine):
                # print(p.smiles())
                prods.append(p)
        self.assertEquals(len(prods), 1)

        expect = """c1cc(-c2ccc(-c3ccncc3)cc2)ccn1"""
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_all_benzene_pyridine(self):
        """
        react_all a 2-site benzene and 2-site pyridine.

        This is an example to contrast with react_sym and react_fill.
        We end up with 5 products.
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            prods = set(benzene.react_all(pyridine))
        # for p in prods:
        #     print(p.smiles())

        self.assertEquals(len(prods), 5)
        expect = """[Mg]c1cccc(-c2cc([Mg])ccn2)c1
                    [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                    [Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                    [Mg]c1cccc(-c2ccnc([Mg])c2)c1
                    """
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_all_2by2asymmetric(self):
        """
        react_all a 2-site furane and 2-site pyridine.

        This is an example to contrast with react_sym and react_fill.
        We end up with 5 products.
        """
        furane = Block(smiles="[Mg]C1=CC([Mg])=CO1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            prods = set(furane.react_all(pyridine))
        # for p in prods:
        #     print(p.smiles(False))

        self.assertEquals(len(prods), 8)
        expect = """[Mg]c1coc(-c2ccnc([Mg])c2)c1
                    [Mg]c1cc(-c2ccnc([Mg])c2)co1
                    [Mg]c1cc(-c2coc(-c3ccnc([Mg])c3)c2)ccn1
                    [Mg]c1cc(-c2cc([Mg])ccn2)co1
                    [Mg]c1coc(-c2cc([Mg])ccn2)c1
                    [Mg]c1ccnc(-c2cc(-c3ccnc([Mg])c3)co2)c1
                    [Mg]c1ccnc(-c2coc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1ccnc(-c2coc(-c3ccnc([Mg])c3)c2)c1
                    """
        expect = set(MolToSmiles(MolFromSmiles(i)) for i in expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_all_2by_blockset_asymmetric(self):
        """
        react_all a 2-site pyridine against a set of 2-site furane and thiolene.

        This is an example to contrast with react_sym and react_fill.
        We end up with 5 products.
        """
        furane = Block(smiles="[Mg]C1=CC([Mg])=CO1")
        thiofene = Block(smiles="[Mg]C1=CC([Mg])=CS1")

        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            prods = set(pyridine.react_all(BlockSet([furane, thiofene])))
        # for p in prods:
            # print(p.smiles(False))

        self.assertEquals(len(prods), 24)
        expect = """[Mg]c1csc(-c2ccnc(-c3cc([Mg])cs3)c2)c1
                    [Mg]c1csc(-c2cc(-c3csc([Mg])c3)ccn2)c1
                    [Mg]c1coc(-c2ccnc(-c3cc([Mg])cs3)c2)c1
                    [Mg]c1csc(-c2cc(-c3coc([Mg])c3)ccn2)c1
                    [Mg]c1csc(-c2cc([Mg])ccn2)c1
                    [Mg]c1csc(-c2ccnc([Mg])c2)c1
                    [Mg]c1csc(-c2ccnc(-c3csc([Mg])c3)c2)c1
                    [Mg]c1cc(-c2ccnc(-c3csc([Mg])c3)c2)cs1
                    [Mg]c1coc(-c2ccnc(-c3csc([Mg])c3)c2)c1
                    [Mg]c1cc(-c2ccnc(-c3csc([Mg])c3)c2)co1
                    [Mg]c1cc(-c2cc([Mg])ccn2)cs1
                    [Mg]c1cc(-c2ccnc([Mg])c2)cs1
                    [Mg]c1coc(-c2cc(-c3cc([Mg])cs3)ccn2)c1
                    [Mg]c1coc(-c2cc(-c3csc([Mg])c3)ccn2)c1
                    [Mg]c1coc(-c2ccnc(-c3cc([Mg])co3)c2)c1
                    [Mg]c1coc(-c2cc(-c3coc([Mg])c3)ccn2)c1
                    [Mg]c1coc(-c2cc([Mg])ccn2)c1
                    [Mg]c1coc(-c2ccnc([Mg])c2)c1
                    [Mg]c1csc(-c2ccnc(-c3coc([Mg])c3)c2)c1
                    [Mg]c1cc(-c2cc(-c3csc([Mg])c3)ccn2)co1
                    [Mg]c1coc(-c2ccnc(-c3coc([Mg])c3)c2)c1
                    [Mg]c1cc(-c2ccnc(-c3coc([Mg])c3)c2)co1
                    [Mg]c1cc(-c2cc([Mg])ccn2)co1
                    [Mg]c1cc(-c2ccnc([Mg])c2)co1
                    """
        expect = set([Block(s) for s in expect.split()])
        self.assertEquals(prods, expect)



    def test_react_all_2by3asymmetric(self):
        """
        react_all a 2-site furane and 3-site pyridine.

        This is an example to contrast with react_sym and react_fill.
        We end up with 5 products.
        """
        furane = Block(smiles="[Mg]C1=CC([Mg])=CO1")
        pyridine = Block(smiles="c1nc([Mg])c([Mg])c([Mg])c1")
        with self.rxn_env:
            prods = set(furane.react_all(pyridine))
        # for p in prods:
            # print(p.smiles(False))

        self.assertEquals(len(prods), 15)
        expect = """[Mg]c1ccnc([Mg])c1-c1coc(-c2ccnc([Mg])c2[Mg])c1
                    [Mg]c1ccnc([Mg])c1-c1coc(-c2c([Mg])ccnc2[Mg])c1
                    [Mg]c1coc(-c2c([Mg])ccnc2[Mg])c1
                    [Mg]c1cc(-c2ccnc([Mg])c2[Mg])co1
                    [Mg]c1ccnc(-c2coc(-c3c([Mg])ccnc3[Mg])c2)c1[Mg]
                    [Mg]c1coc(-c2ccnc([Mg])c2[Mg])c1
                    [Mg]c1ccnc(-c2cc(-c3ccnc([Mg])c3[Mg])co2)c1[Mg]
                    [Mg]c1cc(-c2nccc([Mg])c2[Mg])co1
                    [Mg]c1ccnc(-c2cc(-c3c([Mg])ccnc3[Mg])co2)c1[Mg]
                    [Mg]c1nccc(-c2coc(-c3ccnc([Mg])c3[Mg])c2)c1[Mg]
                    [Mg]c1coc(-c2nccc([Mg])c2[Mg])c1
                    [Mg]c1ccnc([Mg])c1-c1cc(-c2ccnc([Mg])c2[Mg])co1
                    [Mg]c1ccnc(-c2coc(-c3nccc([Mg])c3[Mg])c2)c1[Mg]
                    [Mg]c1cc(-c2c([Mg])ccnc2[Mg])co1
                    [Mg]c1ccnc(-c2coc(-c3ccnc([Mg])c3[Mg])c2)c1[Mg]
                    """
        expect = set([Block(s) for s in expect.split()])
        self.assertEquals(prods, expect)


    def test_react_all_benzene3_pyridine(self):
        """
        react_all a 3-site benzene and 2-site pyridine.

        This is an example to contrast with react_sym and react_fill.
        We end up with ? products.
        """
        benzene = Block(smiles="c1([Mg])cc([Mg])cc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            prods = BlockSet(benzene.react_all(pyridine))

        # for p in prods:
        #     print(p.smiles())
        self.assertEquals(len(prods), 9)
        expect = """[Mg]c1cc(-c2cc(-c3ccnc([Mg])c3)cc(-c3ccnc([Mg])c3)c2)ccn1
                    [Mg]c1cc([Mg])cc(-c2ccnc([Mg])c2)c1
                    [Mg]c1ccnc(-c2cc([Mg])cc([Mg])c2)c1
                    [Mg]c1ccnc(-c2cc(-c3ccnc([Mg])c3)cc(-c3ccnc([Mg])c3)c2)c1
                    [Mg]c1ccnc(-c2cc([Mg])cc(-c3ccnc([Mg])c3)c2)c1
                    [Mg]c1ccnc(-c2cc(-c3ccnc([Mg])c3)cc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1cc(-c2ccnc([Mg])c2)cc(-c2ccnc([Mg])c2)c1
                    [Mg]c1ccnc(-c2cc([Mg])cc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1ccnc(-c2cc(-c3cc([Mg])ccn3)cc(-c3cc([Mg])ccn3)c2)c1
                    """
        expect = set(MolToSmiles(MolFromSmiles(i)) for i in expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_fill_with_set(self):
        """
        react_fill two sets
        """
        b = Block(smiles="c1([Mg])cccc([Mg])c1")
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        with self.rxn_env:
            prods = BlockSet(b.react_fill(reactants))

        expect = """[Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                    N#Cc1cccc(-c2cc([Mg])ccn2)c1
                    [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                    N#Cc1cccc(C#N)c1
                    N#Cc1cccc(-c2ccnc([Mg])c2)c1
                    """
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_all_with_set(self):
        """
        react_fill two sets
        """
        b = Block(smiles="c1([Mg])cccc([Mg])c1")
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        with self.rxn_env:
            prods = BlockSet(b.react_all(reactants))

        expect = """N#Cc1cccc([Mg])c1
                    [Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                    N#Cc1cccc(-c2cc([Mg])ccn2)c1
                    [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                    [Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1cccc(-c2cc([Mg])ccn2)c1
                    N#Cc1cccc(C#N)c1
                    [Mg]c1cccc(-c2ccnc([Mg])c2)c1
                    N#Cc1cccc(-c2ccnc([Mg])c2)c1
                    """

        # for p in prods:
        #     print(p.smiles())
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react2(self):
        b1 = Block("FC1=CC=C(c2sc3c(c2F)C2(OCCO2)c2c(F)c([Mg])sc2-3)[SiH2]1")
        b2 = Block("[Mg]c1ccc([Mg])c2nsnc21")
        with self.rxn_env:
            canonical_product = MolToSmiles(MolFromSmiles("FC1=CC=C(c2sc3c(c2F)C2(OCCO2)c2c(F)c(-c4ccc([Mg])c5nsnc54)sc2-3)[SiH2]1"))
            self.assertEquals(canonical_product,
                              next(b1.react(b2)).smiles())

    def test_react3(self):
        b1 = Block("FC1=CC=C(c2sc3c(c2F)C2(OCCO2)c2c(F)c([Mg])sc2-3)[SiH2]1")
        b2 = BlockSet([Block("[Mg]c1ccc([Mg])c2nsnc21")])
        with self.rxn_env:
            canonical_product = MolToSmiles(MolFromSmiles("FC1=CC=C(c2sc3c(c2F)C2(OCCO2)c2c(F)c(-c4ccc([Mg])c5nsnc54)sc2-3)[SiH2]1"))
            self.assertEquals(canonical_product,
                              next(b1.react(b2)).smiles())

    def test_parents_after_react_all(self):
        furane = Block(smiles="[Mg]C1=CC([Mg])=CO1")
        pyridine = Block(smiles="c1nc([Mg])c([Mg])c([Mg])c1")
        with self.rxn_env:
            prods = list(furane.react_all(pyridine))
        for p in prods:
            self.assertEquals(p.parents, [furane, pyridine])
            self.assertEquals(p.generation, 1)

    def test_parents_after_react_fill(self):
        furane = Block(smiles="[Mg]C1=CC([Mg])=CO1")
        pyridine = Block(smiles="c1nc([Mg])c([Mg])c([Mg])c1")
        with self.rxn_env:
            prods = list(furane.react_fill(pyridine))
        for p in prods:
            self.assertEquals(p.parents, [furane, pyridine])
            self.assertEquals(p.generation, 1)

    def test_parents_after_react_sym(self):
        furane = Block(smiles="[Mg]C1=CC([Mg])=CO1")
        pyridine = Block(smiles="c1nc([Mg])c([Mg])c([Mg])c1")
        with self.rxn_env:
            prods = list(furane.react_sym(pyridine))
        for p in prods:
            self.assertEquals(p.parents, [furane, pyridine])
            self.assertEquals(p.generation, 1)

    def test_parents_after_clean(self):
        furane = Block(smiles="[Mg]C1=CC([Mg])=CO1")
        pyridine = Block(smiles="c1nc([Mg])c([Mg])c([Mg])c1")
        with self.rxn_env:
            prods = list(furane.react(pyridine))
            for p in prods:
                self.assertEquals(p.cleaned().parents, [furane.cleaned(), pyridine.cleaned()])
                self.assertEquals(p.generation, 1)

    def test_parents_after_blockset_react_all(self):
        b = Block(smiles="c1([Mg])cccc([Mg])c1")
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        expect = {"N#Cc1cccc([Mg])c1": ["[Mg]c1cccc([Mg])c1", "N#C[Mg]"],
                  "N#Cc1cccc(C#N)c1": ["[Mg]c1cccc([Mg])c1", "N#C[Mg]"],
                  "N#Cc1cccc(-c2cc([Mg])ccn2)c1": ["[Mg]c1cccc([Mg])c1", "N#C[Mg]", "[Mg]c1ccnc([Mg])c1"],
                  "N#Cc1cccc(-c2ccnc([Mg])c2)c1": ["[Mg]c1cccc([Mg])c1", "N#C[Mg]", "[Mg]c1ccnc([Mg])c1"],
                  "[Mg]c1cccc(-c2cc([Mg])ccn2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "N#Cc1cccc(-c2cc([Mg])ccn2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1", "N#C[Mg]"],
                  "[Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "[Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "[Mg]c1cccc(-c2ccnc([Mg])c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "N#Cc1cccc(-c2ccnc([Mg])c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1", "N#C[Mg]"],
                  "[Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "[Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  }

        with self.rxn_env:
            for i, p in enumerate(b.react_all(reactants)):
                self.assertEquals(set([par.smiles() for par in p.parents]), set(expect[p.smiles()]))

    def test_parents_after_blockset_react_fill(self):
        b = Block(smiles="c1([Mg])cccc([Mg])c1")
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        expect = {"N#Cc1cccc(C#N)c1": ["[Mg]c1cccc([Mg])c1", "N#C[Mg]"],
                  "N#Cc1cccc(-c2ccnc([Mg])c2)c1": ["[Mg]c1cccc([Mg])c1", "N#C[Mg]", "[Mg]c1ccnc([Mg])c1"],
                  "N#Cc1cccc(-c2cc([Mg])ccn2)c1": ["[Mg]c1cccc([Mg])c1", "N#C[Mg]", "[Mg]c1ccnc([Mg])c1"],
                  "N#Cc1cccc(-c2ccnc([Mg])c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1", "N#C[Mg]"],
                  "[Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "[Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "N#Cc1cccc(-c2cc([Mg])ccn2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1", "N#C[Mg]"],
                  "[Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"],
                  "[Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1": ["[Mg]c1cccc([Mg])c1", "[Mg]c1ccnc([Mg])c1"]}

        with self.rxn_env:
            for i, p in enumerate(b.react_fill(reactants)):
                self.assertEquals(set([par.smiles() for par in p.parents]), set(expect[p.smiles()]))
                self.assertEquals(p.generation, 1)

    def test_inchi(self):
        b = Block(inchi="InChI=1S/C32H22N4O/c1-3-11-23(12-4-1)31-33-34-32(37-31)24-19-21-26(22-20-24)36-29-17-9-7-15-27(29)35(25-13-5-2-6-14-25)28-16-8-10-18-30(28)36/h1-22H")
        self.assertEquals(b.smiles(), "c1ccc(-c2nnc(-c3ccc(N4c5ccccc5N(c5ccccc5)c5ccccc54)cc3)o2)cc1")

    def test_empty_after_bad_react_sym(self):
        renv = RDKitReactionEnvironment("([*:1][Cd:3]).([*:2][At:4])>>[*:1][*:2].[Cd:3][At:4]", handles=["At", "Be", "Cd", "Dy"])
        LEFT = "[Be]c1ccc2c(c1)c1cc([Be])ccc1n2-c1cc([Dy])cc([Dy])c1"
        RIGHT = "[Be]c1nnc([Be])n1[At]"
        lb = Block(LEFT)
        rb = Block(RIGHT)
        with renv:
            prods = list(lb.react_sym(rb))
        self.assertEquals(len(prods), 0)

    def test_empty_after_bad_react(self):
        renv = RDKitReactionEnvironment("([*:1][Cd:3]).([*:2][At:4])>>[*:1][*:2].[Cd:3][At:4]", handles=["At", "Be", "Cd", "Dy"])
        LEFT = "[Be]c1ccc2c(c1)c1cc([Be])ccc1n2-c1cc([Dy])cc([Dy])c1"
        RIGHT = "[Be]c1nnc([Be])n1[At]"
        lb = Block(LEFT)
        rb = Block(RIGHT)
        with renv:
            prods = list(lb.react(rb))
        self.assertEquals(len(prods), 0)

    def test_empty_after_bad_react_fill(self):
        renv = RDKitReactionEnvironment("([*:1][Cd:3]).([*:2][At:4])>>[*:1][*:2].[Cd:3][At:4]", handles=["At", "Be", "Cd", "Dy"])
        LEFT = "[Be]c1ccc2c(c1)c1cc([Be])ccc1n2-c1cc([Dy])cc([Dy])c1"
        RIGHT = "[Be]c1nnc([Be])n1[At]"
        lb = Block(LEFT)
        rb = Block(RIGHT)
        with renv:
            prods = list(lb.react_fill(rb))
        self.assertEquals(len(prods), 0)

    def test_react_against_self(self):
        b = Block(smiles="[Mg]S(=O)(=O)[Mg]")

        expect = """O=S(=O)([Mg])S(=O)(=O)S(=O)(=O)[Mg]
                    O=S(=O)([Mg])S(=O)(=O)[Mg]
                 """
        with self.rxn_env:
            prods = BlockSet()
            for p in b.react_all(b):
                prods.add(p)

        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_quinone_react_all(self):

        def compare_weight(a,b):
            return cmp(CalcExactMolWt(a.mol), CalcExactMolWt(b.mol))

        smarts_str = "[*:1][He:3].[*:2][He:4]>>[*:1][*:2].[He:3][He:4]"
        combination_env = RDKitReactionEnvironment(smarts_str, ["He"])
        beta = Block("[He][Ne]")
        reactant = Block("Oc1c([He])c([He])c(O)c2c([He])c([He])c([He])c([He])c12")

        expect = ["[Ne]c1ccc2c(O)ccc(O)c2c1",
                  "[Ne]c1cccc2c(O)ccc(O)c12",
                  "[Ne]c1cc(O)c2ccccc2c1O",
                  "[Ne]c1c([Ne])c(O)c2ccccc2c1O",
                  "[Ne]c1cccc2c(O)c([Ne])cc(O)c12",
                  "[Ne]c1cccc2c(O)cc([Ne])c(O)c12",
                  "[Ne]c1ccc2c(O)c([Ne])cc(O)c2c1",
                  "[Ne]c1ccc2c(O)cc([Ne])c(O)c2c1",
                  "[Ne]c1cc([Ne])c2c(O)ccc(O)c2c1",
                  "[Ne]c1ccc2c(O)ccc(O)c2c1[Ne]",
                  "[Ne]c1ccc([Ne])c2c(O)ccc(O)c12",
                  "[Ne]c1cc2c(O)ccc(O)c2cc1[Ne]",
                  "[Ne]c1ccc2c(O)c([Ne])c([Ne])c(O)c2c1",
                  "[Ne]c1ccc2c(O)cc([Ne])c(O)c2c1[Ne]",
                  "[Ne]c1cc2c(O)cc([Ne])c(O)c2cc1[Ne]",
                  "[Ne]c1ccc([Ne])c2c(O)c([Ne])cc(O)c12",
                  "[Ne]c1ccc2c(O)c([Ne])cc(O)c2c1[Ne]",
                  "[Ne]c1cc([Ne])c2c(O)c([Ne])cc(O)c2c1",
                  "[Ne]c1cc([Ne])c2c(O)cc([Ne])c(O)c2c1",
                  "[Ne]c1cc([Ne])c2c(O)ccc(O)c2c1[Ne]",
                  "[Ne]c1cccc2c(O)c([Ne])c([Ne])c(O)c12",
                  "[Ne]c1cc2c(O)ccc(O)c2c([Ne])c1[Ne]",
                  "[Ne]c1ccc2c(O)c([Ne])c([Ne])c(O)c2c1[Ne]",
                  "[Ne]c1cc2c(O)cc([Ne])c(O)c2c([Ne])c1[Ne]",
                  "[Ne]c1cc2c(O)c([Ne])c([Ne])c(O)c2cc1[Ne]",
                  "[Ne]c1c([Ne])c([Ne])c2c(O)ccc(O)c2c1[Ne]",
                  "[Ne]c1cc([Ne])c2c(O)cc([Ne])c(O)c2c1[Ne]",
                  "[Ne]c1cc2c(O)c([Ne])cc(O)c2c([Ne])c1[Ne]",
                  "[Ne]c1cc([Ne])c2c(O)c([Ne])c([Ne])c(O)c2c1",
                  "[Ne]c1ccc([Ne])c2c(O)c([Ne])c([Ne])c(O)c12",
                  "[Ne]c1cc([Ne])c2c(O)c([Ne])cc(O)c2c1[Ne]",
                  "[Ne]c1cc([Ne])c2c(O)c([Ne])c([Ne])c(O)c2c1[Ne]",
                  "[Ne]c1cc(O)c2c([Ne])c([Ne])c([Ne])c([Ne])c2c1O",
                  "[Ne]c1cc2c(O)c([Ne])c([Ne])c(O)c2c([Ne])c1[Ne]",
                  "[Ne]c1c([Ne])c([Ne])c2c(O)c([Ne])c([Ne])c(O)c2c1[Ne]"]

        combined = BlockSet()
        with combination_env:
            i = 0
            for prod in reactant.react_all(beta):
                print(i)
                i += 1
                combined.add(prod.cleaned())

        # for i in sorted(list(combined), compare_weight):
            # print(i.smiles())

        for expected in expect:
            self.assertIn(Block(expected), combined)

    def test_fixed_hydrogens(self):
        smiles_list = ['C1CN=C[OH+]1',
                       'C1C[NH+]=CO1']
        inchis_fixed_H = ['InChI=1/C3H5NO/c1-2-5-3-4-1/h3H,1-2H2/p+1/fC3H6NO/h5H/q+1',
                          'InChI=1/C3H5NO/c1-2-5-3-4-1/h3H,1-2H2/p+1/fC3H6NO/h4H/q+1']
        inchikey_fixed_H = ["IMSODMZESSGVBE-DTWDKUFRNA-O",
                             "IMSODMZESSGVBE-NKNRQFNBNA-O"]

        for i, smile in enumerate(smiles_list):
            mol = Block(smile)
            self.assertEquals(mol.inchi(), inchis_fixed_H[i])
            self.assertEquals(mol.inchikey, inchikey_fixed_H[i])

    def test_metal_connection(self):
        smiles_list = ['[Li]C1=C([Li])C=C([Na])C([Na])=C1',
                       '[Li]C1=CC([Li])=C([Na])C=C1[Na]']
        inchis_rec_metal = ['InChI=1/C6H2.2Li.2Na/c1-2-4-6-5-3-1;;;;/h1,6H;;;;/rC6H2Li2Na2/c7-3-1-5(9)6(10)2-4(3)8/h1-2H',
                            'InChI=1/C6H2.2Li.2Na/c1-2-4-6-5-3-1;;;;/h1,6H;;;;/rC6H2Li2Na2/c7-3-1-4(8)6(10)2-5(3)9/h1-2H']
        inchikey_rec_metal = ["WEJQLDLGZBJGBT-MEFSQGSDNA-N",
                               "WEJQLDLGZBJGBT-UBHHOOLBNA-N"]

        for i, smile in enumerate(smiles_list):
            mol = Block(smile)
            self.assertEquals(mol.inchi(), inchis_rec_metal[i])
            self.assertEquals(mol.inchikey, inchikey_rec_metal[i])

    def test_isomeric_smiles(self):
        smiles_list = ['C(=C\c1ccccc1)\c1ccccc1',
                       'C(=C/c1ccccc1)\c1ccccc1',
                       'C(=Cc1ccccc1)c1ccccc1']
        inchikeys = ['PJANXHGTPQOBST-QXMHVHEDNA-N',
                      'PJANXHGTPQOBST-VAWYXSNFNA-N',
                      'PJANXHGTPQOBST-UHFFFAOYNA-N']

        for i, smile in enumerate(smiles_list):
            mol = Block(smile)
            self.assertEquals(mol.smiles(), smiles_list[i])
            self.assertEquals(mol.inchikey, inchikeys[i])

if __name__ == "__main__":
    unittest.main()
