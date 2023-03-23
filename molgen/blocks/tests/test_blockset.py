"""
unittests for blockset

@author Tim Hirzel <hirzel@chemistry.harvard.edu>
"""

import operator
import re
import unittest

from ..block import Block
from ..blockset import BlockSet
from ..reactionenvironment import (RDKitReactionEnvironment,
                                   ReactionEnvironment)

from rdkit import RDLogger

LINK_SMARTS = "[*:1][Mg:3].[*:2][Mg:4]>>[*:1][*:2].[Mg:3][Mg:4]"

SMILE_A = u"c1([Mg])ccccc1"
SMILE_B = "c1cc([Mg])ncc1"
SMILE_C = "c1nc([Mg])cc([Mg])c1"
SMILE_D = "[Mg]C#N"

# hide RDKit warnings by setting level to ERROR
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class TestBlockSets(unittest.TestCase):
    def setUp(self):
        def est_func(*counts):
            total = 1
            for c in counts:
                total *= c
            return total
        self.rxn_env = RDKitReactionEnvironment(LINK_SMARTS,
                                                handles=["Mg"],
                                                estimate_func=est_func)
        self.block_a = Block(smiles=SMILE_A)
        self.block_b = Block(smiles=SMILE_B)
        self.block_c = Block(smiles=SMILE_C)
        self.block_d = Block(smiles=SMILE_D)

    def test_react_one(self):
        expect_list = ["c1ccc(-c2ccccn2)cc1"]
        block_a = Block(smiles=SMILE_A)
        block_b = Block(smiles=SMILE_B)
        with self.rxn_env:
            result = BlockSet(block_a.react(block_b))
        self.assertEquals(expect_list, [b.smiles() for b in result])

    def test_eq(self):
        """
        test that equality works between sets
        """
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_a, self.block_b, self.block_c])
        self.assertEquals(set_a, set_b)

    def test_noteq(self):
        """
        test that equality works between sets
        """
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_a, self.block_c])
        self.assertNotEquals(set_a, set_b)

    def test_sub(self):
        """
        test that equality works between sets
        """
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_a, self.block_c])
        self.assertTrue(set_b < set_a)

    def test_notsub(self):
        """
        test that equality works between sets
        """
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_a, self.block_c, self.block_d])
        self.assertFalse(set_b < set_a)

    def test_react(self):
        expect = ['N#Cc1ccccc1',
                  'N#Cc1ccccn1',
                  'N#Cc1ccnc([Mg])c1',
                  'N#Cc1cc([Mg])ccn1']
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_d])
        with self.rxn_env:
            test = [b.smiles() for b in set_a.react(set_b)]
        # cast to sets so order doesn't matter
        self.assertEquals(set(test), set(expect))

    def test_react_into_blockset(self):
        expect = ['N#Cc1ccccc1',
                  'N#Cc1ccccn1',
                  'N#Cc1ccnc([Mg])c1',
                  'N#Cc1cc([Mg])ccn1']
        expect_blockset = BlockSet([Block(smiles=s) for s in expect])
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_d])
        with self.rxn_env:
            test_blockset = BlockSet()
            for product in set_a.react(set_b):
                test_blockset.add(product)
        # cast to sets so order doesn't matter
        self.assertEquals(test_blockset, expect_blockset)

    def test_estimation(self):
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_d])
        with self.rxn_env:
            total = set_a.estimate(set_b)
        # cast to sets so order doesn't matter
        self.assertEquals(total, 4)

    def test_single_reactant(self):
        class NullMonoReactionEnvironment(ReactionEnvironment):
            def react(self, a):
                return [a]
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        with NullMonoReactionEnvironment("", []):
            test_set = BlockSet(set_a.react())
        self.assertEquals(test_set, set_a)

    def test_three_reactant(self):
        class NullTripleReactionEnvironment(ReactionEnvironment):
            def react(self, a, b, c):
                return [a]
        set_a = BlockSet([self.block_a, self.block_b, self.block_c])
        set_b = BlockSet([self.block_a, self.block_d, self.block_c])
        set_c = BlockSet([self.block_a, self.block_b, self.block_d])

        with NullTripleReactionEnvironment("",[]):
            test_set = BlockSet(set_a.react(set_b, set_c))
        self.assertEquals(test_set, set_a)

    def test_react2(self):
        """
        react_fill two sets
        """
        bset = BlockSet((Block(smiles="c1([Mg])cccc([Mg])c1"),))
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        with self.rxn_env:
            prods = BlockSet(bset.react(reactants))

        expect = """N#Cc1cccc([Mg])c1
                    [Mg]c1cccc(-c2ccnc([Mg])c2)c1
                    [Mg]c1cccc(-c2cc([Mg])ccn2)c1
                    """
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_fill(self):
        """
        react_fill two sets
        """
        bset = BlockSet((Block(smiles="c1([Mg])cccc([Mg])c1"),))
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        with self.rxn_env:
            prods = BlockSet(bset.react_fill(reactants))

        expect = """[Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                    [Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                    N#Cc1cccc(-c2cc([Mg])ccn2)c1
                    [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                    N#Cc1cccc(C#N)c1
                    N#Cc1cccc(-c2ccnc([Mg])c2)c1
                    """
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_fill_2sets(self):
        """
        react_fill two sets with two each
        """
        bset = BlockSet((Block(smiles="c1([Mg])cccc([Mg])c1"),
                         Block(smiles="[Mg]S(=O)(=O)[Mg]")))
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        with self.rxn_env:
            prods = BlockSet(bset.react_fill(reactants))

        expect = """N#CS(=O)(=O)C#N
                    [Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                    N#CS(=O)(=O)c1cc([Mg])ccn1
                    [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                    [Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                    O=S(=O)(c1ccnc([Mg])c1)c1ccnc([Mg])c1
                    N#CS(=O)(=O)c1ccnc([Mg])c1
                    N#Cc1cccc(-c2cc([Mg])ccn2)c1
                    N#Cc1cccc(C#N)c1
                    O=S(=O)(c1cc([Mg])ccn1)c1cc([Mg])ccn1
                    O=S(=O)(c1ccnc([Mg])c1)c1cc([Mg])ccn1
                    N#Cc1cccc(-c2ccnc([Mg])c2)c1
                    """
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_all_2sets(self):
        bset = BlockSet((Block(smiles="c1([Mg])cccc([Mg])c1"),
                         Block(smiles="[Mg]S(=O)(=O)[Mg]")))
        reactants = BlockSet([Block(smiles="c1nc([Mg])cc([Mg])c1"),
                              Block(smiles="[Mg]C#N")])

        with self.rxn_env:
            prods = BlockSet()
            for p in bset.react_all(reactants):
                print(p.smiles())
                prods.add(p)

        expect = """N#CS(=O)(=O)C#N
                [Mg]c1cccc(-c2ccnc([Mg])c2)c1
                [Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                O=S(=O)([Mg])c1ccnc([Mg])c1
                [Mg]c1cccc(-c2cc([Mg])ccn2)c1
                N#CS(=O)(=O)c1cc([Mg])ccn1
                [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                N#CS(=O)(=O)[Mg]
                N#Cc1cccc([Mg])c1
                O=S(=O)(c1ccnc([Mg])c1)c1ccnc([Mg])c1
                N#CS(=O)(=O)c1ccnc([Mg])c1
                O=S(=O)([Mg])c1cc([Mg])ccn1
                N#Cc1cccc(-c2cc([Mg])ccn2)c1
                N#Cc1cccc(C#N)c1
                O=S(=O)(c1cc([Mg])ccn1)c1cc([Mg])ccn1
                O=S(=O)(c1ccnc([Mg])c1)c1cc([Mg])ccn1
                [Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                N#Cc1cccc(-c2ccnc([Mg])c2)c1
                N#CS(=O)(=O)C#N
                [Mg]c1cc(-c2cccc(-c3ccnc([Mg])c3)c2)ccn1
                N#CS(=O)(=O)c1cc([Mg])ccn1
                [Mg]c1ccnc(-c2cccc(-c3ccnc([Mg])c3)c2)c1
                [Mg]c1ccnc(-c2cccc(-c3cc([Mg])ccn3)c2)c1
                O=S(=O)(c1ccnc([Mg])c1)c1ccnc([Mg])c1
                N#CS(=O)(=O)c1ccnc([Mg])c1
                N#Cc1cccc(-c2cc([Mg])ccn2)c1
                N#Cc1cccc(C#N)c1
                O=S(=O)(c1cc([Mg])ccn1)c1cc([Mg])ccn1
                O=S(=O)(c1ccnc([Mg])c1)c1cc([Mg])ccn1
                N#Cc1cccc(-c2ccnc([Mg])c2)c1
                """
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_all_2sets_symmetric_reactants(self):
        bset = BlockSet((Block(smiles="c1([Mg])cccc([Mg])c1"),
                         Block(smiles="[Mg]S(=O)(=O)[Mg]")))
        reactants = BlockSet([Block(smiles="[Mg]N1C2=C(C=CC=C2)N([Mg])C2=C1C=CC=C2"),
                              Block(smiles="[Mg]C#N")])

        with self.rxn_env:
            prods = BlockSet(bset.react_all(reactants))

        for p in prods:
            print(p.smiles())

        expect = """N#CS(=O)(=O)C#N
                    O=S(=O)(N1c2ccccc2N([Mg])c2ccccc21)N1c2ccccc2N([Mg])c2ccccc21
                    O=S(=O)([Mg])N1c2ccccc2N([Mg])c2ccccc21
                    [Mg]N1c2ccccc2N(c2cccc(N3c4ccccc4N([Mg])c4ccccc43)c2)c2ccccc21
                    N#CS(=O)(=O)[Mg]
                    N#Cc1cccc([Mg])c1
                    N#CS(=O)(=O)N1c2ccccc2N([Mg])c2ccccc21
                    [Mg]c1cccc(N2c3ccccc3N([Mg])c3ccccc32)c1
                    N#Cc1cccc(C#N)c1
                    N#Cc1cccc(N2c3ccccc3N([Mg])c3ccccc32)c1
                    """
        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)

    def test_react_set_against_self(self):
        bset = BlockSet((Block(smiles="[Mg]S(=O)(=O)[Mg]"),))

        expect = """O=S(=O)([Mg])S(=O)(=O)S(=O)(=O)[Mg]
                    O=S(=O)([Mg])S(=O)(=O)[Mg]
                 """
        with self.rxn_env:
            prods = BlockSet()
            for p in bset.react_all(bset):
                prods.add(p)

        expect = set(expect.split())
        self.assertEquals(set([p.smiles() for p in prods]), expect)
