"""
unittests for protect

@author Tim Hirzel <hirzel@chemistry.harvard.edu>
"""
import unittest

from rdkit import RDLogger

from ..block import Block
from ..blockset import BlockSet
from ..reactionenvironment import RDKitReactionEnvironment
from ..protect import ExposeOnly, Protect

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


class TestProtect(unittest.TestCase):
    def setUp(self):
        self.rxn_env = RDKitReactionEnvironment(LINK_SMARTS, handles=["Mg"])

    def test_protect(self):
        """
        take a 2 site block_2 and react against a single block,
        limiting the reaction to one site using ExposeOnly
        and Protect
        """
        block_2 = Block(smiles="c1nc([Mg])cc([Mg])c1")
        block_1 = Block(smiles="[Mg][Cl]")

        with self.rxn_env:
            sites = block_2.sites()
            with ExposeOnly(sites[0]):
                prods1 = list(block_2.react(block_1))
                self.assertEquals(len(prods1), 1)
            with Protect(sites[1]):
                prods2 = list(block_2.react(block_1))
                self.assertEquals(len(prods2), 1)
            # these should be both end up with the same reaction
            self.assertEquals(set(prods1), set(prods2))
            self.assertEquals(Block(smiles="[Mg]c1ccnc(Cl)c1"), prods1[0])
            self.assertEquals("[Mg]c1ccnc(Cl)c1", prods1[0].smiles())

    def test_protect_shuts_off(self):
        """
        take a 2 site block_2 and react against a single block,
        limiting the reaction to one site using ExposeOnly
        and Protect
        """
        block_2 = Block(smiles="c1nc([Mg])cc([Mg])c1")
        block_1 = Block(smiles="[Mg][Cl]")

        with self.rxn_env:
            sites = block_2.sites()
            with ExposeOnly(sites[0]):
                prods1 = BlockSet(block_2.react(block_1))
                self.assertEquals(len(prods1), 1)

            prods2 = list(prods1.react(block_1))
            self.assertEquals(len(prods2), 1)
            # these should be both end up with the same reaction
            self.assertEquals(Block(smiles="c1nc(Cl)cc(Cl)c1"), prods2[0])
            self.assertEquals("[Mg]c1ccnc(Cl)c1", prods1.pop().smiles())

    def test_cross_nested_protect_react(self):
        """
        take a 2 site pyridine and react against a 2 site benzene,
        protect one site from each block and expect a single product
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")

        with self.rxn_env:
            bsites = benzene.sites()
            psites = pyridine.sites(1)
            with ExposeOnly(bsites[0], psites[0]):
                prods1 = list(benzene.react(pyridine))
                self.assertEquals(len(prods1), 1)

            self.assertEquals('[Mg]c1cccc(-c2cc([Mg])ccn2)c1', prods1[0].smiles())

    def test_cross_nested_protect_react2(self):
        """
        take a 2 site pyridine and react against a 2 site benzene,
        protect one site from each block and expect a single product
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")

        with self.rxn_env:
            bsites = benzene.sites()
            psites = pyridine.sites(1)
            with ExposeOnly(bsites[0]):
                with ExposeOnly(psites[0]):
                    prods1 = list(benzene.react(pyridine))
                    self.assertEquals(len(prods1), 1)

                prods2 = list(benzene.react(pyridine))
                self.assertEquals(len(prods2), 2)

            self.assertEquals('[Mg]c1cccc(-c2cc([Mg])ccn2)c1', prods1[0].smiles())

    def test_protect_push_pop(self):
        """
        take a 2 site block_2 and react against a single block
        with one site protected, expect 1 back.  Protect the
        other site, and expect no results.  exit the
        inner Protect block and expect 1 result again.
        """
        block_2 = Block(smiles="c1nc([Mg])cc([Mg])c1")
        block_1 = Block(smiles="[Mg][Cl]")

        with self.rxn_env:
            sites = block_2.sites()
            with Protect(sites[0]):
                prods1 = list(block_2.react(block_1))
                self.assertEquals(len(prods1), 1)
                with Protect(sites[1]):
                    prods2 = list(block_2.react(block_1))
                    self.assertEquals(len(prods2), 0)
                prods3 = list(block_2.react(block_1))
                self.assertEquals(len(prods3), 1)

    def test_is_protected(self):
        """
        test is_protected call when inside ExposeOnly
        """
        block_2 = Block(smiles="c1nc([Mg])cc([Mg])c1")
        with self.rxn_env:
            sites = block_2.sites()
            with ExposeOnly(sites[0]):
                self.assertTrue(block_2.sites()[1].is_protected())
                self.assertFalse(block_2.sites()[0].is_protected())
            self.assertFalse(block_2.sites()[1].is_protected())
            self.assertFalse(block_2.sites()[0].is_protected())

    def test_protect2(self):
        """
        Take a 2 site pyridine and react against a 2-site benzene.
        Test that ExposeOnly on one of the benzene sites
        produces the same result of a Protect call to the other
        site of the benzene
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")

        with self.rxn_env:
            sites = benzene.sites()
            with ExposeOnly(sites[0]):
                prods1 = list(benzene.react(pyridine))
                self.assertEquals(len(prods1), 2)
            with Protect(sites[1]):
                prods2 = list(benzene.react(pyridine))
                self.assertEquals(len(prods2), 2)
            # these should be both end up with the same reaction
            self.assertEquals(set(prods1), set(prods2))
            test_smiles = [prod.smiles() for prod in prods1]
            self.assertIn('[Mg]c1cccc(-c2ccnc([Mg])c2)c1', test_smiles)
            self.assertIn('[Mg]c1cccc(-c2cc([Mg])ccn2)c1', test_smiles)

    def test_molAtomMapNumbers(self):
        """
        test that molAtomMapNumbers survive Protect calls
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")

        with self.rxn_env:
            psites = pyridine.sites()
            for i, a in enumerate(pyridine.atoms()):
                a.SetProp("molAtomMapNumber", str(i+1))
            with ExposeOnly(psites[0]):
                prods1 = list(benzene.react(pyridine))
            for prod in prods1:
                markers = [int(a.GetProp("molAtomMapNumber")) for a in prod.atoms() if a.HasProp("molAtomMapNumber")]
                self.assertEquals(set(markers), set([1, 2, 5, 6, 7, 8]))

    def test_nest_tuple_protect(self):
        """
        take a 2 site pyridine and react against a 2 site benzene,
        test the use of ExposeOnly with a premade tuple of sites
        """
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        pyridine = Block(smiles="c1nc([Mg])cc([Mg])c1")

        with self.rxn_env:
            bsites = benzene.sites()
            psites = pyridine.sites()
            limits = [bsites[0]]
            limits.append(psites[0])
            with ExposeOnly(*limits):
                prods1 = list(benzene.react(pyridine))
                self.assertEquals(len(prods1), 1)
            self.assertEquals('[Mg]c1cccc(-c2cc([Mg])ccn2)c1', prods1[0].smiles())

    def test_protect_products_return_without_markers(self):
        benzene = Block(smiles="c1([Mg])cccc([Mg])c1")
        benzene2 = Block(smiles="c1([Mg])cccc([Mg])c1")

        with self.rxn_env:
            bsites = benzene.sites()
            with ExposeOnly(bsites[0]):
                prods = list(benzene.react_all(benzene2))
        for p in prods:
            self.assertEquals(p.smiles(show_markers=False), p.smiles(show_markers=True))