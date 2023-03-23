from django.test import TestCase

from blocks.block import Block
from blocks.reactionenvironment import RDKitReactionEnvironment

from ..utils import put_block

from pgmols.models import Mol

PROJECT = 'test_project'

class TestUtils(TestCase):
    def setUp(self):
        link_smarts = "[*:1][Mg:3].[*:2][Mg:4]>>[*:1][*:2].[Mg:3][Mg:4]"
        self.rxn_env = RDKitReactionEnvironment(link_smarts, handles=["Mg"])

    def test_put_block(self):
        block_2 = Block(smiles="c1nc([Mg])cc([Mg])c1")
        block_1 = Block(smiles="[Mg][Cl]")
        with self.rxn_env:
            prods = list(block_2.react_sym(block_1))
            prods = [p.cleaned(clean_parents=True) for p in prods]
            clean_parents = [blk.cleaned() for blk in [block_1, block_2]]
            for blk in clean_parents:
                put_block(blk, PROJECT)
        self.assertEquals(len(prods), 1)
        self.assertEquals(prods[0].smiles(), "Clc1ccnc(Cl)c1")
        put_block(prods[0], PROJECT)

        child_mol = Mol.objects.get(inchikey=prods[0].inchikey)
        parent_mols = child_mol.parents.all()
        parent_keys = set([p.inchikey for p in parent_mols])
        expect = set([p.inchikey for p in clean_parents])
        self.assertEquals(expect, parent_keys)
        self.assertEquals(child_mol.details['generation'], 1)
        self.assertEquals(parent_mols[0].details['generation'], 0)
        self.assertEquals(parent_mols[1].details['generation'], 0)
