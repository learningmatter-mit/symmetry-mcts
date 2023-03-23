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

# hide RDKit warnings by setting level to ERROR
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class TestBlockSets(unittest.TestCase):
    def setUp(self):
        def est_func(*blocks):
            mg_finder = re.compile("\[Mg\]")
            smiles = tuple(b.smiles() for b in blocks)
            counts = [len(mg_finder.findall(a)) for a in smiles]
            return reduce(operator.mul, counts)
        self.rxn_env = RDKitReactionEnvironment(LINK_SMARTS,
                                                handles=["Mg"],
                                                estimate_func=est_func)

    def test_react_all_2sets(self):
        set1 = """[Mg]N1C2=CC=C([Be])C=C2C2=C1C=CC([Be])=C2
                    c1([Mg])cccc([Mg])c1
                    [Mg]S(=O)(=O)[Mg]
                    c1nc([Mg])cc([Mg])c1
                    [Mg]C#N
                    """

        bset = BlockSet([Block(smiles=s) for s in set1.split()])
        with self.rxn_env:
            prods = BlockSet()
            for p in bset.react_all(bset):
                prods.add(p)

        for p in prods: print(p.smiles())
        self.assertEquals(len(prods), 99)
