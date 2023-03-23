import os
import unittest

from rdkit import RDLogger

from ..blockset_storage_json import BlockSetStorageJSON

# hide RDKit warnings by setting level to ERROR
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

frag_filepath = os.path.dirname(__file__)


class TestJSONStorage(unittest.TestCase):

    def setUp(self):
        self.storage = BlockSetStorageJSON(os.path.join(frag_filepath, "fragments_small.json"))

    def test_storage(self):
        expect = ["[Be]c1nnc([Be])n1[At]"]

        acceptor_gen = self.storage.filter(groups=["acceptor"])
        smiles = [b.smiles() for b in acceptor_gen]
        self.assertEquals(smiles, expect)
