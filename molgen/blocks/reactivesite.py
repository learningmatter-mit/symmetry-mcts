"""
site.py

A teensy class for storing a block and some indices into it
"""

from . import protect


class ReactiveSite:
    """
    a reference
    """
    def __init__(self, block, atom_indices):
        self.block = block
        self.atom_indices = atom_indices

    def __eq__(self, other):
        if self.block != other.block:
            return False
        if set(self.atom_indices) != set(other.atom_indices):
            return False
        return True

    def is_protected(self):
        base_protector = protect.base_protect()
        if base_protector is None:
            return False
        for a in self.atom_indices:
            atom = self.block.mol.GetAtomWithIdx(a)
            if base_protector.is_protected(atom):
                return True
        return False

    def atoms(self):
        return [self.block.mol.GetAtomWithIdx(a) for a in self.atom_indices]

    def __repr__(self):
        atom_names = [self.block.mol.GetAtomWithIdx(a).GetSmarts() for a in self.atom_indices]
        return "<ReactiveSite {}>".format(",".join(atom_names))
