"""
blockset.py

@version 0.1
@since Feb-2014
@author Tim Hirzel <hirzel@chemistry.harvard.edu>
@author Ed Pyzer-Knapp <e.o.pyzerknapp@gmail.com>
"""

import itertools
from . import reactionenvironment


class BlockSet(set):
    """
    This is a set of Block objects, which allows reactions with other
    BlockSets.
    """

    def react_with_self(self):
        """
        A generator that yields this blockset reacting with itself.
        Use only in commutative (orderless) reaction environments where
        A.react(B) == B.react(A)
        """
        for left, right in itertools.combinations(self, 2):
            yield left.react(right)

    def react(self, *others):
        """
        Reacts this BlockSet against n other BlockSets.
        The number of allowed blocksets is determined by the current reaction environment
        """
        for mine in self:
            for result in mine.react(*others):
                yield result

    def react_fill(self, *others):
        """
        Reacts this BlockSet against n other BlockSets.
        The number of allowed blocksets is determined by the current reaction environment
        """
        for mine in self:
            for result in mine.react_fill(*others):
                yield result

    def react_sym(self, *others):
        """
        Reacts this BlockSet against n other BlockSets.
        The number of allowed blocksets is determined by the current reaction environment
        """
        for mine in self:
            for result in mine.react_sym(*others):
                yield result

    def react_all(self, *others):
        """
        Reacts this BlockSet against n other BlockSets.
        The number of allowed blocksets is determined by the current reaction environment
        """
        for mine in self:
            for result in mine.react_all(*others):
                yield result

    def cleaned(self):
        """
        generator that yields cleaned versions of all the blocks in self
        """
        for block in self:
            yield block.cleaned()

    def estimate(self, *others):
        """
        get estimate of number of products from reacting with n other blocksets
        in the current reaction environment
        """
        totals = [blk.estimate(*others) for blk in self]
        return sum(totals)

    def estimate_sym(self, *others):
        """
        get estimate of number of products from symmetrically reacting with n other blocksets
        in the current reaction environment
        """
        totals = [blk.estimate_sym(*others) for blk in self]
        return sum(totals)


    def filtered(self, filter_func):
        """
        return a generator that returns only members of this BlockSet
        that return true when passed to filter_func
        """
        for block in self:
            if filter_func(block):
                yield block

    def apply_filter(self, filter_func):
        """
        return a generator that returns only members of this BlockSet
        that return true when passed to filter_func
        """
        for block in self.copy():
            if not filter_func(block):
                self.remove(block)
