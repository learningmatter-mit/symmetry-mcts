import logging
import operator

from molgen.blocks import protect, block
from molgen.blocks.reactionenvironment import RDKitReactionEnvironment, LazyMessage


class SingleReactantRxnEnv(RDKitReactionEnvironment):
    """
    Reaction environment for single reactants, needed for cleaning
    """

    @protect.honor_protected_atoms
    def react(self, left_block):
        """
        Reacts the two molecules together and returns only the left side.  Currently takes smiles strings
        but this is likely to be converted to inchi in the near future for
        consistancy
        """
        if self.rxn is None:
            raise Exception("Unable to call react with no smarts")
        mol_1 = left_block.mol
        prods = self.rxn.RunReactants((mol_1, ))
        # this is specific to our kind of smarts string where we only want left reactants.
        first_prods = set([prod[0] for prod in prods])
        try:
            blocks = [block.Block(mol=product) for product in first_prods]
        except ValueError:
            return []
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            for b in blocks:
                logging.debug("%s => %s",
                              LazyMessage(left_block.smiles, True),
                              LazyMessage(b.smiles, True))
        return blocks

    def __str__(self):
        return "SingleReactantRxnEnv({})".format(self.smarts)
