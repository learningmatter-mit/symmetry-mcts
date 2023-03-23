import logging
import operator

from rdkit.Chem.AllChem import ReactionFromSmarts

from blocks import protect, reactivesite, block
from blocks.reactionenvironment import ReactionEnvironment, LazyMessage


class MultiSmartsRxnEnv(ReactionEnvironment):
    """
    Describes the type of reactions that occur between blocks.  Blocks keep a
    reference to this environment when they are reacted together
    Initialised with the smarts string to be used for the reaction.
    """

    def __init__(self, smarts, handles):
        """
        Initialization. accepts:

        smarts: a list of smarts strings
        handles: a list of reactive atoms (ie. ["Mg", "Fe"]).
        estimate_func:  a function used to estimate the number of products given
        input arguments matching a reaction
        """
        if type(smarts) is str:
            raise ValueError("smarts must be a list of string for MultiSmilesRxnEnv")
        ReactionEnvironment.__init__(self, smarts, handles)
        self.rxns = [ReactionFromSmarts(str(s)) for s in smarts]

    def estimate(self, left_block_site_count, right_block_site_count):
        # add to right side set to account for empty option on left
        return (right_block_site_count + 1) ** left_block_site_count - 1

    def estimate_sym(self, left_block_site_count, right_block_site_count):
        # add to right side set to account for empty option on left
        return right_block_site_count

    @protect.honor_protected_atoms
    def react(self, left_block, right_block):
        """
        Reacts the two molecules together and returns only the left side.  Currently takes smiles strings
        but this is likely to be converted to inchi in the near future for
        consistancy
        """

        mol_1 = left_block.mol
        mol_2 = right_block.mol
        prods = []
        for rxn in self.rxns:
            prods.extend(rxn.RunReactants((mol_1, mol_2)))
        prods = set([prod[0] for prod in prods])
        blocks = []
        for product in prods:
            try:
                blocks.append(block.Block(mol=product))
            except ValueError as e:
                logging.error(e)
                continue
        if logging.getLogger().getEffectiveLevel() >= logging.DEBUG:
            for b in blocks:
                logging.debug("%s + %s = %s",
                              LazyMessage(left_block.smiles(), False),
                              LazyMessage(right_block.smiles(), False),
                              LazyMessage(b.smiles(), False))
        return blocks

    def find_reactive_sites(self, block, reactant_index=0):
        """
        return list of tuples providing atom numbers of reactive sites
        for the current reaction environment

        reactant index indicates the index of the reactant according to
        the current SMARTS string.  Default is 0, meaning the leftmost
        reactant
        """
        sites = []
        for rxn in self.rxns:
            reactant_num = rxn.GetNumReactantTemplates()
            if reactant_index >= reactant_num:
                msg = "reactant_index {} greater than number "
                msg += " of reactants ({}) in current reaction {}"
                raise ValueError(msg.format(reactant_index, reactant_num, self.smarts))

            reactant_template = rxn.GetReactantTemplate(reactant_index)
            atom_indices = block.mol.GetSubstructMatches(reactant_template)
            for atuple in atom_indices:
                newsite = reactivesite.ReactiveSite(block, atuple)
                if newsite not in sites:
                    sites.append(newsite)
        return sites

    def __str__(self):
        return "MultiSmartsRxnEnv({})".format(self.smarts)


class MultiSmartsSingleRxnEnv(MultiSmartsRxnEnv):
    """
    Describes the type of reactions that occur between blocks.  Blocks keep a
    reference to this environment when they are reacted together
    Initialised with the smarts string to be used for the reaction.
    """

    @protect.honor_protected_atoms
    def react(self, left_block):
        """
        Reacts the two molecules together and returns only the left side.  Currently takes smiles strings
        but this is likely to be converted to inchi in the near future for
        consistancy
        """

        mol_1 = left_block.mol
        prods = []
        for rxn in self.rxns:
            prods.extend(rxn.RunReactants((mol_1, )))
        prods = set([prod[0] for prod in prods])
        blocks = [block.Block(mol=product) for product in prods]
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            for b in blocks:
                logging.debug("%s => %s",
                              LazyMessage(left_block.smiles, False),
                              LazyMessage(b.smiles, False))
        return blocks

    def __str__(self):
        return "MultiSmartsSingleRxnEnv({})".format(self.smarts)

